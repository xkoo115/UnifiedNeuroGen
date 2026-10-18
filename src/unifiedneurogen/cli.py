"""Training and sampling commands for EEG-conditioned generation."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .data import (
    decode_target,
    encode_target,
    load_npz,
    read_eeg,
    split_indices,
    standardize_eeg,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def new_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def state_from_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "ema" in state:
        return state["ema"]
    if "model" in state:
        return state["model"]
    return state


@torch.no_grad()
def m10_generate(
    eeg,
    transform_path,
    checkpoint,
    config_path,
    device="cpu",
    draws=1,
    seed=7,
    batch_size=64,
):
    from .m10 import build_model

    if draws < 0 or batch_size < 1:
        raise ValueError("draws must be nonnegative and batch_size positive")
    transform = load_npz(transform_path)
    x = standardize_eeg(eeg, transform)
    state = state_from_checkpoint(checkpoint)
    model = build_model(
        read_json(config_path), state.get("baseline_weight"), state.get("baseline_bias")
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    torch.manual_seed(seed)
    means, variances, representations, generated = [], [], [], []
    for start in range(0, len(x), batch_size):
        features = torch.from_numpy(x[start : start + batch_size, None]).to(device)
        output = model(features, sample=False)
        means.append(output.mean.cpu().numpy())
        variance = torch.exp(2 * output.log_std)
        if output.covariance_factor is not None:
            variance = variance + output.covariance_factor.square().sum(-1)
        variances.append(variance.cpu().numpy())
        representations.append(output.eeg_representation.cpu().numpy())
        if draws:
            generated.append(
                np.stack(
                    [
                        model(features, sample=True).sample.cpu().numpy()
                        for _ in range(draws)
                    ]
                )
            )
    mean = np.concatenate(means)
    result = dict(
        mean=mean,
        marginal_variance=np.concatenate(variances),
        activity=np.concatenate(representations),
        surface_mean=decode_target(mean, transform),
        hrf_weights=output.hrf_weights.cpu().numpy(),
    )
    if draws:
        result["samples"] = np.concatenate(generated, axis=1)
    return result


def sample_m10(args):
    result = m10_generate(
        read_eeg(args.eeg),
        args.transform,
        args.checkpoint,
        args.config,
        args.device,
        args.draws,
        args.seed,
        args.batch_size,
    )
    destination = Path(args.output)
    if destination.exists():
        raise ValueError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **result)
    print(f"Generated {len(result['mean'])} EEG-conditioned targets: {destination}")


def train_m10(args):
    from sklearn.cluster import KMeans
    from sklearn.linear_model import Ridge
    from .m10 import build_model, factorized_condition_loss, low_rank_gaussian_nll

    folder = new_output(args.output)
    data, transform, config = (
        load_npz(args.data),
        load_npz(args.transform),
        read_json(args.config),
    )
    fit, tune = split_indices(data)
    x = standardize_eeg(data["eeg_features"], transform)
    y = encode_target(data["targets"], transform)
    if len(x) != len(y) or len(x) != len(data["split"]):
        raise ValueError("EEG, targets and split lengths must agree")
    anchor = Ridge(alpha=args.ridge_alpha, solver="lsqr", tol=1e-5).fit(x[fit], y[fit])
    cluster = KMeans(
        n_clusters=config["condition_count"], n_init=20, random_state=20260830
    ).fit(y[fit, :8])
    conditions = cluster.predict(y[:, :8]).astype(np.int64)
    torch.manual_seed(args.seed)
    model = build_model(
        config,
        torch.from_numpy(anchor.coef_.astype("float32")),
        torch.from_numpy(anchor.intercept_.astype("float32")),
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-4)
    rng = np.random.default_rng(args.seed)

    @torch.no_grad()
    def tune_mse():
        model.eval()
        values = [
            model(torch.from_numpy(x[idx, None]).to(args.device), sample=False)
            .mean.cpu()
            .numpy()
            for idx in np.array_split(
                tune, max(1, int(np.ceil(len(tune) / args.batch_size)))
            )
        ]
        return float(np.mean((np.concatenate(values) - y[tune]) ** 2))

    best_mse, best_epoch, stale = tune_mse(), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    history = [dict(epoch=0, tune_mse=best_mse)]
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        sections = min(int(np.ceil(len(fit) / args.batch_size)), len(fit) // 2)
        for idx in np.array_split(rng.permutation(fit), sections):
            bx = torch.from_numpy(x[idx, None]).to(args.device)
            by = torch.from_numpy(y[idx]).to(args.device)
            bc = torch.from_numpy(conditions[idx]).to(args.device)
            output = model(bx, sample=False, target_for_condition=by)
            loss = low_rank_gaussian_nll(output, by) + factorized_condition_loss(
                output, model.encode_target_activity(by), bc
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite M10 training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        score = tune_mse()
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), tune_mse=score))
        if score < best_mse - 1e-6:
            best_mse, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        print(f"epoch={epoch} loss={np.mean(losses):.5f} tune_mse={score:.5f}")
        if stale >= args.patience:
            break
    torch.save(
        {key: value.cpu() for key, value in best_state.items()}, folder / "m10.pt"
    )
    np.savez_compressed(folder / "transform.npz", **transform)
    np.savez_compressed(
        folder / "condition_centers.npz", centers=cluster.cluster_centers_
    )
    write_json(folder / "config.json", config)
    write_json(
        folder / "training.json",
        dict(
            seed=args.seed,
            best_epoch=best_epoch,
            best_tune_mse=best_mse,
            ridge_alpha=args.ridge_alpha,
            ridge_solver="lsqr",
            ridge_tolerance=1e-5,
            condition_seed=20260830,
            condition_n_init=20,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            fitting_rows=fit.tolist(),
            tuning_rows=tune.tolist(),
            history=history,
            scope="M10 refit using supplied fit-participant transformations.",
        ),
    )


def train_dit(args):
    from .diffusion import create_diffusion

    if args.branch == "hb":
        from .gdit import build_model
    else:
        from .dit import build_model

    folder = new_output(args.output)
    config, data = read_json(args.config), load_npz(args.data)
    fit, _ = split_indices(data)
    torch.manual_seed(args.seed)
    model = build_model(config).to(args.device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    diffusion = create_diffusion("", learn_sigma=model.learn_sigma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for idx in np.array_split(
            rng.permutation(fit), int(np.ceil(len(fit) / args.batch_size))
        ):
            clean = torch.as_tensor(
                data["target"][idx], device=args.device, dtype=torch.float32
            )
            condition = torch.as_tensor(
                data["eeg"][idx], device=args.device, dtype=torch.float32
            )
            timestep = torch.randint(
                0, diffusion.num_timesteps, (len(idx),), device=args.device
            )
            loss = diffusion.training_losses(
                model, clean, timestep, model_kwargs={"cond": condition}
            )["loss"].mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite DiT training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for average, parameter in zip(ema.parameters(), model.parameters()):
                    average.mul_(0.9999).add_(parameter, alpha=0.0001)
            losses.append(float(loss.detach().cpu()))
        history.append(dict(epoch=epoch, loss=float(np.mean(losses))))
        print(f"epoch={epoch} loss={np.mean(losses):.5f}")
    torch.save(
        dict(
            model={k: v.cpu() for k, v in model.state_dict().items()},
            ema={k: v.cpu() for k, v in ema.state_dict().items()},
        ),
        folder / "dit.pt",
    )
    write_json(folder / "config.json", config)
    write_json(
        folder / "training.json",
        dict(
            seed=args.seed,
            epochs=args.epochs,
            history=history,
            fitting_rows=fit.tolist(),
            selection="Fixed final training epoch; no evaluation-target selection",
        ),
    )


@torch.no_grad()
def sample_dit(args):
    from .diffusion import create_diffusion

    if args.branch == "hb":
        from .gdit import build_model
    else:
        from .dit import build_model

    destination = Path(args.output)
    if destination.exists():
        raise ValueError(f"Output already exists: {destination}")
    model = build_model(read_json(args.config)).to(args.device).eval()
    model.load_state_dict(state_from_checkpoint(args.checkpoint), strict=True)
    eeg = read_eeg(args.eeg, key="eeg")
    if eeg.ndim != 3 or len(eeg) == 0 or not np.isfinite(eeg).all():
        raise ValueError(
            "Expected finite EEG encodings [samples, condition_channels, hidden_size]"
        )
    torch.manual_seed(args.seed)
    diffusion = create_diffusion(str(args.steps), learn_sigma=model.learn_sigma)
    generated = []
    for start in range(0, len(eeg), args.batch_size):
        condition = torch.as_tensor(
            eeg[start : start + args.batch_size],
            device=args.device,
            dtype=torch.float32,
        )
        length = model.x_embedder.length_size
        noise = torch.randn(
            len(condition), model.in_channels, length, device=args.device
        )
        value = diffusion.ddim_sample_loop(
            model,
            noise.shape,
            noise=noise,
            clip_denoised=False,
            model_kwargs={"c": condition},
            progress=False,
            device=args.device,
            eta=0.0,
        )
        generated.append(value.cpu().numpy())
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, generated=np.concatenate(generated))
    print(f"Saved DDIM samples: {destination}")


def train_hb(args):
    import joblib
    from .hb import fit

    folder = new_output(args.output)
    data = load_npz(args.data)
    indices, _ = split_indices(data)
    config = read_json(args.config)
    model = fit(
        data["eeg_features"][indices],
        data["targets"][indices],
        data["tasks"][indices],
        config,
    )
    joblib.dump(model, folder / "hb.joblib")
    write_json(folder / "config.json", config)
    write_json(
        folder / "training.json",
        dict(
            fitting_rows=indices.tolist(), selection="User-supplied fixed configuration"
        ),
    )


def sample_hb(args):
    import joblib
    from .hb import predict

    destination = Path(args.output)
    if destination.exists():
        raise ValueError(f"Output already exists: {destination}")
    data = load_npz(args.data)
    generated = predict(
        joblib.load(args.checkpoint), data["eeg_features"], data["tasks"]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, generated=generated)


def demo(args):
    from .validation import verify_manifest

    root, output = Path(args.data_dir), new_output(args.output)
    verify_manifest(root)
    result = m10_generate(
        read_eeg(root / "samples.npz"),
        root / "transforms/fold_transform.npz",
        root / "weights/m10.pt",
        root / "configs/m10.json",
        args.device,
        draws=2,
        seed=args.seed,
    )
    reference = np.load(root / "reference_latent_mean.npy", allow_pickle=False)
    np.testing.assert_allclose(result["mean"], reference, atol=2e-5, rtol=1e-5)
    np.savez_compressed(output / "generated.npz", **result)
    report = dict(
        status="PASS",
        model="M10 historical fold-01 seed-7 checkpoint",
        samples=len(result["mean"]),
        latent_shape=list(result["mean"].shape),
        surface_shape=list(result["surface_mean"].shape),
        finite=all(bool(np.isfinite(value).all()) for value in result.values()),
        maximum_reference_error=float(np.max(np.abs(result["mean"] - reference))),
        target_required_at_inference=False,
        scope="Generation demonstration; no manuscript metric estimation",
    )
    if not report["finite"]:
        raise ValueError("Nonfinite demo generation")
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("demo", help="Run the separately distributed M10 example")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output", default="outputs/demo")
    p.set_defaults(function=demo)
    for name, function in [("m10-sample", sample_m10), ("dit-sample", sample_dit)]:
        p = commands.add_parser(name)
        for option in ("eeg", "checkpoint", "config", "output"):
            p.add_argument(f"--{option}", required=True)
        p.add_argument("--batch-size", type=int, default=16)
        if name == "m10-sample":
            p.add_argument("--transform", required=True)
            p.add_argument("--draws", type=int, default=1)
        else:
            p.add_argument("--steps", type=int, default=50)
            p.add_argument("--branch", choices=["fmri", "hb"], default="fmri")
        p.set_defaults(function=function)
    for name, function in [
        ("m10-train", train_m10),
        ("dit-train", train_dit),
        ("hb-train", train_hb),
    ]:
        p = commands.add_parser(name)
        for option in ("data", "config", "output"):
            p.add_argument(f"--{option}", required=True)
        if name != "hb-train":
            p.add_argument("--epochs", type=int, default=12)
            p.add_argument(
                "--batch-size", type=int, default=1024 if name == "m10-train" else 256
            )
        if name == "m10-train":
            p.add_argument("--transform", required=True)
            p.add_argument("--patience", type=int, default=3)
            p.add_argument("--ridge-alpha", type=float, default=1000.0)
        if name == "dit-train":
            p.add_argument("--branch", choices=["fmri", "hb"], default="fmri")
        p.set_defaults(function=function)
    p = commands.add_parser("hb-sample")
    for option in ("data", "checkpoint", "output"):
        p.add_argument(f"--{option}", required=True)
    p.set_defaults(function=sample_hb)
    for command_parser in commands.choices.values():
        command_parser.add_argument(
            "--device", default="cpu", choices=["cpu", "cuda", "mps"]
        )
        command_parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if (
        args.threads < 1
        or getattr(args, "batch_size", 1) < 1
        or getattr(args, "epochs", 1) < 1
    ):
        parser.error("threads, batch size and epochs must be positive")
    if args.command == "m10-train" and (
        args.batch_size < 2 or args.patience < 1 or args.ridge_alpha <= 0
    ):
        parser.error(
            "M10 batch size must be at least two; patience and Ridge alpha must be positive"
        )
    torch.set_num_threads(args.threads)
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=args.threads):
        args.function(args)


if __name__ == "__main__":
    main()
