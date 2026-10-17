import pytest
import torch

from unifiedneurogen import dit, gdit
from unifiedneurogen.diffusion import create_diffusion


@pytest.mark.parametrize("branch", [dit, gdit], ids=["fmri", "hb"])
def test_original_objective_backward_and_ddim_sampling(branch):
    torch.manual_seed(37)
    config = branch.DEMO_CONFIG
    model = branch.build_model(config)
    target = torch.randn(2, config["in_channels"], config["input_size"])
    condition = torch.randn(2, config["cond_channels"], config["cond_width"])
    diffusion = create_diffusion("4")
    timesteps = torch.tensor([0, 2])

    losses = diffusion.training_losses(
        model, target, timesteps, model_kwargs={"cond": condition}
    )
    assert set(losses) == {"mse", "vb", "loss"}
    torch.testing.assert_close(losses["loss"], losses["mse"] + losses["vb"])
    assert all(torch.isfinite(value).all() for value in losses.values())
    losses["loss"].mean().backward()
    gradient = model.final_layer.linear.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.step()
    model.eval()
    with torch.no_grad():
        samples = diffusion.ddim_sample_loop(
            model.forward,
            target.shape,
            noise=torch.randn_like(target),
            clip_denoised=False,
            model_kwargs={"c": condition},
            progress=False,
            device="cpu",
            eta=0.0,
        )
    assert samples.shape == target.shape
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize("branch", [dit, gdit], ids=["fmri", "hb"])
@pytest.mark.parametrize(
    "override, message",
    [
        ({"cond_channels": 7}, "cond_channels"),
        ({"cond_width": 7}, "cond_width"),
        ({"input_size": 9, "patch_size": 2}, "divisible by patch_size"),
        ({"input_size": 6, "patch_size": 2}, "even number"),
        (
            {"hidden_size": 18, "cond_width": 18, "num_heads": 4},
            "divisible by num_heads",
        ),
        ({"depth": 0}, "positive integer"),
        ({"learn_sigma": "yes"}, "boolean"),
        ({"unrecognized_field": 1}, "Unsupported"),
    ],
)
def test_rejects_configuration_incompatible_with_original_shapes(
    branch, override, message
):
    with pytest.raises(ValueError, match=message):
        branch.build_model({**branch.DEMO_CONFIG, **override})


@pytest.mark.parametrize("branch", [dit, gdit], ids=["fmri", "hb"])
def test_checkpoint_preserves_unused_original_value_projection(branch):
    model = branch.build_model(branch.DEMO_CONFIG)
    keys = model.state_dict()
    assert "blocks.0.cross_attn.v_proj.weight" in keys
    assert "blocks.0.cross_attn.k_proj.weight" in keys
    model.eval()
    config = branch.DEMO_CONFIG
    target = torch.randn(2, config["in_channels"], config["input_size"])
    condition = torch.randn(2, config["cond_channels"], config["cond_width"])
    with torch.no_grad():
        model.final_layer.linear.weight.normal_()
        before = model(target, torch.tensor([1, 2]), condition)
        model.blocks[0].cross_attn.v_proj.weight.add_(100)
        model.blocks[0].cross_attn.v_proj.bias.add_(100)
        after = model(target, torch.tensor([1, 2]), condition)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
