import json
import sys

import numpy as np
import pytest

from unifiedneurogen import cli, dit, gdit


@pytest.mark.parametrize("name, branch", [("fmri", dit), ("hb", gdit)])
def test_cli_train_then_sample_with_eeg_only(name, branch, tmp_path, monkeypatch):
    config = branch.DEMO_CONFIG
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    rng = np.random.default_rng(731)
    eeg = rng.normal(size=(6, config["cond_channels"], config["cond_width"])).astype(
        "float32"
    )
    target = rng.normal(size=(6, config["in_channels"], config["input_size"])).astype(
        "float32"
    )
    data_path = tmp_path / "pairs.npz"
    np.savez_compressed(
        data_path,
        eeg=eeg,
        target=target,
        split=np.array(["fit"] * 4 + ["tune"] * 2),
    )
    trained = tmp_path / "trained"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "unifiedneurogen",
            "--threads",
            "2",
            "dit-train",
            "--branch",
            name,
            "--data",
            str(data_path),
            "--config",
            str(config_path),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--output",
            str(trained),
        ],
    )
    cli.main()
    assert (trained / "dit.pt").is_file()
    training = json.loads((trained / "training.json").read_text())
    assert training["fitting_rows"] == [0, 1, 2, 3]
    assert training["epochs"] == 1
    assert len(training["history"]) == 1
    assert np.isfinite(training["history"][0]["loss"])

    eeg_path = tmp_path / "unpaired_eeg.npy"
    np.save(eeg_path, eeg[4:])
    generated_path = tmp_path / "generated.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "unifiedneurogen",
            "--threads",
            "2",
            "dit-sample",
            "--branch",
            name,
            "--eeg",
            str(eeg_path),
            "--checkpoint",
            str(trained / "dit.pt"),
            "--config",
            str(trained / "config.json"),
            "--steps",
            "4",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--output",
            str(generated_path),
        ],
    )
    cli.main()
    with np.load(generated_path, allow_pickle=False) as archive:
        generated = archive["generated"]
    assert generated.shape == (2, config["in_channels"], config["input_size"])
    assert np.isfinite(generated).all()
