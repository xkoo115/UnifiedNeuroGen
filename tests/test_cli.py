"""Regression checks for M10 training labels and small final batches."""

import json
import sys

import numpy as np
import pytest

from unifiedneurogen.cli import main


@pytest.mark.parametrize("fitting_rows", [4, 5])
def test_small_m10_training_and_eeg_only_sampling(tmp_path, monkeypatch, fitting_rows):
    rng = np.random.default_rng(112)
    rows, feature_dim, target_dim = fitting_rows + 3, 12, 4
    features = rng.normal(size=(rows, feature_dim)).astype("float32")
    targets = rng.normal(size=(rows, target_dim)).astype("float32")
    roles = np.array(["fit"] * fitting_rows + ["tune", "tune", "evaluate"])
    data = tmp_path / "pairs.npz"
    np.savez_compressed(data, eeg_features=features, targets=targets, split=roles)
    transform = tmp_path / "transform.npz"
    np.savez_compressed(
        transform,
        feature_mean=np.zeros(feature_dim),
        feature_std=np.ones(feature_dim),
        vertex_mean=np.zeros(target_dim),
        vertex_std=np.ones(target_dim),
        pca_mean=np.zeros(target_dim),
        pca_components=np.eye(target_dim),
        latent_mean=np.zeros(target_dim),
        latent_std=np.ones(target_dim),
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "feature_dim": feature_dim,
                "token_features": 4,
                "time_tokens": 3,
                "target_dim": target_dim,
                "condition_count": 2,
                "width": 8,
                "heads": 2,
                "depth": 1,
                "hrf_banks": 2,
                "query_tokens": 2,
                "activity_dim": 4,
                "context_dim": 4,
                "covariance_rank": 1,
                "dropout": 0.0,
            }
        )
    )
    trained = tmp_path / "trained"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "unifiedneurogen",
            "--threads",
            "1",
            "m10-train",
            "--data",
            str(data),
            "--transform",
            str(transform),
            "--config",
            str(config),
            "--output",
            str(trained),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ],
    )
    main()
    report = json.loads((trained / "training.json").read_text())
    assert report["fitting_rows"] == list(range(fitting_rows))
    assert report["tuning_rows"] == [fitting_rows, fitting_rows + 1]
    assert report["history"][-1]["epoch"] == 1
    assert np.isfinite(report["history"][-1]["loss"])

    eeg_only = tmp_path / "eeg_only.npz"
    np.savez_compressed(eeg_only, eeg_features=features[-1:])
    generated = tmp_path / "generated.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "unifiedneurogen",
            "--threads",
            "1",
            "m10-sample",
            "--eeg",
            str(eeg_only),
            "--transform",
            str(trained / "transform.npz"),
            "--checkpoint",
            str(trained / "m10.pt"),
            "--config",
            str(trained / "config.json"),
            "--output",
            str(generated),
            "--draws",
            "1",
            "--device",
            "cpu",
        ],
    )
    main()
    with np.load(generated, allow_pickle=False) as result:
        assert result["mean"].shape == (1, target_dim)
        assert result["samples"].shape == (1, 1, target_dim)
        assert result["surface_mean"].shape == (1, target_dim)
        assert all(np.isfinite(result[key]).all() for key in result.files)
