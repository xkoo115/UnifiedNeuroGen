"""Pooled EEG-to-haemoglobin response generator with protocol-specific heads."""

# Derived from scripts/practice_shared_hb_functional_20261001.py:90-140.
from __future__ import annotations

import warnings

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

TASKS = ["MI", "MA"]
DEFAULT_CONFIG = {
    "pca_rank": 128,
    "latent_rank": 32,
    "alpha": 1000.0,
    "seed": 1037,
    "encoder_rank": 32,
}


def _validate(features, tasks, targets=None):
    features = np.asarray(features, dtype=np.float64)
    tasks = np.asarray(tasks)
    if features.ndim != 2 or tasks.ndim != 1 or len(features) != len(tasks):
        raise ValueError("features must be [events, features] and tasks [events]")
    if not np.isfinite(features).all():
        raise ValueError("EEG features must be finite")
    if not np.isin(tasks, TASKS).all():
        raise ValueError("tasks must contain acquisition protocols MI or MA")
    if targets is None:
        return features, tasks
    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape != (len(features), 144) or not np.isfinite(targets).all():
        raise ValueError("targets must be finite [events, 144] Hb mean/slope features")
    if set(tasks) != set(TASKS):
        raise ValueError("fitting data must contain both MI and MA protocols")
    return features, tasks, targets


def fit(features, targets, tasks, config=None):
    """Fit a fixed configuration on fitting events without within-protocol labels."""
    features, tasks, targets = _validate(features, tasks, targets)
    settings = {**DEFAULT_CONFIG, **({} if config is None else dict(config))}
    if set(settings) != set(DEFAULT_CONFIG):
        raise ValueError("unknown haemoglobin generator configuration field")
    pca_rank = int(settings["pca_rank"])
    latent_rank = int(settings["latent_rank"])
    encoder_rank = int(settings["encoder_rank"])
    alpha = float(settings["alpha"])
    if not (1 <= latent_rank <= encoder_rank <= pca_rank):
        raise ValueError("require 1 <= latent_rank <= encoder_rank <= pca_rank")
    if pca_rank > min(features.shape) or alpha < 0:
        raise ValueError(
            "PCA rank exceeds fitting data dimensions or alpha is negative"
        )

    yscalers = {task: StandardScaler().fit(targets[tasks == task]) for task in TASKS}
    target = np.zeros((len(features), 288), dtype=np.float64)
    for j, task in enumerate(TASKS):
        ix = np.flatnonzero(tasks == task)
        target[ix, j * 144 : (j + 1) * 144] = yscalers[task].transform(targets[ix])
    xscaler = StandardScaler().fit(features)
    scaled = xscaler.transform(features)
    xpca = PCA(
        n_components=pca_rank,
        svd_solver="randomized",
        random_state=int(settings["seed"]),
    ).fit(scaled)
    z = xpca.transform(scaled)
    encoder = PLSRegression(
        n_components=encoder_rank, scale=False, max_iter=1000, tol=1e-6
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encoder.fit(z, target)
    score = encoder.transform(z)
    heads = {}
    for task in TASKS:
        mask = tasks == task
        heads[task] = Ridge(alpha=alpha).fit(
            score[mask, :latent_rank], yscalers[task].transform(targets[mask])
        )
    return {
        "xscaler": xscaler,
        "xpca": xpca,
        "encoder": encoder,
        "yscalers": yscalers,
        "heads": heads,
        "latent_rank": latent_rank,
        "PCA": pca_rank,
        "alpha": alpha,
        "arm": "paired",
        "config": settings,
    }


def predict(model, features, tasks):
    """Generate 144 Hb response coordinates from EEG and acquisition protocol."""
    features, tasks = _validate(features, tasks)
    z = model["xpca"].transform(model["xscaler"].transform(features))
    score = model["encoder"].transform(z)[:, : model["latent_rank"]]
    out = np.zeros((len(features), 144), dtype=np.float64)
    for task in TASKS:
        ix = np.flatnonzero(tasks == task)
        if not len(ix):
            continue
        out[ix] = model["yscalers"][task].inverse_transform(
            model["heads"][task].predict(score[ix])
        )
    return out
