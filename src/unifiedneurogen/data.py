"""Prepared-array transforms used by the M10 generation branch."""

from pathlib import Path

import numpy as np


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def finite_matrix(value, name):
    value = np.asarray(value)
    if value.ndim != 2 or len(value) == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return value


def standardize_eeg(features, transform):
    features = finite_matrix(features, "eeg_features")
    if features.shape[1:] != transform["feature_mean"].shape:
        raise ValueError("EEG feature dimension does not match the fitted transform")
    if np.any(transform["feature_std"] <= 0):
        raise ValueError("EEG feature scales must be positive")
    return ((features - transform["feature_mean"]) / transform["feature_std"]).astype(
        "float32"
    )


def encode_target(targets, transform):
    targets = finite_matrix(targets, "targets")
    surface = (targets - transform["vertex_mean"]) / transform["vertex_std"]
    coefficients = (surface - transform["pca_mean"]) @ transform["pca_components"].T
    return ((coefficients - transform["latent_mean"]) / transform["latent_std"]).astype(
        "float32"
    )


def decode_target(latent, transform):
    coefficients = (
        np.asarray(latent) * transform["latent_std"] + transform["latent_mean"]
    )
    return (
        coefficients @ transform["pca_components"] + transform["pca_mean"]
    ) * transform["vertex_std"] + transform["vertex_mean"]


def read_eeg(path, key="eeg_features"):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    with np.load(path, allow_pickle=False) as archive:
        return archive[key]


def split_indices(data):
    split = np.asarray(data["split"])
    if split.ndim != 1 or not set(split.tolist()) <= {"fit", "tune", "evaluate"}:
        raise ValueError("split must contain fit, tune or evaluate strings")
    fit, tune = (np.flatnonzero(split == role) for role in ("fit", "tune"))
    if len(fit) < 4 or not len(tune):
        raise ValueError(
            "At least four fitting samples and one tuning sample are required"
        )
    return fit, tune


def remove_legacy_vertices(eeg, target=None):
    """Prepare the original surface encoding; target normalization is (BOLD/5+1)/2."""
    eeg = np.asarray(eeg)
    if eeg.ndim != 3 or eeg.shape[1] != 2562:
        raise ValueError("Expected EEG encodings [samples, 2562, hidden_size]")
    condition = np.delete(eeg, [2274, 2275], axis=1).astype("float32")
    if target is None:
        return condition
    target = np.asarray(target)
    if target.shape != (len(eeg), 2562):
        raise ValueError("Expected paired BOLD [samples, 2562]")
    target = np.delete((target / 5 + 1) / 2, [2274, 2275], axis=1)
    return condition, target[:, None].astype("float32")


def restore_legacy_vertices(generated, physical_scale=False):
    generated = np.asarray(generated)
    if generated.shape[-1] != 2560:
        raise ValueError("Expected a 2560-value generated surface")
    if physical_scale:
        generated = (generated * 2 - 1) * 5
    generated = np.insert(generated, 2274, 0, axis=-1)
    return np.insert(generated, 2275, 0, axis=-1)
