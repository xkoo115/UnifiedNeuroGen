"""M01 dynamic/scale and M02 regional generation in prepared ROI coordinates."""

# Derived from run_functional_bold_practice_20261001.py and
# rescue_natview_oof_bold_20261001.py. Selection and readouts are external.
import numpy as np


def weighted_ridge_fit(features, targets, subjects, alpha):
    """Fit a fixed-penalty Ridge map with equal total weight per fitting person."""
    x, y = np.asarray(features, float), np.asarray(targets, float)
    _, inverse, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    weights /= weights.mean()
    xm = np.average(x, axis=0, weights=weights)
    ym = np.average(y, axis=0, weights=weights)
    a, b = x - xm, y - ym
    values, vectors = np.linalg.eigh(a.T @ (weights[:, None] * a))
    projected = vectors.T @ (a.T @ (weights[:, None] * b))
    return dict(
        input_mean=xm,
        target_mean=ym,
        weight=vectors @ (projected / (np.maximum(values, 0)[:, None] + alpha)),
    )


def ridge_predict(features, parameters):
    return (features - parameters["input_mean"]) @ parameters["weight"] + parameters[
        "target_mean"
    ]


def _features(history, model, rank, feature):
    history = np.asarray(history)
    if history.ndim != 3 or history.shape[1:] != (8, 366) or not len(history):
        raise ValueError("Expected nonempty physical EEG history [frames, 8, 366]")
    standardized = (history - model["input_mean"]) / model["input_std"]
    tokens = model["pca"].transform(np.nan_to_num(standardized).reshape(-1, 366))
    tokens = tokens.reshape(len(history), 8, 32)[:, :, :rank]
    if feature == "current":
        return tokens[:, -1]
    if feature == "fixed_double_gamma":
        return np.einsum("ntf,t->nf", tokens, model["hrf"])
    if feature == "full_FIR":
        return tokens.reshape(len(history), -1)
    raise ValueError(f"Unsupported history representation: {feature}")


def predict_m01(history, dynamic_head, scale_head):
    """Generate one complete retained run; its EEG supplies the offline run context."""
    features = _features(
        history, dynamic_head, dynamic_head["rank"], dynamic_head["feature"]
    )
    dynamics = (
        ridge_predict(features, dynamic_head["param"]) * dynamic_head["target_std"]
        + dynamic_head["target_mean"]
    )
    current = np.asarray(history)[:, -1]
    context = np.r_[np.nanmean(current, axis=0), np.nanstd(current, axis=0)][None]
    context = np.where(np.isfinite(context), context, scale_head["impute_mean"])
    context = np.clip(scale_head["scaler"].transform(context), -8, 8)
    context = scale_head["pca"].transform(context) / scale_head["scale"]
    log_sd = scale_head["target_scaler"].inverse_transform(
        scale_head["model"].predict(context)
    )[0]
    scale = np.exp(np.clip(log_sd, -20, 20))
    own_sd = dynamics.std(axis=0)
    shape = np.divide(
        dynamics - dynamics.mean(axis=0),
        own_sd,
        out=np.zeros_like(dynamics),
        where=own_sd > 1e-12,
    )
    return shape * scale


def predict_m02(history, model, regional_relative=False):
    """Generate fitting-standardized ROI values, optionally removing each frame's ROI mean."""
    rank, feature, _ = model["selected"]
    generated = ridge_predict(_features(history, model, rank, feature), model["ridge"])
    if regional_relative:
        generated = generated - generated.mean(axis=1, keepdims=True)
    return generated
