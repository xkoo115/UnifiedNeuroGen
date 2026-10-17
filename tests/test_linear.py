import numpy as np
import pytest
from sklearn.linear_model import Ridge

from unifiedneurogen import linear


class TokenProjection:
    def transform(self, values):
        return values[:, :32]


class ContextScaler:
    def transform(self, values):
        self.observed = values.copy()
        return values


class IdentityTransform:
    def inverse_transform(self, values):
        return values


class FixedLogScale:
    def __init__(self, log_scale):
        self.log_scale = log_scale

    def predict(self, values):
        return np.broadcast_to(self.log_scale, (len(values), len(self.log_scale)))


def test_equal_person_ridge_matches_weighted_sklearn_solution():
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(20, 5)), rng.normal(size=(20, 3))
    subjects = np.array(["a"] * 3 + ["b"] * 5 + ["c"] * 12)
    weights = np.array([1 / 3] * 3 + [1 / 5] * 5 + [1 / 12] * 12)
    weights /= weights.mean()
    expected = Ridge(alpha=3.5, solver="cholesky").fit(x, y, sample_weight=weights)
    actual = linear.weighted_ridge_fit(x, y, subjects, 3.5)
    np.testing.assert_allclose(actual["weight"].T, expected.coef_, atol=1e-12)
    np.testing.assert_allclose(
        linear.ridge_predict(x, actual), expected.predict(x), atol=1e-12
    )


def test_m01_uses_whole_run_eeg_context_and_predicted_amplitude():
    rng = np.random.default_rng(7)
    history = rng.normal(size=(15, 8, 366))
    dynamic = {
        "input_mean": np.zeros(366),
        "input_std": np.ones(366),
        "pca": TokenProjection(),
        "rank": 8,
        "feature": "full_FIR",
        "target_mean": np.zeros(142),
        "target_std": np.ones(142),
        "param": {
            "input_mean": np.zeros(64),
            "target_mean": np.zeros(142),
            "weight": rng.normal(size=(64, 142)),
        },
    }
    scaler = ContextScaler()
    log_scale = np.linspace(-1, 0.5, 142)
    scale = {
        "impute_mean": np.zeros(732),
        "scaler": scaler,
        "pca": TokenProjection(),
        "scale": np.ones(32),
        "target_scaler": IdentityTransform(),
        "model": FixedLogScale(log_scale),
    }
    generated = linear.predict_m01(history, dynamic, scale)
    expected_context = np.r_[history[:, -1].mean(0), history[:, -1].std(0)][None]
    np.testing.assert_allclose(scaler.observed, expected_context)
    np.testing.assert_allclose(generated.mean(0), 0, atol=1e-14)
    np.testing.assert_allclose(generated.std(0), np.exp(log_scale), atol=1e-14)


def test_m02_preserves_standardized_coordinates_and_relative_operation():
    rng = np.random.default_rng(7)
    history = rng.normal(size=(10, 8, 366))
    parameters = {
        "input_mean": np.zeros(8),
        "target_mean": np.arange(142),
        "weight": rng.normal(size=(8, 142)),
    }
    model = {
        "input_mean": np.zeros(366),
        "input_std": np.ones(366),
        "pca": TokenProjection(),
        "selected": (8, "current", 1.0),
        "ridge": parameters,
        "target_mean": np.full(142, 100),
        "target_std": np.full(142, 100),
    }
    expected = history[:, -1, :8] @ parameters["weight"] + parameters["target_mean"]
    actual = linear.predict_m02(history, model)
    np.testing.assert_allclose(actual, expected)
    relative = linear.predict_m02(history, model, regional_relative=True)
    np.testing.assert_allclose(relative, expected - expected.mean(1, keepdims=True))
    np.testing.assert_allclose(relative.mean(1), 0, atol=1e-12)


def test_history_shape_is_checked():
    with pytest.raises(ValueError, match="physical EEG history"):
        linear.predict_m01(np.zeros((8, 366)), {"rank": 8, "feature": "current"}, {})
