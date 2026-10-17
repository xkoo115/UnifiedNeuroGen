import numpy as np
import pytest
import torch

from unifiedneurogen import hb, m10


def small_model():
    return m10.build_model(
        {
            "feature_dim": 24,
            "token_features": 8,
            "time_tokens": 3,
            "target_dim": 5,
            "condition_count": 3,
            "width": 8,
            "heads": 2,
            "depth": 1,
            "hrf_banks": 2,
            "query_tokens": 2,
            "activity_dim": 4,
            "context_dim": 4,
            "covariance_rank": 2,
            "dropout": 0.0,
        }
    )


def test_low_rank_nll_matches_multivariate_normal():
    torch.manual_seed(7)
    mean = torch.randn(3, 5, dtype=torch.float64)
    log_std = torch.randn(3, 5, dtype=torch.float64) * 0.2
    factor = torch.randn(3, 5, 2, dtype=torch.float64) * 0.1
    target = torch.randn(3, 5, dtype=torch.float64)
    output = m10.LatentDecoderOutput(
        mean, log_std, factor, mean, torch.empty(3, 4), torch.empty(2, 3)
    )
    covariance = torch.diag_embed(torch.exp(2 * log_std)) + factor @ factor.mT
    expected = -torch.distributions.MultivariateNormal(
        mean, covariance_matrix=covariance
    ).log_prob(target)
    torch.testing.assert_close(
        m10.low_rank_gaussian_nll(output, target, reduction="none"), expected
    )


def test_temporal_weights_receive_gradient():
    torch.manual_seed(7)
    model = small_model()
    output = model(torch.randn(4, 1, 24), sample=False)
    output.mean.square().mean().backward()
    assert model.backbone.hrf_logits.grad.abs().max() > 0
    torch.testing.assert_close(output.hrf_weights.sum(-1), torch.ones(2))


def test_inference_uses_no_target_encoder():
    model = small_model().eval()

    def forbidden_target_encoder(target):
        raise AssertionError("target encoder must not run during EEG-only inference")

    model.encode_target_activity = forbidden_target_encoder
    with torch.no_grad():
        output = model(torch.randn(4, 1, 24), sample=False)
    assert output.mean.shape == (4, 5)
    assert output.target_condition_logits is None
    torch.testing.assert_close(output.mean, output.sample)


def test_hb_single_protocol_and_unknown_protocol():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(60, 12))
    targets = rng.normal(size=(60, 144))
    tasks = np.array(["MI", "MA"] * 30)
    model = hb.fit(
        features,
        targets,
        tasks,
        {
            "pca_rank": 8,
            "encoder_rank": 4,
            "latent_rank": 3,
            "alpha": 1.0,
            "seed": 1037,
        },
    )
    selected = tasks == "MI"
    actual = hb.predict(model, features[selected], tasks[selected])
    expected = hb.predict(model, features, tasks)[selected]
    np.testing.assert_allclose(actual, expected)
    assert actual.shape == (30, 144)
    with pytest.raises(ValueError, match="acquisition protocols"):
        hb.predict(model, features[:1], np.array(["unknown"]))
