import numpy as np
import pytest

from unifiedneurogen.data import (
    decode_target,
    encode_target,
    remove_legacy_vertices,
    restore_legacy_vertices,
    standardize_eeg,
)
from unifiedneurogen.validation import verify_manifest


def test_prepared_transform_preserves_full_rank_coordinates():
    rng = np.random.default_rng(31)
    target = rng.normal(size=(5, 4))
    transform = dict(
        vertex_mean=rng.normal(size=4),
        vertex_std=np.arange(1, 5),
        pca_mean=rng.normal(size=4),
        pca_components=np.eye(4),
        latent_mean=rng.normal(size=4),
        latent_std=np.arange(2, 6),
        feature_mean=np.zeros(3),
        feature_std=np.ones(3),
    )
    np.testing.assert_allclose(
        decode_target(encode_target(target, transform), transform),
        target,
        atol=2e-6,
        rtol=2e-6,
    )
    features = np.array([[1000.0, -1000.0, 7.0]])
    np.testing.assert_array_equal(standardize_eeg(features, transform), features)


def test_legacy_surface_index_and_scale_contract():
    eeg = np.ones((1, 2562, 32), dtype=np.float32)
    target = np.linspace(-5, 5, 2562, dtype=np.float32)[None]
    condition, encoded = remove_legacy_vertices(eeg, target)
    assert condition.shape == (1, 2560, 32)
    restored = restore_legacy_vertices(encoded[:, 0], physical_scale=True)
    keep = np.ones(2562, dtype=bool)
    keep[[2274, 2275]] = False
    np.testing.assert_allclose(restored[:, keep], target[:, keep], atol=1e-6)
    np.testing.assert_array_equal(restored[:, ~keep], 0)


def test_demo_manifest_rejects_corruption_and_path_escape(tmp_path):
    (tmp_path / "data").write_text("payload")
    (tmp_path / "SHA256SUMS").write_text("0" * 64 + "  data\n")
    with pytest.raises(ValueError, match="Checksum"):
        verify_manifest(tmp_path)
    (tmp_path / "SHA256SUMS").write_text("0" * 64 + "  ../outside\n")
    with pytest.raises(ValueError, match="Invalid manifest"):
        verify_manifest(tmp_path)
