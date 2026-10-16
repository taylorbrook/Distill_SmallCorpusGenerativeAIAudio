"""Latent space analysis: slider axes must come from the training encodings."""

from __future__ import annotations

import numpy as np
import torch

from distill.controls.analyzer import LatentSpaceAnalyzer

LATENT_DIM = 16


class _StubDecoder:
    fc = object()  # "initialised"


class _StubModel:
    """Encoder returns the batch itself as ``mu``; decoder returns zeros."""

    latent_dim = LATENT_DIM
    training = False
    decoder = _StubDecoder()

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def encode(self, x):
        mu = x.squeeze(1)
        return mu, torch.zeros_like(mu)

    def decode(self, z, target_shape=None):
        return torch.zeros(z.shape[0], 1, 8, 8)


class _StubConfig:
    sample_rate = 48_000


class _StubSpectrogram:
    config = _StubConfig()

    def waveform_to_mel(self, waveforms):
        return waveforms

    def get_mel_shape(self, num_samples):
        return (8, 8)

    def mel_to_waveform(self, mel):
        return torch.randn(mel.shape[0], 1, 4800) * 0.1


def test_pca_recovers_the_training_data_axes():
    """Small-variance encodings (as with a low KL weight) must not be swamped.

    The old analyzer mixed 200 N(0, I) samples into the PCA input; with
    encodings this tight that made every component noise.
    """
    rng = np.random.default_rng(0)
    axis_1 = np.eye(LATENT_DIM)[3]
    axis_2 = np.eye(LATENT_DIM)[7]
    n = 40
    mu = (
        0.5
        + 0.30 * rng.standard_normal((n, 1)) * axis_1
        + 0.15 * rng.standard_normal((n, 1)) * axis_2
        + 0.01 * rng.standard_normal((n, LATENT_DIM))
    ).astype(np.float32)
    batches = [torch.from_numpy(mu[i : i + 8]).unsqueeze(1) for i in range(0, n, 8)]

    result = LatentSpaceAnalyzer().analyze(
        _StubModel(),
        batches,
        _StubSpectrogram(),
        torch.device("cpu"),
    )

    assert result.n_active_components == 2
    assert abs(result.pca_components[0] @ axis_1) > 0.99
    assert abs(result.pca_components[1] @ axis_2) > 0.95
    np.testing.assert_allclose(result.pca_mean, mu.mean(axis=0), atol=1e-5)
    # Safe ranges reflect the data's spread, not the N(0, 1) prior's
    assert result.safe_max[0] - result.safe_min[0] < 2.0
