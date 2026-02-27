"""Tests for ISTFT reconstruction via ComplexSpectrogram.complex_mel_to_waveform.

Verifies the complete pipeline: 2-channel magnitude + IF spectrogram -> waveform
via phase reconstruction (cumulative sum of IF) and ISTFT.
"""

from __future__ import annotations

import torch
import pytest

from distill.audio.spectrogram import ComplexSpectrogram
from distill.training.config import ComplexSpectrogramConfig

SAMPLE_RATE = 48_000


@pytest.fixture()
def spec() -> ComplexSpectrogram:
    """Create a ComplexSpectrogram instance with default config."""
    return ComplexSpectrogram(ComplexSpectrogramConfig())


def _make_sine_wave(freq: float = 440.0, duration: float = 1.0) -> torch.Tensor:
    """Generate a sine wave as [1, 1, samples]."""
    t = torch.linspace(0, duration, int(SAMPLE_RATE * duration))
    wav = torch.sin(2 * torch.pi * freq * t)
    return wav.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]


def _make_white_noise(duration: float = 1.0) -> torch.Tensor:
    """Generate white noise as [1, 1, samples]."""
    samples = int(SAMPLE_RATE * duration)
    return torch.randn(1, 1, samples)


class TestISTFTReconstruction:
    """Test suite for complex_mel_to_waveform."""

    def test_reconstruction_output_shape(self, spec: ComplexSpectrogram) -> None:
        """Output shape is [B, 1, samples] with samples near input length."""
        wav = _make_sine_wave(440.0, 1.0)
        spectrogram = spec.waveform_to_complex_mel(wav)
        reconstructed = spec.complex_mel_to_waveform(spectrogram, sample_rate=SAMPLE_RATE)

        assert reconstructed.dim() == 3, f"Expected 3 dims, got {reconstructed.dim()}"
        assert reconstructed.shape[0] == 1, f"Expected batch=1, got {reconstructed.shape[0]}"
        assert reconstructed.shape[1] == 1, f"Expected channels=1, got {reconstructed.shape[1]}"
        # Samples should be approximately input length (within hop_length tolerance)
        hop = spec.config.hop_length
        assert abs(reconstructed.shape[2] - wav.shape[2]) < hop * 2, (
            f"Sample length {reconstructed.shape[2]} too far from input {wav.shape[2]}"
        )

    def test_reconstruction_no_nan_inf(self, spec: ComplexSpectrogram) -> None:
        """Reconstructed waveform must contain no NaN or Inf values."""
        wav = _make_sine_wave(440.0, 1.0)
        spectrogram = spec.waveform_to_complex_mel(wav)
        reconstructed = spec.complex_mel_to_waveform(spectrogram, sample_rate=SAMPLE_RATE)

        assert torch.isnan(reconstructed).sum() == 0, "Output contains NaN values"
        assert torch.isinf(reconstructed).sum() == 0, "Output contains Inf values"

    def test_reconstruction_reasonable_amplitude(self, spec: ComplexSpectrogram) -> None:
        """Output peak amplitude is non-silent and not exploded."""
        wav = _make_sine_wave(440.0, 1.0)
        spectrogram = spec.waveform_to_complex_mel(wav)
        reconstructed = spec.complex_mel_to_waveform(spectrogram, sample_rate=SAMPLE_RATE)

        peak = reconstructed.abs().max().item()
        assert peak > 0.001, f"Output is nearly silent (peak={peak})"
        assert peak < 10.0, f"Output amplitude exploded (peak={peak})"

    def test_round_trip_sine_wave(self, spec: ComplexSpectrogram) -> None:
        """Round-trip encode->reconstruct produces audio resembling original."""
        wav = _make_sine_wave(440.0, 1.0)
        spectrogram = spec.waveform_to_complex_mel(wav)
        reconstructed = spec.complex_mel_to_waveform(spectrogram, sample_rate=SAMPLE_RATE)

        # Trim to shorter length for comparison
        min_len = min(wav.shape[2], reconstructed.shape[2])
        original = wav[:, :, :min_len]
        recon = reconstructed[:, :, :min_len]

        mse = ((original - recon) ** 2).mean().item()
        # Mel-domain round-trip is lossy due to mel binning, so threshold is relaxed
        assert mse < 0.5, f"MSE {mse} exceeds threshold 0.5"

    def test_round_trip_with_normalization(self, spec: ComplexSpectrogram) -> None:
        """Round-trip with normalize/denormalize still produces valid audio."""
        wav = _make_sine_wave(440.0, 1.0)
        spectrogram = spec.waveform_to_complex_mel(wav)

        # Compute stats and normalize
        stats = spec.compute_dataset_statistics([spectrogram.squeeze(0)])
        normalized = spec.normalize(spectrogram, stats)

        # Reconstruct from normalized spectrogram (pass stats so method denormalizes)
        reconstructed = spec.complex_mel_to_waveform(
            normalized, stats=stats, sample_rate=SAMPLE_RATE,
        )

        assert torch.isnan(reconstructed).sum() == 0, "Output contains NaN after normalization round-trip"
        assert torch.isinf(reconstructed).sum() == 0, "Output contains Inf after normalization round-trip"
        peak = reconstructed.abs().max().item()
        assert peak > 0.001, f"Output is nearly silent after normalization round-trip (peak={peak})"
        assert peak < 10.0, f"Output amplitude exploded after normalization round-trip (peak={peak})"

    def test_reconstruction_white_noise(self, spec: ComplexSpectrogram) -> None:
        """White noise round-trip produces non-silent output with no NaN."""
        wav = _make_white_noise(1.0)
        spectrogram = spec.waveform_to_complex_mel(wav)
        reconstructed = spec.complex_mel_to_waveform(spectrogram, sample_rate=SAMPLE_RATE)

        assert torch.isnan(reconstructed).sum() == 0, "Output contains NaN for noise input"
        assert torch.isinf(reconstructed).sum() == 0, "Output contains Inf for noise input"
        peak = reconstructed.abs().max().item()
        assert peak > 0.001, f"Output is nearly silent for noise input (peak={peak})"

    def test_batch_reconstruction(self, spec: ComplexSpectrogram) -> None:
        """Batch of 2 different signals reconstructs with correct shape."""
        sine = _make_sine_wave(440.0, 1.0)
        noise = _make_white_noise(1.0)

        # Encode both
        spec_sine = spec.waveform_to_complex_mel(sine)
        spec_noise = spec.waveform_to_complex_mel(noise)

        # Trim time dimension to match (may differ slightly)
        min_time = min(spec_sine.shape[-1], spec_noise.shape[-1])
        spec_sine = spec_sine[..., :min_time]
        spec_noise = spec_noise[..., :min_time]

        # Stack into batch of 2
        batch = torch.cat([spec_sine, spec_noise], dim=0)  # [2, 2, n_mels, time]
        reconstructed = spec.complex_mel_to_waveform(batch, sample_rate=SAMPLE_RATE)

        assert reconstructed.shape[0] == 2, f"Expected batch=2, got {reconstructed.shape[0]}"
        assert reconstructed.shape[1] == 1, f"Expected channels=1, got {reconstructed.shape[1]}"
        assert reconstructed.dim() == 3, f"Expected 3 dims, got {reconstructed.dim()}"
