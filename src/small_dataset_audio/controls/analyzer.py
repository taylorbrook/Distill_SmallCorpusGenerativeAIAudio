"""Latent space analysis engine.

Runs PCA on encoded training data, correlates PCA components with audio
features, and computes safe/warning ranges for slider controls.

The ``LatentSpaceAnalyzer`` orchestrates the full pipeline:
encode training data -> fit PCA -> correlate features -> compute ranges
-> build labels -> return ``AnalysisResult``.

Design notes:
- Lazy imports for torch, numpy, scipy, sklearn (project pattern).
- ``@torch.no_grad()`` context for all model inference.
- Model eval/train mode management: save ``was_training``, set eval, restore
  after (project pattern from ``inference/chunking.py``).
- Does NOT save sklearn PCA object -- stores numpy arrays only for
  checkpoint portability (research pitfall #3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Analysis result
# ---------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    """Complete result of latent space PCA analysis.

    Stores PCA decomposition, feature correlations, safe/warning ranges,
    and component labels.  All array fields are numpy ndarrays.

    This is a regular (mutable) dataclass because ``component_labels``
    may be updated by the user.  PCA arrays (``pca_components``,
    ``pca_mean``) should not be modified after construction.
    """

    pca_components: "np.ndarray"
    """Shape ``[n_active_components, latent_dim]`` -- PCA directions."""

    pca_mean: "np.ndarray"
    """Shape ``[latent_dim]`` -- mean of encoded data (latent space center)."""

    explained_variance_ratio: "np.ndarray"
    """Shape ``[n_active_components]`` -- fraction of variance per component."""

    n_active_components: int
    """Number of components above the variance threshold."""

    component_labels: list[str]
    """Current display labels (user-editable, start as 'Axis 1', 'Axis 2', ...)."""

    suggested_labels: list[str]
    """Auto-suggested labels from feature correlation (e.g. 'spectral_centroid')."""

    safe_min: "np.ndarray"
    """Shape ``[n_active_components]`` -- 2nd percentile of projected training data."""

    safe_max: "np.ndarray"
    """Shape ``[n_active_components]`` -- 98th percentile."""

    warning_min: "np.ndarray"
    """Shape ``[n_active_components]`` -- 0.5th percentile (soft warning zone)."""

    warning_max: "np.ndarray"
    """Shape ``[n_active_components]`` -- 99.5th percentile."""

    step_size: "np.ndarray"
    """Shape ``[n_active_components]`` -- safe range / (n_steps - 1)."""

    n_steps: int = 21
    """Number of discrete slider positions (default 21: integer range -10 to +10)."""

    feature_correlations: dict[str, list[float]] = field(default_factory=dict)
    """Maps feature name to list of Pearson correlations (one per component)."""

    latent_dim: int = 64
    """Original latent space dimensionality."""


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class LatentSpaceAnalyzer:
    """Analyze a trained VAE's latent space for musically meaningful controls.

    Orchestrates the full analysis pipeline: encode training data through
    the VAE, fit PCA on collected mu vectors, correlate PCA components
    with audio features, and compute safe/warning ranges for slider controls.

    Parameters
    ----------
    variance_threshold : float
        Minimum ``explained_variance_ratio`` for a PCA component to be
        considered active (default 0.02 = 2%).
    n_steps : int
        Number of discrete slider positions (default 21, giving integer
        range -10 to +10).
    n_sweep_points : int
        Number of points to sweep per component for feature correlation
        (default 20).
    """

    def __init__(
        self,
        variance_threshold: float = 0.02,
        n_steps: int = 21,
        n_sweep_points: int = 20,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.n_steps = n_steps
        self.n_sweep_points = n_sweep_points

    def analyze(
        self,
        model: "ConvVAE",
        dataloader: "DataLoader",
        spectrogram: "AudioSpectrogram",
        device: "torch.device",
        n_random_samples: int = 200,
    ) -> AnalysisResult:
        """Run the full latent space analysis pipeline.

        This is the main entry point, triggered explicitly by the user
        (locked decision: user-triggered analysis).

        Parameters
        ----------
        model : ConvVAE
            Trained VAE model.
        dataloader : DataLoader
            DataLoader yielding training data (raw waveforms).
        spectrogram : AudioSpectrogram
            Spectrogram converter for mel/waveform conversion.
        device : torch.device
            Device the model is on.
        n_random_samples : int
            Number of random prior samples to add to PCA input
            (default 200).

        Returns
        -------
        AnalysisResult
            Complete analysis result with PCA data, correlations,
            ranges, and labels.
        """
        import torch  # noqa: WPS433 -- lazy import
        import numpy as np  # noqa: WPS433 -- lazy import
        from sklearn.decomposition import PCA  # noqa: WPS433 -- lazy import

        # Save training state, switch to eval
        was_training = model.training
        model.eval()

        try:
            # ---------------------------------------------------------------
            # Step 1: Collect mu vectors
            # ---------------------------------------------------------------
            logger.info("Collecting encodings...")
            all_mu: list[np.ndarray] = []

            with torch.no_grad():
                for batch in dataloader:
                    # Dataloader returns raw waveforms -- convert to mel on device
                    if isinstance(batch, (list, tuple)):
                        waveforms = batch[0]
                    else:
                        waveforms = batch
                    waveforms = waveforms.to(device)

                    # Convert waveforms to mel spectrograms
                    # Waveforms shape: [B, 1, samples] or [B, samples]
                    if waveforms.dim() == 2:
                        waveforms = waveforms.unsqueeze(1)
                    mel = spectrogram.waveform_to_mel(waveforms)

                    # Encode to get mu vectors
                    mu, _logvar = model.encode(mel)
                    all_mu.append(mu.cpu().numpy())

            # Add random prior samples for full coverage
            latent_dim = model.latent_dim
            prior_samples = np.random.randn(n_random_samples, latent_dim).astype(
                np.float32,
            )

            if all_mu:
                training_mu = np.concatenate(all_mu, axis=0)
                mu_vectors = np.concatenate(
                    [training_mu, prior_samples], axis=0,
                )
            else:
                logger.warning("No training data encoded, using prior samples only")
                mu_vectors = prior_samples

            logger.info(
                "Collected %d latent vectors (%d training + %d prior)",
                len(mu_vectors),
                len(mu_vectors) - n_random_samples,
                n_random_samples,
            )

            # ---------------------------------------------------------------
            # Step 2: Fit PCA
            # ---------------------------------------------------------------
            max_components = min(20, latent_dim, len(mu_vectors))
            pca = PCA(n_components=max_components)
            pca.fit(mu_vectors)

            # Determine active component count
            n_active = int(
                np.sum(pca.explained_variance_ratio_ >= self.variance_threshold),
            )
            n_active = max(n_active, 1)  # Minimum 1 component (graceful degradation)

            total_variance = float(np.sum(pca.explained_variance_ratio_[:n_active]) * 100)
            logger.info(
                "Fitting PCA (%d components explain %.1f%% variance)...",
                n_active,
                total_variance,
            )

            if n_active < 3:
                logger.warning(
                    "Only %d active PCA component(s) found. "
                    "More training data or longer training may improve variety.",
                    n_active,
                )

            # Extract PCA data (numpy arrays only -- no sklearn object saved)
            pca_components = pca.components_[:n_active].copy()
            pca_mean = pca.mean_.copy()
            explained_variance_ratio = pca.explained_variance_ratio_[:n_active].copy()

            # ---------------------------------------------------------------
            # Step 3: Compute safe ranges
            # ---------------------------------------------------------------
            projected = pca.transform(mu_vectors)[:, :n_active]

            safe_min = np.percentile(projected, 2, axis=0)
            safe_max = np.percentile(projected, 98, axis=0)
            warning_min = np.percentile(projected, 0.5, axis=0)
            warning_max = np.percentile(projected, 99.5, axis=0)
            step_size = (safe_max - safe_min) / (self.n_steps - 1)

            # ---------------------------------------------------------------
            # Step 4: Feature correlation
            # ---------------------------------------------------------------
            logger.info("Computing feature correlations...")
            from scipy.stats import pearsonr  # noqa: WPS433 -- lazy import
            from small_dataset_audio.controls.features import (
                FEATURE_NAMES,
                compute_audio_features,
            )

            mel_shape = spectrogram.get_mel_shape(spectrogram.config.sample_rate)

            # Ensure decoder is initialised
            if model.decoder.fc is None:
                n_mels, time_frames = mel_shape
                pad_h = (16 - n_mels % 16) % 16
                pad_w = (16 - time_frames % 16) % 16
                spatial = (
                    (n_mels + pad_h) // 16,
                    (time_frames + pad_w) // 16,
                )
                model.decoder._init_linear(spatial)

            # feature_correlations: {feature_name: [r_per_component]}
            feature_correlations: dict[str, list[float]] = {
                name: [] for name in FEATURE_NAMES
            }
            # Track p-values for label suggestion
            feature_pvalues: dict[str, list[float]] = {
                name: [] for name in FEATURE_NAMES
            }

            with torch.no_grad():
                for comp_idx in range(n_active):
                    sweep_values = np.linspace(
                        safe_min[comp_idx],
                        safe_max[comp_idx],
                        self.n_sweep_points,
                    )

                    # Collect features at each sweep point
                    sweep_features: dict[str, list[float]] = {
                        name: [] for name in FEATURE_NAMES
                    }

                    for val in sweep_values:
                        # Construct latent vector: mean + value * component_i
                        z_np = pca_mean + val * pca_components[comp_idx]
                        z_tensor = (
                            torch.from_numpy(z_np)
                            .float()
                            .unsqueeze(0)
                            .to(device)
                        )

                        # Decode to mel, convert to waveform
                        mel_out = model.decode(z_tensor, target_shape=mel_shape)
                        wav = spectrogram.mel_to_waveform(mel_out.cpu())
                        wav_np = wav.squeeze().numpy().astype(np.float32)

                        # Compute audio features
                        features = compute_audio_features(
                            wav_np, spectrogram.config.sample_rate,
                        )
                        for name in FEATURE_NAMES:
                            sweep_features[name].append(features[name])

                    # Compute Pearson correlation for each feature
                    for name in FEATURE_NAMES:
                        values = np.array(sweep_features[name])
                        # Check for constant values (pearsonr would error)
                        if np.std(values) < 1e-12:
                            feature_correlations[name].append(0.0)
                            feature_pvalues[name].append(1.0)
                        else:
                            r, p = pearsonr(sweep_values, values)
                            feature_correlations[name].append(float(r))
                            feature_pvalues[name].append(float(p))

            # ---------------------------------------------------------------
            # Step 5: Build labels
            # ---------------------------------------------------------------
            default_labels = [f"Axis {i + 1}" for i in range(n_active)]
            suggested_labels: list[str] = []

            for comp_idx in range(n_active):
                best_feature: str | None = None
                best_abs_r = 0.0

                for name in FEATURE_NAMES:
                    r = feature_correlations[name][comp_idx]
                    p = feature_pvalues[name][comp_idx]

                    if abs(r) > 0.5 and p < 0.05 and abs(r) > best_abs_r:
                        best_feature = name
                        best_abs_r = abs(r)

                if best_feature is not None:
                    suggested_labels.append(best_feature)
                else:
                    suggested_labels.append(default_labels[comp_idx])

            # ---------------------------------------------------------------
            # Step 6: Construct result
            # ---------------------------------------------------------------
            result = AnalysisResult(
                pca_components=pca_components,
                pca_mean=pca_mean,
                explained_variance_ratio=explained_variance_ratio,
                n_active_components=n_active,
                component_labels=list(default_labels),
                suggested_labels=suggested_labels,
                safe_min=safe_min,
                safe_max=safe_max,
                warning_min=warning_min,
                warning_max=warning_max,
                step_size=step_size,
                n_steps=self.n_steps,
                feature_correlations=feature_correlations,
                latent_dim=latent_dim,
            )

            logger.info(
                "Analysis complete: %d active components",
                n_active,
            )
            return result

        finally:
            # Restore training state
            if was_training:
                model.train()
