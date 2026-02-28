"""HiFi-GAN vocoder training loop with cancel/resume and data augmentation.

Provides :class:`VocoderTrainer` which encapsulates the complete GAN
training pipeline for per-model HiFi-GAN V2 vocoder training:

- Data loading from audio directory with random segment extraction
- Mel computation via the model's own :class:`AudioSpectrogram`
- Alternating generator/discriminator GAN training
- Discriminator input augmentation (random gain + noise injection)
- Cancel-safe checkpoint saving into ``.distillgan`` model files
- Resume from checkpoint with full state restoration

Callback events:
- :class:`VocoderEpochMetrics` -- emitted after each epoch
- :class:`VocoderPreviewEvent` -- emitted periodically with preview audio
- :class:`VocoderTrainingCompleteEvent` -- emitted on normal completion

Reference: https://arxiv.org/abs/2010.05646
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callback event dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VocoderEpochMetrics:
    """Emitted after each vocoder training epoch.

    Attributes
    ----------
    epoch : int
        Current epoch (0-indexed).
    total_epochs : int
        Total planned epochs.
    gen_loss : float
        Generator loss for this epoch.
    disc_loss : float
        Discriminator loss for this epoch.
    mel_loss : float
        Mel reconstruction loss for this epoch.
    feature_loss : float
        Feature matching loss for this epoch.
    learning_rate : float
        Current generator learning rate.
    eta_seconds : float
        Estimated time remaining in seconds.
    elapsed_seconds : float
        Total wall-clock time elapsed since training start.
    """

    epoch: int
    total_epochs: int
    gen_loss: float
    disc_loss: float
    mel_loss: float
    feature_loss: float
    learning_rate: float
    eta_seconds: float
    elapsed_seconds: float


@dataclass
class VocoderPreviewEvent:
    """Emitted when a preview audio sample is generated during training.

    Attributes
    ----------
    epoch : int
        Epoch at which the preview was generated.
    audio : np.ndarray
        Preview waveform as float32 array, shape ``[samples]``.
    sample_rate : int
        Sample rate of the preview audio.
    """

    epoch: int
    audio: np.ndarray
    sample_rate: int


@dataclass
class VocoderTrainingCompleteEvent:
    """Emitted when vocoder training completes normally.

    Attributes
    ----------
    epochs_completed : int
        Total number of epochs completed.
    final_gen_loss : float
        Generator loss at the final epoch.
    final_disc_loss : float
        Discriminator loss at the final epoch.
    model_path : Path
        Path to the ``.distillgan`` model file with saved vocoder state.
    """

    epochs_completed: int
    final_gen_loss: float
    final_disc_loss: float
    model_path: Path


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class _VocoderDataset:
    """PyTorch-style dataset that loads audio files and extracts random segments.

    Each ``__getitem__`` call returns a ``(mel_segment, waveform_segment)``
    pair where mel is computed through the model's AudioSpectrogram
    (VAE log1p format).

    Parameters
    ----------
    audio_paths : list[Path]
        Paths to audio files.
    spectrogram : AudioSpectrogram
        The model's spectrogram transform for mel computation.
    segment_size : int
        Number of waveform samples per segment.
    sample_rate : int
        Expected sample rate (files are resampled if needed).
    """

    def __init__(
        self,
        audio_paths: list[Path],
        spectrogram: "AudioSpectrogram",
        segment_size: int,
        sample_rate: int,
    ) -> None:
        import torch
        import torchaudio

        self._torch = torch
        self._torchaudio = torchaudio
        self._spectrogram = spectrogram
        self._segment_size = segment_size
        self._sample_rate = sample_rate

        # Pre-load and cache all audio (small datasets: 5-50 files)
        self._waveforms: list[torch.Tensor] = []
        for p in audio_paths:
            wav, sr = torchaudio.load(str(p))
            # Convert to mono if needed
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            # Resample if needed
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            # Normalize to [-1, 1]
            peak = wav.abs().max()
            if peak > 0:
                wav = wav / peak
            # Only keep files long enough for at least one segment
            if wav.shape[-1] >= segment_size:
                self._waveforms.append(wav)

        if not self._waveforms:
            raise ValueError(
                f"No audio files long enough for segment_size={segment_size} "
                f"({segment_size / sample_rate:.2f}s). "
                f"Checked {len(audio_paths)} files."
            )

        logger.info(
            "Loaded %d audio files (%d usable) for vocoder training",
            len(audio_paths),
            len(self._waveforms),
        )

    def __len__(self) -> int:
        return len(self._waveforms)

    def __getitem__(self, idx: int) -> tuple:
        torch = self._torch
        wav = self._waveforms[idx]

        # Random segment extraction
        max_start = wav.shape[-1] - self._segment_size
        start = torch.randint(0, max_start + 1, (1,)).item()
        segment = wav[:, start : start + self._segment_size]  # [1, segment_size]

        # Compute mel via AudioSpectrogram (VAE log1p format)
        # Input: [B, 1, samples], Output: [B, 1, n_mels, T]
        with torch.no_grad():
            mel = self._spectrogram.waveform_to_mel(
                segment.unsqueeze(0)
            )  # [1, 1, 128, T]
            mel = mel.squeeze(0)  # [1, 128, T]

        return mel, segment  # [1, 128, T], [1, segment_size]


# ---------------------------------------------------------------------------
# Discriminator augmentation
# ---------------------------------------------------------------------------


def _augment_disc_input(
    waveform: "torch.Tensor",
    gain_db_range: float = 3.0,
    noise_snr_range: tuple[float, float] = (30.0, 50.0),
) -> "torch.Tensor":
    """Apply data augmentation to discriminator inputs.

    Applies random gain variation and noise injection to prevent
    discriminator overfitting on small datasets. Applied to both
    real and fake audio equally.

    Parameters
    ----------
    waveform : torch.Tensor
        Audio tensor, shape ``[B, 1, T]``.
    gain_db_range : float
        Maximum gain variation in dB (applied uniformly +/-).
    noise_snr_range : tuple[float, float]
        SNR range in dB for additive noise (min_snr, max_snr).

    Returns
    -------
    torch.Tensor
        Augmented audio (same shape).
    """
    import torch

    B = waveform.shape[0]
    device = waveform.device

    # Random gain: +/- gain_db_range dB per sample in batch
    gain_db = (
        torch.rand(B, 1, 1, device=device) * 2 * gain_db_range - gain_db_range
    )
    gain_linear = 10.0 ** (gain_db / 20.0)
    waveform = waveform * gain_linear

    # Random noise injection at random SNR
    min_snr, max_snr = noise_snr_range
    snr_db = torch.rand(B, 1, 1, device=device) * (max_snr - min_snr) + min_snr
    signal_power = waveform.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-8)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = torch.randn_like(waveform) * noise_power.sqrt()
    waveform = waveform + noise

    return waveform


# ---------------------------------------------------------------------------
# VocoderTrainer
# ---------------------------------------------------------------------------


class VocoderTrainer:
    """HiFi-GAN V2 vocoder trainer with cancel/resume support.

    Encapsulates the full GAN training loop including data pipeline,
    discriminator augmentation, checkpoint management, and callback
    events.

    Parameters
    ----------
    config : HiFiGANConfig
        Training and model configuration.
    device : str
        Device preference: ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """

    def __init__(
        self,
        config: "HiFiGANConfig",
        device: str = "auto",
    ) -> None:
        from distill.hardware.device import select_device

        self._config = config
        self._device = select_device(device)

    @property
    def config(self) -> "HiFiGANConfig":
        """Return the training configuration."""
        return self._config

    @property
    def device(self) -> "torch.device":
        """Return the active device."""
        return self._device

    def train(
        self,
        model_path: Path,
        audio_dir: Path,
        epochs: int = 100,
        callback: Callable | None = None,
        cancel_event: threading.Event | None = None,
        checkpoint: dict | None = None,
        preview_interval: int = 20,
    ) -> dict:
        """Train a HiFi-GAN vocoder and return vocoder_state dict.

        Parameters
        ----------
        model_path : Path
            Path to the ``.distillgan`` model file. Used to load the
            model's AudioSpectrogram for mel computation and as the
            save target for vocoder state.
        audio_dir : Path
            Directory containing training audio files (WAV/FLAC/MP3).
        epochs : int
            Total training epochs (default 100).
        callback : Callable | None
            Optional callback receiving :class:`VocoderEpochMetrics`,
            :class:`VocoderPreviewEvent`, or
            :class:`VocoderTrainingCompleteEvent` events.
        cancel_event : threading.Event | None
            If set, training stops after the current batch and saves
            a checkpoint into the model file.
        checkpoint : dict | None
            Resume checkpoint dict (from a previous cancelled run).
            Contains all training state for exact resumption.
        preview_interval : int
            Generate a preview audio sample every N epochs (default 20).

        Returns
        -------
        dict
            The vocoder_state dict saved into the model file.
        """
        import torch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import ExponentialLR
        from torch.utils.data import DataLoader

        from distill.models.persistence import load_model
        from distill.vocoder.hifigan.config import HiFiGANConfig
        from distill.vocoder.hifigan.discriminator import (
            MultiPeriodDiscriminator,
            MultiScaleDiscriminator,
        )
        from distill.vocoder.hifigan.generator import HiFiGANGenerator
        from distill.vocoder.hifigan.losses import (
            discriminator_loss,
            feature_loss,
            generator_loss,
        )

        config = self._config
        device = self._device

        # --- Load model for its AudioSpectrogram ---
        loaded = load_model(model_path, device="cpu")
        spectrogram = loaded.spectrogram.to(device)

        # --- Build data pipeline ---
        audio_dir = Path(audio_dir)
        audio_extensions = {".wav", ".flac", ".mp3", ".ogg", ".opus"}
        audio_paths = sorted(
            p
            for p in audio_dir.iterdir()
            if p.suffix.lower() in audio_extensions and p.is_file()
        )
        if not audio_paths:
            raise ValueError(f"No audio files found in {audio_dir}")

        dataset = _VocoderDataset(
            audio_paths=audio_paths,
            spectrogram=spectrogram,
            segment_size=config.segment_size,
            sample_rate=config.sample_rate,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,  # In-process for small datasets
        )

        # --- Create models ---
        generator = HiFiGANGenerator(config).to(device)
        mpd = MultiPeriodDiscriminator(config).to(device)
        msd = MultiScaleDiscriminator().to(device)

        # --- Create optimizers ---
        optim_g = AdamW(
            generator.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_b1, config.adam_b2),
        )
        optim_d = AdamW(
            list(mpd.parameters()) + list(msd.parameters()),
            lr=config.learning_rate * 0.5,
            betas=(config.adam_b1, config.adam_b2),
        )

        # --- Create schedulers ---
        sched_g = ExponentialLR(optim_g, gamma=config.lr_decay)
        sched_d = ExponentialLR(optim_d, gamma=config.lr_decay)

        # --- Resume from checkpoint ---
        start_epoch = 0
        if checkpoint is not None:
            ckpt = checkpoint.get("checkpoint")
            if ckpt is not None:
                generator.load_state_dict(ckpt["generator_state_dict"])
                mpd.load_state_dict(ckpt["mpd_state_dict"])
                msd.load_state_dict(ckpt["msd_state_dict"])
                optim_g.load_state_dict(ckpt["optim_g_state_dict"])
                optim_d.load_state_dict(ckpt["optim_d_state_dict"])
                sched_g.load_state_dict(ckpt["sched_g_state_dict"])
                sched_d.load_state_dict(ckpt["sched_d_state_dict"])
                start_epoch = ckpt["epoch"] + 1
                logger.info("Resumed vocoder training from epoch %d", start_epoch)

        # --- Training loop ---
        train_start = time.monotonic()
        final_gen_loss = 0.0
        final_disc_loss = 0.0

        for epoch in range(start_epoch, epochs):
            generator.train()
            mpd.train()
            msd.train()

            epoch_gen_loss = 0.0
            epoch_disc_loss = 0.0
            epoch_mel_loss = 0.0
            epoch_feat_loss = 0.0
            num_batches = 0

            for mel_batch, wav_batch in dataloader:
                mel_batch = mel_batch.to(device)  # [B, 1, 128, T]
                wav_batch = wav_batch.to(device)  # [B, 1, segment_size]

                # Prepare mel for generator: [B, 1, 128, T] -> [B, 128, T]
                mel_input = mel_batch.squeeze(1)

                # ===== Discriminator step =====
                y_g_hat = generator(mel_input)  # [B, 1, T*hop_size]

                # Match lengths (generator output may differ slightly)
                min_len = min(wav_batch.shape[-1], y_g_hat.shape[-1])
                wav_real = wav_batch[..., :min_len]
                wav_fake = y_g_hat[..., :min_len].detach()

                # Augment discriminator inputs
                wav_real_aug = _augment_disc_input(wav_real)
                wav_fake_aug = _augment_disc_input(wav_fake)

                # MPD discriminator loss
                mpd_real_out, mpd_fake_out, _, _ = mpd(wav_real_aug, wav_fake_aug)
                loss_disc_mpd, _, _ = discriminator_loss(mpd_real_out, mpd_fake_out)

                # MSD discriminator loss
                msd_real_out, msd_fake_out, _, _ = msd(wav_real_aug, wav_fake_aug)
                loss_disc_msd, _, _ = discriminator_loss(msd_real_out, msd_fake_out)

                loss_disc = loss_disc_mpd + loss_disc_msd

                optim_d.zero_grad()
                loss_disc.backward()
                optim_d.step()

                # ===== Generator step =====
                y_g_hat = generator(mel_input)
                wav_fake_g = y_g_hat[..., :min_len]

                # Compute mel of generated audio for mel loss
                with torch.no_grad():
                    wav_for_mel = wav_fake_g.clamp(-1.0, 1.0)

                mel_fake = spectrogram.waveform_to_mel(wav_for_mel)  # [B, 1, 128, T']
                mel_real_for_loss = mel_batch[..., : mel_fake.shape[-1]]
                mel_fake_for_loss = mel_fake[..., : mel_real_for_loss.shape[-1]]

                loss_mel = torch.nn.functional.l1_loss(
                    mel_fake_for_loss, mel_real_for_loss
                )

                # Run discriminators on unaugmented audio for generator
                mpd_real_out, mpd_fake_out, mpd_real_fmaps, mpd_fake_fmaps = mpd(
                    wav_real, wav_fake_g
                )
                msd_real_out, msd_fake_out, msd_real_fmaps, msd_fake_fmaps = msd(
                    wav_real, wav_fake_g
                )

                # Generator losses
                loss_gen_mpd, _ = generator_loss(mpd_fake_out)
                loss_gen_msd, _ = generator_loss(msd_fake_out)
                loss_fm_mpd = feature_loss(mpd_real_fmaps, mpd_fake_fmaps)
                loss_fm_msd = feature_loss(msd_real_fmaps, msd_fake_fmaps)

                loss_gen = (
                    loss_gen_mpd
                    + loss_gen_msd
                    + loss_fm_mpd
                    + loss_fm_msd
                    + loss_mel * 45
                )

                optim_g.zero_grad()
                loss_gen.backward()
                optim_g.step()

                # Accumulate epoch metrics
                epoch_gen_loss += loss_gen.item()
                epoch_disc_loss += loss_disc.item()
                epoch_mel_loss += loss_mel.item()
                epoch_feat_loss += (loss_fm_mpd + loss_fm_msd).item()
                num_batches += 1

                # Check cancel event after each batch
                if cancel_event is not None and cancel_event.is_set():
                    logger.info(
                        "Cancel requested at epoch %d -- saving checkpoint", epoch
                    )
                    vocoder_state = self._build_vocoder_state(
                        generator=generator,
                        config=config,
                        epoch=epoch,
                        epochs=epochs,
                        gen_loss=epoch_gen_loss / max(num_batches, 1),
                        checkpoint_data={
                            "generator_state_dict": generator.state_dict(),
                            "mpd_state_dict": mpd.state_dict(),
                            "msd_state_dict": msd.state_dict(),
                            "optim_g_state_dict": optim_g.state_dict(),
                            "optim_d_state_dict": optim_d.state_dict(),
                            "sched_g_state_dict": sched_g.state_dict(),
                            "sched_d_state_dict": sched_d.state_dict(),
                            "epoch": epoch,
                        },
                    )
                    self._save_vocoder_state(model_path, vocoder_state)
                    return vocoder_state

            # Step schedulers
            sched_g.step()
            sched_d.step()

            # Compute epoch averages
            num_batches = max(num_batches, 1)
            avg_gen = epoch_gen_loss / num_batches
            avg_disc = epoch_disc_loss / num_batches
            avg_mel = epoch_mel_loss / num_batches
            avg_feat = epoch_feat_loss / num_batches
            final_gen_loss = avg_gen
            final_disc_loss = avg_disc

            # ETA
            elapsed = time.monotonic() - train_start
            epochs_done = epoch - start_epoch + 1
            avg_epoch_time = elapsed / epochs_done
            remaining = (epochs - epoch - 1) * avg_epoch_time

            # Emit epoch metrics
            if callback is not None:
                callback(
                    VocoderEpochMetrics(
                        epoch=epoch,
                        total_epochs=epochs,
                        gen_loss=avg_gen,
                        disc_loss=avg_disc,
                        mel_loss=avg_mel,
                        feature_loss=avg_feat,
                        learning_rate=optim_g.param_groups[0]["lr"],
                        eta_seconds=remaining,
                        elapsed_seconds=elapsed,
                    )
                )

            # Preview audio
            if (
                callback is not None
                and preview_interval > 0
                and (epoch + 1) % preview_interval == 0
            ):
                self._emit_preview(
                    generator=generator,
                    mel_batch=mel_batch,
                    epoch=epoch,
                    sample_rate=config.sample_rate,
                    callback=callback,
                )

            # Check cancel event at epoch boundary
            if cancel_event is not None and cancel_event.is_set():
                logger.info(
                    "Cancel requested at epoch %d boundary -- saving checkpoint",
                    epoch,
                )
                vocoder_state = self._build_vocoder_state(
                    generator=generator,
                    config=config,
                    epoch=epoch,
                    epochs=epochs,
                    gen_loss=avg_gen,
                    checkpoint_data={
                        "generator_state_dict": generator.state_dict(),
                        "mpd_state_dict": mpd.state_dict(),
                        "msd_state_dict": msd.state_dict(),
                        "optim_g_state_dict": optim_g.state_dict(),
                        "optim_d_state_dict": optim_d.state_dict(),
                        "sched_g_state_dict": sched_g.state_dict(),
                        "sched_d_state_dict": sched_d.state_dict(),
                        "epoch": epoch,
                    },
                )
                self._save_vocoder_state(model_path, vocoder_state)
                return vocoder_state

        # --- Normal completion ---
        vocoder_state = self._build_vocoder_state(
            generator=generator,
            config=config,
            epoch=epochs - 1,
            epochs=epochs,
            gen_loss=final_gen_loss,
            checkpoint_data=None,  # No checkpoint on normal completion
        )
        self._save_vocoder_state(model_path, vocoder_state)

        if callback is not None:
            callback(
                VocoderTrainingCompleteEvent(
                    epochs_completed=epochs,
                    final_gen_loss=final_gen_loss,
                    final_disc_loss=final_disc_loss,
                    model_path=model_path,
                )
            )

        logger.info(
            "Vocoder training complete: %d epochs, gen_loss=%.4f, disc_loss=%.4f",
            epochs,
            final_gen_loss,
            final_disc_loss,
        )

        return vocoder_state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_vocoder_state(
        generator: "HiFiGANGenerator",
        config: "HiFiGANConfig",
        epoch: int,
        epochs: int,
        gen_loss: float,
        checkpoint_data: dict | None,
    ) -> dict:
        """Build the vocoder_state dict for persistence.

        Parameters
        ----------
        generator : HiFiGANGenerator
            Trained generator (state_dict extracted).
        config : HiFiGANConfig
            Model configuration.
        epoch : int
            Current/final epoch number.
        epochs : int
            Total planned epochs.
        gen_loss : float
            Final/current generator loss.
        checkpoint_data : dict | None
            Full training state for resume. ``None`` on normal completion
            (only generator weights are saved).

        Returns
        -------
        dict
            The vocoder_state dict suitable for bundling into ``.distillgan``.
        """
        from datetime import datetime, timezone

        state: dict = {
            "type": "hifigan_v2",
            "generator_state_dict": generator.state_dict(),
            "config": asdict(config),
            "training_metadata": {
                "epochs": epoch + 1,
                "total_epochs": epochs,
                "final_loss": gen_loss,
                "training_date": datetime.now(timezone.utc).isoformat(),
            },
        }

        if checkpoint_data is not None:
            state["checkpoint"] = checkpoint_data

        return state

    @staticmethod
    def _save_vocoder_state(model_path: Path, vocoder_state: dict) -> None:
        """Save vocoder_state into an existing .distillgan model file.

        Loads the model file, injects ``vocoder_state`` into the saved
        dict, and writes back.

        Parameters
        ----------
        model_path : Path
            Path to the ``.distillgan`` file.
        vocoder_state : dict
            Vocoder state dict to embed.
        """
        import torch

        model_path = Path(model_path)
        saved = torch.load(model_path, map_location="cpu", weights_only=False)
        saved["vocoder_state"] = vocoder_state
        torch.save(saved, model_path)
        logger.info("Saved vocoder state to %s", model_path)

    @staticmethod
    def _emit_preview(
        generator: "HiFiGANGenerator",
        mel_batch: "torch.Tensor",
        epoch: int,
        sample_rate: int,
        callback: Callable,
    ) -> None:
        """Generate and emit a preview audio sample.

        Parameters
        ----------
        generator : HiFiGANGenerator
            Current generator model.
        mel_batch : torch.Tensor
            A mel batch to generate preview from (uses first sample).
        epoch : int
            Current epoch number.
        sample_rate : int
            Audio sample rate.
        callback : Callable
            Event callback.
        """
        import torch

        generator.eval()
        with torch.inference_mode():
            mel_input = mel_batch[0:1].squeeze(1)  # [1, 128, T]
            preview_wav = generator(mel_input)  # [1, 1, samples]
            preview_np = (
                preview_wav.squeeze().cpu().numpy().astype(np.float32)
            )
        generator.train()

        callback(
            VocoderPreviewEvent(
                epoch=epoch,
                audio=preview_np,
                sample_rate=sample_rate,
            )
        )
