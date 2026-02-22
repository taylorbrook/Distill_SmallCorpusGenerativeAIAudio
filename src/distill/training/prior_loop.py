"""Training loop for the autoregressive prior over VQ-VAE code sequences.

Orchestrates the full prior training pipeline: loading a frozen VQ-VAE,
extracting code sequences from the training dataset, splitting into
train/val sets, training the CodePrior with cross-entropy loss, monitoring
validation perplexity, detecting memorization, and tracking the best model
state.

Key features:

- :func:`train_prior` is the top-level orchestrator (parallel to
  :func:`~distill.training.loop.train_vqvae`).
- :func:`train_prior_epoch` / :func:`validate_prior_epoch` handle single
  epochs on pre-extracted code tensors (no data loaders needed).
- :func:`check_memorization` uses relaxed adaptive thresholds per user
  decision -- only warns when memorization is very likely.
- NaN detection skips bad gradient updates (project pattern from loop.py).
- Best checkpoint tracked via ``deepcopy(state_dict)`` when validation
  perplexity improves.

Design notes:

- Lazy imports for heavy dependencies (torch) inside function bodies
  (matches :mod:`distill.training.loop` pattern).
- ``from __future__ import annotations`` for modern type syntax.
- Module-level logger.
- TYPE_CHECKING block for type hints only.
- The VQ-VAE model is only used for code extraction, NOT during prior
  training itself.  After extracting codes, prior trains purely on code
  tensors.  This ensures no gradients leak into the VQ-VAE
  (RESEARCH.md pitfall 5).
"""

from __future__ import annotations

import copy
import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import threading

    import torch

    from distill.training.metrics import MetricsCallback
    from distill.training.prior_config import PriorConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memorization Detection
# ---------------------------------------------------------------------------


def check_memorization(
    val_perplexity: float,
    codebook_size: int,
    dataset_file_count: int,
) -> tuple[bool, str]:
    """Check whether the prior is memorizing the training set.

    Uses relaxed adaptive thresholds per user decision -- only warns when
    memorization is very likely.  Small datasets naturally have low
    perplexity, so thresholds adapt to the dataset size.

    Threshold tiers:

    - ``<= 20 files``: threshold 2.0
    - ``<= 100 files``: threshold 3.0
    - ``> 100 files``: threshold 5.0

    Parameters
    ----------
    val_perplexity:
        Current validation perplexity (``exp(cross_entropy_loss)``).
    codebook_size:
        Number of entries in each codebook (for context).
    dataset_file_count:
        Number of audio files in the dataset.

    Returns
    -------
    tuple[bool, str]
        ``(is_memorizing, message)`` -- True with descriptive warning
        if memorizing, False with empty string otherwise.
    """
    # Adaptive threshold by dataset size tier
    if dataset_file_count <= 20:
        threshold = 2.0
    elif dataset_file_count <= 100:
        threshold = 3.0
    else:
        threshold = 5.0

    if val_perplexity < threshold:
        message = (
            f"Memorization likely: validation perplexity {val_perplexity:.2f} "
            f"is below threshold {threshold:.1f} for {dataset_file_count} files. "
            f"The prior may be memorizing the training set. "
            f"Consider using the best checkpoint before this point."
        )
        return True, message

    return False, ""


# ---------------------------------------------------------------------------
# Train Epoch
# ---------------------------------------------------------------------------


def train_prior_epoch(
    prior_model: "torch.nn.Module",
    train_codes: "torch.Tensor",
    optimizer: "torch.optim.Optimizer",
    device: "torch.device",
    codebook_size: int,
    batch_size: int = 32,
    gradient_clip_norm: float = 1.0,
    epoch: int = 0,
    callback: "MetricsCallback | None" = None,
) -> float:
    """Train one epoch of the prior on shuffled code sequences.

    Uses standard next-token prediction: ``input = batch[:, :-1]``,
    ``target = batch[:, 1:]``.  Shuffles data each epoch via
    ``torch.randperm``.

    Parameters
    ----------
    prior_model:
        The :class:`~distill.models.prior.CodePrior` model (must be in
        train mode).
    train_codes:
        Shape ``[N_train, flat_seq_len]`` -- flattened code sequences
        for training.
    optimizer:
        Optimiser (AdamW).
    device:
        Target device for computation.
    codebook_size:
        Number of entries in each codebook (vocabulary size).
    batch_size:
        Mini-batch size.
    gradient_clip_norm:
        Maximum gradient L2 norm for clipping.
    epoch:
        Current epoch number (for metrics emission).
    callback:
        Optional metrics event subscriber.

    Returns
    -------
    float
        Average training cross-entropy loss for the epoch.
    """
    import torch  # noqa: WPS433
    import torch.nn.functional as F  # noqa: WPS433, N812

    from distill.training.metrics import PriorStepMetrics

    prior_model.train()

    N = train_codes.size(0)
    perm = torch.randperm(N, device=train_codes.device)
    shuffled = train_codes[perm]

    total_loss_sum = 0.0
    valid_steps = 0
    total_steps = max(1, (N + batch_size - 1) // batch_size)
    lr = optimizer.param_groups[0]["lr"]

    for step_idx in range(0, N, batch_size):
        step_start = time.time()
        step = step_idx // batch_size

        batch = shuffled[step_idx : step_idx + batch_size].to(device)

        # Next-token prediction setup
        inp = batch[:, :-1]   # [B, T-1]
        target = batch[:, 1:]  # [B, T-1]

        # Forward pass
        logits = prior_model(inp)  # [B, T-1, codebook_size]

        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.reshape(-1, codebook_size),
            target.reshape(-1),
        )

        # NaN detection: skip bad gradient updates (project pattern)
        if loss.isnan():
            logger.warning(
                "NaN loss at epoch %d step %d -- skipping gradient update",
                epoch, step,
            )
            step_time = time.time() - step_start
            if callback is not None:
                callback(PriorStepMetrics(
                    epoch=epoch, step=step, total_steps=total_steps,
                    train_loss=float("nan"),
                    learning_rate=lr, step_time_s=step_time,
                ))
            continue

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                prior_model.parameters(), max_norm=gradient_clip_norm,
            )

        optimizer.step()

        # Accumulate
        total_loss_sum += loss.item()
        valid_steps += 1

        step_time = time.time() - step_start

        # Log progress every 10 steps (or every step in first epoch)
        if step % 10 == 0 or epoch == 0:
            print(
                f"[PRIOR-TRAIN] Epoch {epoch} step {step + 1}/{total_steps}  "
                f"loss={loss.item():.4f}  step_time={step_time:.2f}s",
                flush=True,
            )

        # Emit step metrics
        if callback is not None:
            callback(PriorStepMetrics(
                epoch=epoch, step=step, total_steps=total_steps,
                train_loss=loss.item(),
                learning_rate=lr, step_time_s=step_time,
            ))

    # Average over valid steps
    if valid_steps > 0:
        return total_loss_sum / valid_steps
    return float("nan")


# ---------------------------------------------------------------------------
# Validate Epoch
# ---------------------------------------------------------------------------


def validate_prior_epoch(
    prior_model: "torch.nn.Module",
    val_codes: "torch.Tensor",
    device: "torch.device",
    codebook_size: int,
    batch_size: int = 32,
) -> float:
    """Validate one epoch of the prior.

    Computes cross-entropy loss with ``reduction="sum"`` divided by total
    tokens for an accurate average across all validation sequences.

    Parameters
    ----------
    prior_model:
        The :class:`~distill.models.prior.CodePrior` model.
    val_codes:
        Shape ``[N_val, flat_seq_len]`` -- flattened code sequences
        for validation.
    device:
        Target device for computation.
    codebook_size:
        Number of entries in each codebook (vocabulary size).
    batch_size:
        Mini-batch size.

    Returns
    -------
    float
        Average validation cross-entropy loss.
    """
    import torch  # noqa: WPS433
    import torch.nn.functional as F  # noqa: WPS433, N812

    prior_model.eval()

    N = val_codes.size(0)
    total_loss_sum = 0.0
    total_tokens = 0

    with torch.no_grad():
        for step_idx in range(0, N, batch_size):
            batch = val_codes[step_idx : step_idx + batch_size].to(device)

            # Next-token prediction setup
            inp = batch[:, :-1]
            target = batch[:, 1:]

            # Forward pass
            logits = prior_model(inp)  # [B, T-1, codebook_size]

            # Cross-entropy loss with reduction="sum" for accurate average
            loss = F.cross_entropy(
                logits.reshape(-1, codebook_size),
                target.reshape(-1),
                reduction="sum",
            )

            total_loss_sum += loss.item()
            total_tokens += target.numel()

    prior_model.train()

    if total_tokens > 0:
        return total_loss_sum / total_tokens
    return float("nan")


# ---------------------------------------------------------------------------
# Main Training Orchestrator
# ---------------------------------------------------------------------------


def train_prior(
    model_path: "str | Path",
    dataset_dir: "str | Path",
    prior_config: "PriorConfig",
    callback: "MetricsCallback | None" = None,
    cancel_event: "threading.Event | None" = None,
) -> dict:
    """Full prior training orchestrator.

    Loads a frozen VQ-VAE, extracts code sequences from the dataset,
    splits into train/val, trains a :class:`~distill.models.prior.CodePrior`
    with cross-entropy loss, monitors validation perplexity, detects
    memorization, and tracks the best model state.

    Steps:

    1. Load VQ-VAE model via ``load_model_v2``
    2. Freeze VQ-VAE parameters
    3. Create data loaders from dataset directory
    4. Extract code sequences through frozen VQ-VAE
    5. Flatten codes to ``[N, flat_seq_len]``
    6. Create CodePrior model
    7. Train with cross-entropy, track best checkpoint, detect memorization
    8. Return trained prior with best weights loaded

    Parameters
    ----------
    model_path:
        Path to the ``.distill`` VQ-VAE model file.
    dataset_dir:
        Path to the dataset directory containing audio files.
    prior_config:
        :class:`~distill.training.prior_config.PriorConfig` with all
        hyperparameters.
    callback:
        Optional metrics event subscriber for UI updates.
    cancel_event:
        If set, training stops early.

    Returns
    -------
    dict
        ``{prior_model, prior_config, prior_metadata, codebook_size,
        seq_len, num_quantizers}``
    """
    import torch  # noqa: WPS433
    from dataclasses import asdict as _asdict

    from distill.models.persistence import load_model_v2
    from distill.models.prior import CodePrior, extract_code_sequences, flatten_codes
    from distill.training.config import RegularizationConfig, TrainingConfig
    from distill.training.dataset import create_data_loaders
    from distill.training.metrics import (
        PriorEpochMetrics,
        PriorTrainingCompleteEvent,
    )

    # -----------------------------------------------------------------
    # 1. Load VQ-VAE model
    # -----------------------------------------------------------------
    model_path = Path(model_path)
    dataset_dir = Path(dataset_dir)

    # Resolve device
    if prior_config.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(prior_config.device)

    print(
        f"[PRIOR-TRAIN] Loading VQ-VAE from {model_path} onto {device}",
        flush=True,
    )
    loaded = load_model_v2(model_path, device=str(device))
    vqvae_model = loaded.model
    spectrogram = loaded.spectrogram
    metadata = loaded.metadata
    vqvae_config = loaded.vqvae_config or {}

    # 2. Freeze VQ-VAE
    vqvae_model.eval()
    for param in vqvae_model.parameters():
        param.requires_grad_(False)

    # -----------------------------------------------------------------
    # 3. Create data loaders from dataset_dir
    # -----------------------------------------------------------------
    # Collect audio files
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif"}
    file_paths = sorted([
        p for p in dataset_dir.iterdir()
        if p.is_file() and p.suffix.lower() in audio_extensions
    ])
    dataset_file_count = len(file_paths)

    if dataset_file_count == 0:
        raise ValueError(f"No audio files found in {dataset_dir}")

    print(
        f"[PRIOR-TRAIN] Found {dataset_file_count} audio files in {dataset_dir}",
        flush=True,
    )

    # Build a minimal TrainingConfig for create_data_loaders compatibility
    loader_config = TrainingConfig(
        batch_size=prior_config.batch_size,
        val_fraction=prior_config.val_fraction,
        chunk_duration_s=1.0,
        num_workers=0,
        regularization=RegularizationConfig(),
    )

    train_loader, val_loader = create_data_loaders(
        file_paths, loader_config, None,
    )

    print(
        f"[PRIOR-TRAIN] {len(train_loader.dataset)} train chunks, "
        f"{len(val_loader.dataset)} val chunks",
        flush=True,
    )

    # -----------------------------------------------------------------
    # 4. Extract code sequences through frozen VQ-VAE
    # -----------------------------------------------------------------
    print("[PRIOR-TRAIN] Extracting train code sequences...", flush=True)
    train_indices = extract_code_sequences(
        vqvae_model, train_loader, spectrogram, device,
    )
    print(
        f"[PRIOR-TRAIN] Train codes shape: {train_indices.shape}",
        flush=True,
    )

    print("[PRIOR-TRAIN] Extracting val code sequences...", flush=True)
    val_indices = extract_code_sequences(
        vqvae_model, val_loader, spectrogram, device,
    )
    print(
        f"[PRIOR-TRAIN] Val codes shape: {val_indices.shape}",
        flush=True,
    )

    # -----------------------------------------------------------------
    # 5. Flatten codes
    # -----------------------------------------------------------------
    train_codes = flatten_codes(train_indices)  # [N_train, flat_seq_len]
    val_codes = flatten_codes(val_indices)      # [N_val, flat_seq_len]

    flat_seq_len = train_codes.size(1)
    num_quantizers = vqvae_config.get("num_quantizers", train_indices.size(2))
    codebook_size = vqvae_config.get("codebook_size", 256)

    print(
        f"[PRIOR-TRAIN] Flattened: {train_codes.size(0)} train, "
        f"{val_codes.size(0)} val, seq_len={flat_seq_len}, "
        f"codebook_size={codebook_size}, num_quantizers={num_quantizers}",
        flush=True,
    )

    # -----------------------------------------------------------------
    # 6. Create CodePrior model
    # -----------------------------------------------------------------
    prior_model = CodePrior(
        codebook_size=codebook_size,
        seq_len=flat_seq_len,
        num_quantizers=num_quantizers,
        hidden_size=prior_config.hidden_size,
        num_layers=prior_config.num_layers,
        num_heads=prior_config.num_heads,
        dropout=prior_config.dropout,
    )
    prior_model = prior_model.to(device)

    total_params = sum(p.numel() for p in prior_model.parameters())
    print(
        f"[PRIOR-TRAIN] CodePrior: {total_params:,} parameters "
        f"(hidden={prior_config.hidden_size}, layers={prior_config.num_layers}, "
        f"heads={prior_config.num_heads})",
        flush=True,
    )

    # -----------------------------------------------------------------
    # 7. Create optimizer and scheduler
    # -----------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        prior_model.parameters(),
        lr=prior_config.learning_rate,
        weight_decay=prior_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=prior_config.max_epochs,
    )

    # -----------------------------------------------------------------
    # 8. Training loop
    # -----------------------------------------------------------------
    best_val_perplexity = float("inf")
    best_state_dict: dict | None = None
    was_memorizing = False
    train_start_time = time.time()
    epochs_trained = 0
    train_loss = float("nan")

    for epoch in range(prior_config.max_epochs):
        epoch_start = time.time()

        # Check cancellation
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Cancel detected before epoch %d", epoch)
            break

        print(
            f"[PRIOR-TRAIN] Starting epoch {epoch + 1}/{prior_config.max_epochs}",
            flush=True,
        )

        # Train epoch
        train_loss = train_prior_epoch(
            prior_model=prior_model,
            train_codes=train_codes,
            optimizer=optimizer,
            device=device,
            codebook_size=codebook_size,
            batch_size=prior_config.batch_size,
            gradient_clip_norm=prior_config.gradient_clip_norm,
            epoch=epoch,
            callback=callback,
        )

        # Validate epoch
        val_loss = validate_prior_epoch(
            prior_model=prior_model,
            val_codes=val_codes,
            device=device,
            codebook_size=codebook_size,
            batch_size=prior_config.batch_size,
        )

        # Compute perplexity
        val_perplexity = math.exp(min(val_loss, 20.0))  # clamp to avoid overflow

        # Track best checkpoint
        if val_perplexity < best_val_perplexity:
            best_val_perplexity = val_perplexity
            best_state_dict = copy.deepcopy(prior_model.state_dict())
            print(
                f"[PRIOR-TRAIN] New best perplexity: {best_val_perplexity:.2f}",
                flush=True,
            )

        # Check memorization
        is_memorizing, memorization_message = check_memorization(
            val_perplexity, codebook_size, dataset_file_count,
        )
        if is_memorizing:
            was_memorizing = True
            logger.warning(memorization_message)
            print(
                f"[PRIOR-TRAIN] WARNING: {memorization_message}",
                flush=True,
            )

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[PRIOR-TRAIN] Epoch {epoch + 1}/{prior_config.max_epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"perplexity={val_perplexity:.2f}  best={best_val_perplexity:.2f}  "
            f"time={epoch_time:.1f}s",
            flush=True,
        )

        # Emit epoch metrics
        if callback is not None:
            callback(PriorEpochMetrics(
                epoch=epoch,
                total_epochs=prior_config.max_epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                val_perplexity=val_perplexity,
                best_perplexity=best_val_perplexity,
                is_memorizing=is_memorizing,
                memorization_message=memorization_message,
                epoch_time_s=epoch_time,
                learning_rate=current_lr,
            ))

        # Track epochs completed
        epochs_trained = epoch + 1

        # Step scheduler
        scheduler.step()

        # Check cancellation at end of epoch
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Cancel detected after epoch %d", epoch)
            break

    # -----------------------------------------------------------------
    # 9. Finalize: load best checkpoint
    # -----------------------------------------------------------------

    if best_state_dict is not None:
        prior_model.load_state_dict(best_state_dict)
        print(
            f"[PRIOR-TRAIN] Loaded best checkpoint (perplexity={best_val_perplexity:.2f})",
            flush=True,
        )

    # Compute final metrics
    final_val_loss = validate_prior_epoch(
        prior_model, val_codes, device, codebook_size, prior_config.batch_size,
    )
    final_perplexity = math.exp(min(final_val_loss, 20.0))
    final_train_loss = train_loss

    # Emit completion event
    if callback is not None:
        callback(PriorTrainingCompleteEvent(
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            final_perplexity=final_perplexity,
            best_perplexity=best_val_perplexity,
            epochs_trained=epochs_trained,
            was_memorizing=was_memorizing,
        ))

    elapsed = time.time() - train_start_time
    print(
        f"[PRIOR-TRAIN] Training complete: {epochs_trained} epochs in {elapsed:.1f}s "
        f"(best perplexity={best_val_perplexity:.2f})",
        flush=True,
    )

    from datetime import datetime, timezone

    return {
        "prior_model": prior_model,
        "prior_config": _asdict(prior_config),
        "prior_metadata": {
            "epochs_trained": epochs_trained,
            "final_perplexity": final_perplexity,
            "best_perplexity": best_val_perplexity,
            "training_date": datetime.now(timezone.utc).isoformat(),
        },
        "codebook_size": codebook_size,
        "seq_len": flat_seq_len,
        "num_quantizers": num_quantizers,
    }
