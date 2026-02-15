"""Phase 5 Human Verification Test Script.

Runs the full pipeline: train → analyze → generate with slider controls.
Produces WAV files for listening comparison.
"""

import logging
import sys
import time
from pathlib import Path

# Force unbuffered output
logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s", stream=sys.stdout, force=True)

def log(msg):
    print(msg, flush=True)

# ── Step 1: Discover dataset ──────────────────────────────────────────

log("\n" + "=" * 60)
log(" STEP 1: Loading dataset")
log("=" * 60)

dataset_dir = Path("data/datasets/test")
file_paths = sorted(dataset_dir.glob("*.wav"))
log(f"Found {len(file_paths)} files")
for f in file_paths:
    log(f"  {f.name}")

# ── Step 2: Configure & train ─────────────────────────────────────────

log("\n" + "=" * 60)
log(" STEP 2: Training model (~50 epochs)")
log("=" * 60)

import torch
from distill.training import (
    TrainingConfig,
    RegularizationConfig,
    OverfittingPreset,
    create_data_loaders,
)
from distill.audio import AudioSpectrogram, SpectrogramConfig

# Pick device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
log(f"Device: {device}")

# Short training config for testing
config = TrainingConfig(
    latent_dim=64,
    batch_size=8,
    max_epochs=50,
    learning_rate=1e-3,
    kl_warmup_fraction=0.3,
    free_bits=0.5,
    val_fraction=0.25,
    chunk_duration_s=1.0,
    checkpoint_interval=25,
    preview_interval=25,
    preset=OverfittingPreset.CONSERVATIVE,
    regularization=RegularizationConfig(
        dropout=0.3,
        weight_decay=0.01,
        augmentation_expansion=5,
        gradient_clip_norm=1.0,
    ),
)

output_dir = Path("data/test_phase5_output")
output_dir.mkdir(parents=True, exist_ok=True)

epoch_count = 0
def on_event(event):
    global epoch_count
    from distill.training import EpochMetrics
    if isinstance(event, EpochMetrics):
        epoch_count += 1
        if epoch_count % 10 == 0 or epoch_count == 1:
            log(f"  Epoch {event.epoch}: loss={event.train_loss:.4f} val_loss={event.val_loss:.4f}")

t0 = time.time()

# Run training synchronously for clearer error reporting
from distill.training.loop import train as train_fn
result = train_fn(
    config=config,
    file_paths=file_paths,
    output_dir=output_dir,
    device=device,
    callback=on_event,
)
elapsed = time.time() - t0

model = result["model"]
log(f"\nTraining complete in {elapsed:.0f}s")

# ── Step 3: Analyze latent space ──────────────────────────────────────

log("\n" + "=" * 60)
log(" STEP 3: Analyzing latent space (PCA + feature correlation)")
log("=" * 60)

from distill.controls import (
    LatentSpaceAnalyzer,
    sliders_to_latent,
    randomize_sliders,
    center_sliders,
    get_slider_info,
)

spectrogram = AudioSpectrogram(SpectrogramConfig())
spectrogram.to(device)

# Create dataloader for analysis (training files, no augmentation)
train_loader, _ = create_data_loaders(file_paths, config)

analyzer = LatentSpaceAnalyzer(variance_threshold=0.02, n_steps=21)

t0 = time.time()
analysis = analyzer.analyze(
    model=model,
    dataloader=train_loader,
    spectrogram=spectrogram,
    device=device,
    n_random_samples=200,
)
elapsed = time.time() - t0

log(f"\nAnalysis complete in {elapsed:.0f}s")
log(f"Active PCA components: {analysis.n_active_components}")
log(f"Variance explained: {sum(analysis.explained_variance_ratio) * 100:.1f}%")
log("")

slider_info = get_slider_info(analysis)
log("Slider mapping:")
for info in slider_info:
    log(f"  Axis {info['index']+1}: {info['suggested_label']:25s} "
        f"(explains {info['variance_explained_pct']:.1f}%,  "
        f"safe range: [{info['safe_min_step']}, {info['safe_max_step']}])")

# ── Step 4: Generate audio at different slider positions ──────────────

log("\n" + "=" * 60)
log(" STEP 4: Generating audio with different slider settings")
log("=" * 60)

from distill.inference import GenerationPipeline, GenerationConfig, export_wav
import numpy as np

pipeline = GenerationPipeline(model=model, spectrogram=spectrogram, device=device)

gen_dir = Path("data/test_phase5_output/generated")
gen_dir.mkdir(parents=True, exist_ok=True)

SEED = 42
DURATION = 3.0  # 3 seconds per clip

def generate_and_export(name, slider_state, seed=SEED):
    """Generate audio from slider state and export."""
    latent = sliders_to_latent(slider_state, analysis)
    gen_config = GenerationConfig(
        duration_s=DURATION,
        seed=seed,
        latent_vector=latent,
        sample_rate=48_000,
        bit_depth="24-bit",
    )
    gen_result = pipeline.generate(gen_config)
    out_path = gen_dir / f"{name}.wav"
    export_wav(gen_result.audio, out_path, gen_result.sample_rate, "24-bit")
    q = gen_result.quality
    log(f"  {name:30s} -> SNR: {q.get('snr_db', 0):.1f}dB  "
        f"Clipping: {q.get('clipping_percent', 0):.2f}%  "
        f"Rating: {q.get('rating', '?')}")
    return gen_result

# Test 1: Center (all sliders at 0 = latent mean)
log("\n1. Center position (latent mean):")
center = center_sliders(analysis)
generate_and_export("01_center", center)

# Test 2: Random positions (within safe bounds)
log("\n2. Random positions (safe bounds, seed=42):")
rand1 = randomize_sliders(analysis, seed=42)
log(f"   Positions: {rand1.positions}")
generate_and_export("02_random_seed42", rand1)

# Test 3: Different random seed -> different positions
log("\n3. Random positions (safe bounds, seed=99):")
rand2 = randomize_sliders(analysis, seed=99)
log(f"   Positions: {rand2.positions}")
generate_and_export("03_random_seed99", rand2)

# Test 4: Reproducibility check -- same sliders + same seed = same audio?
log("\n4. Reproducibility test (same as #2, should be identical):")
result_repro = generate_and_export("04_repro_check", rand1, seed=SEED)

# Verify bit-identical
result_orig = pipeline.generate(GenerationConfig(
    duration_s=DURATION, seed=SEED,
    latent_vector=sliders_to_latent(rand1, analysis),
    sample_rate=48_000, bit_depth="24-bit",
))
if np.array_equal(result_repro.audio, result_orig.audio):
    log("   IDENTICAL -- reproducible generation confirmed")
else:
    diff = np.max(np.abs(result_repro.audio - result_orig.audio))
    log(f"   NOT identical -- max diff: {diff:.6f}")

# Test 5-6: Sweep first axis from min to max
if analysis.n_active_components >= 1:
    n_half = analysis.n_steps // 2  # 10
    label = analysis.suggested_labels[0]

    log(f"\n5. First axis ({label}) at MINIMUM safe position:")
    state_min = center_sliders(analysis)
    state_min.positions[0] = slider_info[0]['safe_min_step']
    log(f"   Positions: {state_min.positions}")
    generate_and_export("05_axis1_min", state_min)

    log(f"\n6. First axis ({label}) at MAXIMUM safe position:")
    state_max = center_sliders(analysis)
    state_max.positions[0] = slider_info[0]['safe_max_step']
    log(f"   Positions: {state_max.positions}")
    generate_and_export("06_axis1_max", state_max)

# Test 7-8: Sweep second axis if available
if analysis.n_active_components >= 2:
    label2 = analysis.suggested_labels[1]

    log(f"\n7. Second axis ({label2}) at MINIMUM safe position:")
    state_min2 = center_sliders(analysis)
    state_min2.positions[1] = slider_info[1]['safe_min_step']
    log(f"   Positions: {state_min2.positions}")
    generate_and_export("07_axis2_min", state_min2)

    log(f"\n8. Second axis ({label2}) at MAXIMUM safe position:")
    state_max2 = center_sliders(analysis)
    state_max2.positions[1] = slider_info[1]['safe_max_step']
    log(f"   Positions: {state_max2.positions}")
    generate_and_export("08_axis2_max", state_max2)

# Test 9: Warning zone (beyond safe, within warning)
log("\n9. Warning zone test (axis 1 pushed to warning boundary):")
state_warn = center_sliders(analysis)
warn_step = slider_info[0].get('warning_max_step', slider_info[0]['safe_max_step'])
state_warn.positions[0] = warn_step
log(f"   Position: {warn_step} (warning_max_step)")
generate_and_export("09_warning_zone", state_warn)

# ── Summary ───────────────────────────────────────────────────────────

log("\n" + "=" * 60)
log(" RESULTS")
log("=" * 60)
log(f"\nGenerated {len(list(gen_dir.glob('*.wav')))} WAV files in:")
log(f"  {gen_dir.resolve()}")
log("")
log("Listen and compare:")
log("  01_center        -- Baseline (latent mean)")
log("  02 vs 03         -- Different random positions -> different sounds?")
log("  02 vs 04         -- Same position + seed -> identical? (reproducibility)")
log("  05 vs 06         -- Axis 1 min vs max -> audible difference?")
if analysis.n_active_components >= 2:
    log("  07 vs 08         -- Axis 2 min vs max -> audible difference?")
log("  06 vs 09         -- Safe max vs warning zone -> degradation?")
log("")
log(f"Open folder:  open {gen_dir.resolve()}")
