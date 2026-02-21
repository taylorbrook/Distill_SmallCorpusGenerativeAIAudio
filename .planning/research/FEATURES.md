# Feature Landscape: v1.1 Neural Vocoder Integration

**Domain:** Neural vocoder integration into existing generative audio tool
**Researched:** 2026-02-21
**Confidence:** HIGH (BigVGAN-v2 parameters verified against official config.json; HiFi-GAN architecture from original paper; codebase analysis from direct code reading)

## Context

This feature landscape covers ONLY the v1.1 milestone: replacing Griffin-Lim with neural vocoders (BigVGAN-v2 universal + optional per-model HiFi-GAN V2). All features below are scoped to vocoder integration; existing v1.0 features (VAE training, slider controls, export pipeline, model library, CLI) are already shipped and stable.

### Critical Technical Context

The existing system operates on mel spectrograms with these parameters:
- `sample_rate=48000`, `n_fft=2048`, `hop_length=512`, `n_mels=128`, `f_min=0.0`, `f_max=None`
- Normalization: `log1p(mel)` / `expm1(mel_log)` (log1p/expm1 pair)

BigVGAN-v2's closest model (`bigvgan_v2_44khz_128band_512x`) uses:
- `sampling_rate=44100`, `n_fft=2048`, `hop_size=512`, `num_mels=128`, `fmin=0`, `fmax=null`
- Normalization: `log(clamp(mel, min=1e-5))` (plain log, NOT log1p)

The n_fft, hop_size, and n_mels are identical. The sample rate (48kHz vs 44.1kHz) and mel normalization (log1p vs log) differ. This mismatch is the single most important technical constraint for this milestone. See PITFALLS.md for detailed analysis.

---

## Table Stakes

Features users expect when a vocoder upgrade ships. Missing any of these means the upgrade feels broken or incomplete.

| Feature | Why Expected | Complexity | Depends On |
|---------|--------------|------------|------------|
| **Transparent vocoder swap in generation pipeline** | Users press "Generate" and get better audio; workflow unchanged | Medium | Spectrogram mel normalization adapter, vocoder model loaded in pipeline |
| **Automatic BigVGAN-v2 download on first use** | Users should not manually download model files, clone git repos, or manage checkpoints | Medium | huggingface_hub download with progress, local cache directory |
| **Audio quality improvement over Griffin-Lim** | The entire point of the upgrade; output must be audibly, measurably better | Low (inherent to neural vocoder) | Correct mel parameter alignment between VAE and vocoder |
| **Same export pipeline works** | WAV/MP3/FLAC/OGG export, metadata, sidecar JSON, spatial audio all continue working | Low | Vocoder produces same output format (numpy float32 waveform) |
| **CLI generation works with vocoder** | `distill generate model_name` uses neural vocoder automatically | Low | Vocoder integrated at pipeline level, not UI level |
| **All hardware targets supported** | CUDA, MPS (Apple Silicon), CPU fallback all work | Medium | BigVGAN uses standard PyTorch ops; verify no CUDA-only ops in inference path |
| **Vocoder selection in UI** | Dropdown or radio to choose "BigVGAN (universal)" vs "Per-model HiFi-GAN" (if trained) | Low | UI component in Generate tab, state management |
| **Vocoder selection in CLI** | `--vocoder bigvgan` or `--vocoder hifigan` flag on generate command | Low | CLI argument, pipeline routing |
| **Download progress indication** | First-run BigVGAN download (~500MB) shows progress bar in both UI and CLI | Low | huggingface_hub has built-in progress; surface in Gradio and Rich |
| **Graceful fallback if download fails** | If BigVGAN can't be downloaded (offline, network error), fall back to Griffin-Lim with warning | Low | Try/except around download, fallback path |

### Table Stakes Dependency Chain

```
BigVGAN Download Manager
    |-- downloads bigvgan_v2_44khz_128band_512x from HuggingFace (~500MB)
    |-- caches in local data directory (alongside models)
    |-- provides status/progress callbacks
    v
Mel Normalization Adapter
    |-- converts VAE output (log1p-normalized mel) to BigVGAN input (log-normalized mel)
    |-- handles sample rate mismatch (48kHz VAE -> 44.1kHz vocoder -> resample to target)
    v
Vocoder Wrapper (abstract interface)
    |-- BigVGANVocoder (universal, downloaded)
    |-- HiFiGANVocoder (per-model, trained)
    |-- GriffinLimVocoder (fallback, existing)
    v
Generation Pipeline Integration
    |-- replaces spectrogram.mel_to_waveform() call
    |-- vocoder selection logic (auto/bigvgan/hifigan/griffinlim)
    |-- waveform output compatible with existing spatial/export pipeline
```

---

## Differentiators

Features that make this vocoder integration stand out beyond a basic swap.

| Feature | Value Proposition | Complexity | Depends On |
|---------|-------------------|------------|------------|
| **Per-model HiFi-GAN V2 training** | Fine-tune a lightweight vocoder on the user's specific audio domain for maximum fidelity; e.g., a vocoder tuned to "my field recordings" will reconstruct those textures better than a universal model | High | Training loop for HiFi-GAN V2, mel-spectrogram ground truth generation, discriminator training, checkpoint management |
| **Vocoder training integrated in Train tab** | "Train Vocoder" button alongside VAE training; same UX patterns (progress, loss chart, cancel, resume) | Medium | TrainingRunner pattern reuse, HiFi-GAN training loop, UI components |
| **Vocoder training in CLI** | `distill train-vocoder --model my_model --epochs 500` | Low | CLI command, training loop |
| **Per-model vocoder bundled in .distill file** | When user saves a model with a trained vocoder, the vocoder weights are stored in the same .distill file | Medium | Model format v2 with optional vocoder_state_dict, backward-compatible loading |
| **Automatic vocoder selection** | Pipeline auto-selects best available vocoder: per-model HiFi-GAN if trained > BigVGAN universal > Griffin-Lim fallback | Low | Vocoder registry per model, priority logic |
| **A/B comparison: vocoder vs Griffin-Lim** | Let users hear the difference side-by-side; builds confidence in the upgrade | Low | Generate same mel through both paths, existing A/B comparison UI |
| **Vocoder quality metrics in UI** | Show SNR, spectral convergence, or PESQ-like score comparing vocoder output to target | Medium | Objective quality metrics, comparison against ground truth (if available) |
| **Lazy vocoder loading** | BigVGAN model loaded only when first generation is requested, not at app startup | Low | Deferred initialization pattern, existing lazy import conventions |

### Per-Model HiFi-GAN V2 Training Workflow (User Perspective)

1. User has a trained VAE model (e.g., "Field Recordings v3")
2. In the Train tab, selects "Field Recordings v3" and clicks "Train Vocoder"
3. Training uses the SAME audio files from the original dataset
4. Pipeline generates mel spectrograms using the VAE's spectrogram config
5. HiFi-GAN V2 learns to reconstruct waveforms from THOSE specific mels
6. Training shows loss curve, previews vocoder reconstruction quality
7. On completion, vocoder is bundled into the .distill model file
8. Future generations with that model automatically use the per-model vocoder

### Per-Model HiFi-GAN V2 Training Workflow (Technical)

```
User's Audio Files (WAV, 48kHz)
    |
    v
AudioSpectrogram.waveform_to_mel() --> ground truth mel spectrograms
    |
    v
HiFi-GAN V2 Generator: mel --> reconstructed waveform
    |
    v
Multi-Period Discriminator + Multi-Scale Discriminator
    |-- Adversarial loss (generator vs discriminator)
    |-- Mel spectrogram reconstruction loss
    |-- Feature matching loss
    v
Trained HiFi-GAN V2 weights (~4MB for V2)
    |
    v
Bundled into .distill file alongside VAE weights
```

**Training time estimate (consumer hardware):**
- HiFi-GAN V2 has only 0.92M parameters (vs V1's 13.92M)
- Fine-tuning from universal pretrained weights: ~30-60 min on consumer GPU
- Training from scratch: ~2-4 hours on consumer GPU
- V2 chosen specifically because it balances quality (4.23 MOS) with small size and fast training

---

## Anti-Features

Features to explicitly NOT build for v1.1.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **48kHz BigVGAN training from scratch** | Requires 8x A100 GPUs, weeks of training, massive diverse audio dataset; completely infeasible for a product update | Use the 44.1kHz pretrained model and resample output to 48kHz; quality difference between native 48kHz and resampled 44.1kHz is inaudible for generated textures |
| **Custom BigVGAN fine-tuning per model** | BigVGAN has 122M parameters; fine-tuning is impractical on consumer hardware (OOM on 8GB VRAM GPUs) | Use BigVGAN as-is for universal quality; offer lightweight HiFi-GAN V2 (0.92M params) for per-model fine-tuning |
| **Real-time vocoder streaming** | BigVGAN is not designed for real-time; project is explicitly non-real-time; adds complexity for no value | Keep non-real-time generation; BigVGAN inference is fast enough (167x real-time on V100, much faster on modern GPUs) |
| **Vocos as primary vocoder** | The 48kHz Vocos model is an unofficial alpha from an unaffiliated author; not production ready | BigVGAN-v2 is NVIDIA-backed, well-tested, with official HuggingFace integration; revisit Vocos if an official 48kHz model ships |
| **MusicHiFi stereo vocoder** | Research paper from 2024; no official pretrained models available; cascade of 3 GANs is complex | Handle stereo via existing spatial processing pipeline (mid-side widening, binaural HRTF) applied AFTER mono vocoder output |
| **Multiple BigVGAN model variants** | Offering 22kHz/24kHz/44kHz variants adds UI complexity and user confusion for marginal benefit | Ship only `bigvgan_v2_44khz_128band_512x` (highest available quality); resample to target SR |
| **Griffin-Lim removal in v1.1** | Removing the fallback path before the neural vocoder is battle-tested risks breaking generation for users who can't download BigVGAN | Keep Griffin-Lim as fallback; mark as deprecated; remove in v1.2 after vocoder stability is confirmed |
| **CUDA kernel optimization for BigVGAN** | The fused CUDA kernel requires CUDA 12.1+ and nvcc/ninja compilation; fails on MPS/CPU; adds build complexity | Use standard PyTorch inference (still very fast); add CUDA kernel as optional optimization later |

---

## Feature Dependencies

### Vocoder Integration Dependency Graph

```
BigVGAN Download Manager
    |
    +-- Mel Normalization Adapter (converts log1p -> log)
    |       |
    |       +-- VocoderWrapper (abstract: BigVGAN / HiFi-GAN / Griffin-Lim)
    |               |
    |               +-- GenerationPipeline.generate() integration
    |                       |
    |                       +-- UI: Vocoder dropdown in Generate tab
    |                       |
    |                       +-- CLI: --vocoder flag on generate command
    |
    +-- HiFi-GAN V2 Training Loop
            |
            +-- HiFi-GAN V2 Training CLI command
            |
            +-- HiFi-GAN V2 Training UI (Train tab)
            |
            +-- .distill Model Format v2 (vocoder_state_dict)
                    |
                    +-- Model Library: vocoder status indicator
                    |
                    +-- Automatic vocoder selection logic
```

### Critical Path (minimum viable vocoder)

```
1. Mel Normalization Adapter
2. BigVGAN Download Manager
3. VocoderWrapper abstraction
4. GenerationPipeline integration
5. UI dropdown + CLI flag
```

This delivers the core value: "press Generate, get better audio." Per-model HiFi-GAN training is a follow-on within the same milestone.

### Secondary Path (per-model training)

```
6. HiFi-GAN V2 training loop (generator + discriminators)
7. Training CLI command
8. Training UI integration (reuse TrainingRunner pattern)
9. .distill format v2 (bundle vocoder weights)
10. Auto-selection logic (per-model > BigVGAN > Griffin-Lim)
```

---

## Detailed Feature Specifications

### 1. BigVGAN-v2 Download and Cache Management

**User story:** On first generation attempt, the app downloads BigVGAN-v2 (~500MB) with a progress bar. On subsequent runs, it loads from cache instantly.

**Implementation:**
- Use `huggingface_hub.hf_hub_download()` for `nvidia/bigvgan_v2_44khz_128band_512x`
- Cache in `{data_dir}/vocoders/bigvgan_v2_44khz_128band_512x/`
- Download only `bigvgan_generator.pt` and `config.json` (skip discriminator/optimizer)
- Show progress via `huggingface_hub`'s built-in tqdm (Gradio wraps this; CLI uses Rich progress)
- First load also calls `model.remove_weight_norm()` and caches the processed state

**Files affected:** New `distill/vocoder/download.py`

**Complexity:** Medium (HuggingFace API is straightforward; error handling for offline/partial downloads needs care)

### 2. Mel Normalization Adapter

**User story:** Invisible to user. VAE produces log1p-normalized mels at 48kHz; BigVGAN expects log-normalized mels at 44.1kHz.

**Implementation:**
```python
def adapt_mel_for_bigvgan(mel_log1p: Tensor, source_sr: int = 48000, target_sr: int = 44100) -> Tensor:
    """Convert VAE mel output to BigVGAN mel input.

    1. Undo log1p: mel_linear = expm1(mel_log1p)
    2. Apply BigVGAN's log: mel_log = log(clamp(mel_linear, min=1e-5))
    3. (Optional) Resample time axis if source_sr != target_sr
    """
```

The frequency axis (128 mels, 0 to Nyquist) maps differently between 48kHz (0-24kHz) and 44.1kHz (0-22.05kHz). However, BigVGAN was trained as a UNIVERSAL vocoder on diverse audio. The mel filterbank mismatch is minor (top 2kHz of frequency range) and BigVGAN's robustness should handle it. The normalization conversion is the critical part.

**Alternative approach:** Resample audio to 44.1kHz before mel computation, use BigVGAN's own mel function, then resample output to 48kHz. This is "safer" but slower and requires changing the generation flow.

**Recommended approach:** Direct mel normalization conversion (log1p -> log) without resampling, generating at 44.1kHz and resampling vocoder output to 48kHz. This is simpler and leverages BigVGAN's universality. The existing pipeline already has a resampler cache for sample rate conversion.

**Files affected:** New `distill/vocoder/adapter.py`

**Complexity:** Medium (math is simple; validating audio quality requires listening tests)

### 3. Vocoder Wrapper Abstraction

**User story:** Pipeline code calls `vocoder.mel_to_waveform(mel)` regardless of which vocoder is active.

**Implementation:**
```python
class BaseVocoder(Protocol):
    def mel_to_waveform(self, mel: Tensor) -> Tensor: ...
    def to(self, device: torch.device) -> "BaseVocoder": ...

class BigVGANVocoder(BaseVocoder):
    """Universal vocoder using BigVGAN-v2."""

class HiFiGANVocoder(BaseVocoder):
    """Per-model vocoder using HiFi-GAN V2."""

class GriffinLimVocoder(BaseVocoder):
    """Legacy vocoder wrapping existing AudioSpectrogram.mel_to_waveform()."""
```

**Files affected:** New `distill/vocoder/__init__.py`, `distill/vocoder/bigvgan.py`, `distill/vocoder/hifigan.py`, `distill/vocoder/griffinlim.py`

**Complexity:** Low (wrapper pattern, existing mel_to_waveform interface)

### 4. Generation Pipeline Integration

**User story:** `GenerationPipeline.generate()` uses selected vocoder instead of `spectrogram.mel_to_waveform()`.

**Current code path (line 339 in generation.py):**
```python
wav = spectrogram.mel_to_waveform(combined_mel)
```

**New code path:**
```python
wav = self.vocoder.mel_to_waveform(combined_mel)
```

The vocoder is set on the pipeline via constructor or setter. The rest of the pipeline (spatial processing, normalization, resampling, quality metrics, export) is unchanged.

**Files affected:** `distill/inference/generation.py`, `distill/inference/chunking.py`

**Complexity:** Low (single method call replacement; wrapper handles mel format conversion internally)

### 5. UI: Vocoder Selection

**User story:** Generate tab shows a dropdown: "Auto (best available)" / "BigVGAN (universal)" / "Per-model HiFi-GAN" / "Griffin-Lim (legacy)". "Auto" is default and picks the best available for the loaded model.

**Implementation:**
- Add `gr.Dropdown` to Generate tab, below the existing output mode selector
- "Auto" logic: if loaded model has trained HiFi-GAN weights -> use HiFi-GAN; else -> use BigVGAN; if BigVGAN not downloaded -> use Griffin-Lim
- Dropdown updates when model is loaded (shows "Per-model HiFi-GAN" only if model has vocoder weights)
- First selection of BigVGAN triggers download if not cached

**Files affected:** `distill/ui/tabs/generate_tab.py`, `distill/ui/state.py`

**Complexity:** Low (follows existing dropdown patterns in the Generate tab)

### 6. CLI: Vocoder Selection

**User story:** `distill generate my_model --vocoder bigvgan` or `distill generate my_model --vocoder auto`

**Implementation:**
- Add `--vocoder` option to generate command: `auto` (default), `bigvgan`, `hifigan`, `griffinlim`
- `auto` follows same priority as UI
- `bigvgan` triggers download if not cached, shows Rich progress bar
- `hifigan` fails with clear error if model has no trained vocoder

**Files affected:** `distill/cli/generate.py`

**Complexity:** Low (single Typer option, pipeline routing)

### 7. Per-Model HiFi-GAN V2 Training

**User story:** User clicks "Train Vocoder" for a model. Training uses the same audio files. After training (~30-60 min on consumer GPU), the vocoder is bundled with the model and automatically used for generation.

**Implementation:**
- HiFi-GAN V2 architecture (0.92M params): generator with multi-receptive field fusion
- Multi-Period Discriminator (MPD) + Multi-Scale Discriminator (MSD)
- Training loop: adversarial loss + mel reconstruction loss + feature matching loss
- Input: ground-truth audio from user's dataset
- Target: reconstruct waveform from mel spectrogram (computed via the model's SpectrogramConfig)
- Fine-tune from BigVGAN or train from scratch (BigVGAN fine-tune preferred for faster convergence)
- Save vocoder weights separately first, then bundle into .distill on completion

**Files affected:** New `distill/vocoder/training.py`, `distill/vocoder/discriminators.py`, `distill/vocoder/hifigan_model.py`

**Complexity:** High (full GAN training loop with multiple discriminators and losses)

### 8. .distill Model Format v2

**User story:** When a user saves a model with a trained vocoder, loading that model on another machine brings the vocoder along.

**Implementation:**
- Increment `SAVED_MODEL_VERSION` to 2
- Add optional `vocoder_state_dict` and `vocoder_config` to saved dict
- Backward compatible: v1 files load without vocoder (BigVGAN used as fallback)
- Forward compatible: v2 files fail gracefully on older software (version check already exists)

**Format change:**
```python
saved = {
    "format": "distill_model",
    "version": 2,  # was 1
    "model_state_dict": ...,
    "latent_dim": ...,
    "spectrogram_config": ...,
    "latent_analysis": ...,
    "training_config": ...,
    "metadata": ...,
    # NEW in v2:
    "vocoder_type": "hifigan_v2",  # or None
    "vocoder_state_dict": ...,     # or None
    "vocoder_config": ...,         # or None
}
```

**Size impact:** HiFi-GAN V2 adds ~4MB to .distill files (0.92M params * 4 bytes). Current .distill files are ~2-5MB. Total ~6-9MB. Acceptable.

**Files affected:** `distill/models/persistence.py`, `distill/library/catalog.py` (add `has_vocoder` field to ModelEntry)

**Complexity:** Medium (format migration, backward compatibility testing)

### 9. Vocoder Training UI Integration

**User story:** Train tab shows "Train Vocoder" button when a saved model exists. Uses same UX patterns as VAE training: progress bar, loss chart, cancel, preview audio.

**Implementation:**
- Add "Vocoder Training" section to Train tab (or new sub-tab)
- Model selector dropdown (from library)
- "Train Vocoder" button, reuses TrainingRunner threading pattern
- Loss chart shows generator loss, discriminator loss, mel reconstruction loss
- Audio preview: original vs reconstructed comparison
- On completion: auto-saves vocoder to model, refreshes library

**Files affected:** `distill/ui/tabs/train_tab.py` (or new `vocoder_train_tab.py`)

**Complexity:** Medium (follows existing patterns; new training loop integration)

### 10. Vocoder Training CLI

**User story:** `distill train-vocoder --model "Field Recordings v3" --epochs 500`

**Implementation:**
- New Typer sub-command `train-vocoder`
- Options: `--model` (required), `--epochs` (default 500), `--batch-size` (default 4), `--learning-rate` (default 2e-4)
- Shows Rich progress bar and periodic loss updates
- On completion: saves vocoder weights to model's .distill file

**Files affected:** New `distill/cli/vocoder.py`, register in `distill/cli/__init__.py`

**Complexity:** Low (follows existing CLI patterns)

---

## MVP Recommendation for v1.1

### Phase A: Core Vocoder Swap (generates better audio)

Ship these first. They deliver the primary value with minimum risk.

1. **Mel Normalization Adapter** - Enables BigVGAN to consume VAE output
2. **BigVGAN Download Manager** - Gets the model onto user's machine
3. **VocoderWrapper abstraction** - Clean architecture for multiple vocoders
4. **GenerationPipeline integration** - The actual swap
5. **UI vocoder dropdown** - User control
6. **CLI --vocoder flag** - Script/batch control
7. **Graceful Griffin-Lim fallback** - Safety net

**Estimated effort:** 3-5 days for a developer familiar with the codebase.

### Phase B: Per-Model Training (maximum fidelity)

Ship after Phase A is validated by listening tests.

8. **HiFi-GAN V2 model architecture** - Generator + discriminators
9. **HiFi-GAN V2 training loop** - Full GAN training
10. **.distill format v2** - Bundle vocoder with model
11. **Training UI** - Train tab integration
12. **Training CLI** - `distill train-vocoder` command
13. **Auto-selection logic** - Per-model > BigVGAN > Griffin-Lim

**Estimated effort:** 5-8 days. GAN training is the most complex part.

### Defer to v1.2

- Full Griffin-Lim removal (after vocoder stability confirmed)
- CUDA kernel optimization for BigVGAN
- Vocoder quality comparison metrics
- Vocos investigation (if official 48kHz model ships)

---

## User Workflow Descriptions

### Workflow 1: New User, First Generation (BigVGAN)

1. User has already trained a VAE model in v1.0
2. Updates to v1.1
3. Goes to Generate tab, loads their model
4. Presses "Generate"
5. **NEW:** App detects BigVGAN not downloaded. Shows: "Downloading BigVGAN-v2 universal vocoder (489 MB)... This is a one-time download."
6. Progress bar fills over 30-120 seconds (depending on connection)
7. Generation proceeds with BigVGAN. Audio output is noticeably cleaner.
8. Subsequent generations are instant (model cached)

### Workflow 2: Power User Trains Per-Model Vocoder

1. User has a model trained on "Modular Synth Patches" (200 files)
2. Goes to Train tab, sees "Train Vocoder" section
3. Selects "Modular Synth Patches" model from dropdown
4. Clicks "Train Vocoder" (uses same audio files from original dataset)
5. Training runs for ~45 minutes on their RTX 3070
6. Loss chart shows generator and discriminator losses converging
7. Periodic previews: "original audio" vs "vocoder reconstruction"
8. Training completes. Vocoder is auto-saved to the model file.
9. Goes to Generate tab. Vocoder dropdown now shows "Per-model HiFi-GAN" option
10. "Auto" mode automatically selects per-model HiFi-GAN
11. Generated audio has even better fidelity for modular synth textures

### Workflow 3: CLI Batch Generation

```bash
# Auto-selects best vocoder (downloads BigVGAN on first run)
distill generate "Modular Synth" -n 10 -d 5.0 --vocoder auto

# Force BigVGAN for universal quality
distill generate "Modular Synth" -n 10 --vocoder bigvgan

# Use per-model vocoder (fails if not trained)
distill generate "Modular Synth" -n 10 --vocoder hifigan

# Train a per-model vocoder from CLI
distill train-vocoder --model "Modular Synth" --epochs 500

# Fall back to Griffin-Lim (legacy, for comparison)
distill generate "Modular Synth" -n 10 --vocoder griffinlim
```

### Workflow 4: Offline/No-Download Scenario

1. User is on an air-gapped machine or behind a firewall
2. Attempts generation. BigVGAN download fails.
3. App shows warning: "Could not download BigVGAN-v2. Using Griffin-Lim (legacy quality). To use neural vocoder, connect to internet or manually place model files in {path}."
4. Generation proceeds with Griffin-Lim (same quality as v1.0)
5. Manual download instructions in docs for offline deployment

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority | Phase |
|---------|------------|---------------------|----------|-------|
| BigVGAN generation (transparent swap) | CRITICAL | MEDIUM | P0 | A |
| Mel normalization adapter | CRITICAL | MEDIUM | P0 | A |
| BigVGAN auto-download with progress | HIGH | MEDIUM | P0 | A |
| Vocoder wrapper abstraction | HIGH | LOW | P0 | A |
| UI vocoder dropdown | HIGH | LOW | P0 | A |
| CLI --vocoder flag | HIGH | LOW | P0 | A |
| Griffin-Lim fallback (graceful) | HIGH | LOW | P0 | A |
| HiFi-GAN V2 training loop | HIGH | HIGH | P1 | B |
| .distill format v2 (vocoder bundling) | HIGH | MEDIUM | P1 | B |
| Vocoder training UI | MEDIUM | MEDIUM | P1 | B |
| Vocoder training CLI | MEDIUM | LOW | P1 | B |
| Auto vocoder selection | MEDIUM | LOW | P1 | B |
| Lazy vocoder loading | MEDIUM | LOW | P1 | A |
| A/B vocoder comparison | LOW | LOW | P2 | B |
| Vocoder quality metrics | LOW | MEDIUM | P2 | Defer |
| CUDA kernel optimization | LOW | MEDIUM | P3 | Defer |
| Griffin-Lim full removal | LOW | LOW | P3 | v1.2 |

**Priority key:**
- P0: Must ship for vocoder to work at all
- P1: Must ship for milestone to be complete
- P2: Nice to have, adds polish
- P3: Defer to future milestone

---

## Sources

**BigVGAN-v2:**
- [NVIDIA BigVGAN GitHub](https://github.com/NVIDIA/BigVGAN) - Official implementation, training configs
- [BigVGAN-v2 44kHz 128band 512x on HuggingFace](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x) - Model weights, config.json (verified: n_fft=2048, hop_size=512, num_mels=128, sampling_rate=44100)
- [BigVGAN-v2 meldataset.py](https://github.com/NVIDIA/BigVGAN/blob/main/meldataset.py) - Mel spectrogram computation (verified: log normalization, NOT log1p)
- [NVIDIA BigVGAN-v2 Blog Post](https://developer.nvidia.com/blog/achieving-state-of-the-art-zero-shot-waveform-audio-generation-across-audio-types/) - v2 improvements, CUDA kernel

**HiFi-GAN:**
- [HiFi-GAN GitHub](https://github.com/jik876/hifi-gan) - Original implementation (V1/V2/V3 configs)
- [HiFi-GAN Paper (NeurIPS 2020)](https://arxiv.org/abs/2010.05646) - Architecture details (V2: 0.92M params, 4.23 MOS, 764.8x real-time)
- [NVIDIA HiFi-GAN on HuggingFace](https://huggingface.co/nvidia/tts_hifigan) - Pretrained universal model

**Vocos (considered, not recommended):**
- [Vocos Paper (ICLR 2024)](https://arxiv.org/html/2306.00814v3) - 13x faster than HiFi-GAN, 70x faster than BigVGAN
- [kittn/vocos-mel-48khz-alpha1](https://huggingface.co/kittn/vocos-mel-48khz-alpha1) - Unofficial 48kHz alpha (NOT production ready)

**MusicHiFi (considered, not recommended):**
- [MusicHiFi Paper](https://arxiv.org/abs/2403.10493) - Stereo vocoder, cascade of 3 GANs, no pretrained models available

**Model Management Patterns:**
- [Coqui TTS Model Management](https://deepwiki.com/coqui-ai/TTS/3.2-model-management) - Download, cache, auto-load patterns
- [HuggingFace Hub File Download](https://huggingface.co/docs/huggingface_hub/package_reference/file_download) - Progress bars, caching, local_files_only

---
*Feature research for: v1.1 Neural Vocoder Integration*
*Researched: 2026-02-21*
*Confidence: HIGH (mel parameters verified from official BigVGAN config.json; HiFi-GAN architecture from original NeurIPS paper; codebase integration points identified from direct code reading)*
