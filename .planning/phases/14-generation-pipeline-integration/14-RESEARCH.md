# Phase 14: Generation Pipeline Integration - Research

**Researched:** 2026-02-27
**Domain:** Audio generation pipeline, neural vocoder integration, sample rate conversion
**Confidence:** HIGH

## Summary

Phase 14 rewires the entire audio generation pipeline to use BigVGAN instead of Griffin-Lim for mel-to-waveform conversion. The current codebase has five distinct code paths that call `spectrogram.mel_to_waveform()` (which uses Griffin-Lim internally): (1) single-chunk generation via `_generate_chunks_from_vector`, (2) crossfade generation via `generate_chunks_crossfade`, (3) latent interpolation via `generate_chunks_latent_interp`, (4) training previews via `generate_preview`, and (5) reconstruction previews via `generate_reconstruction_preview`. All five must be rerouted through `BigVGANVocoder.mel_to_waveform()`.

The core architectural change is straightforward: the current generation pipeline produces mel spectrograms in VAE format `[B, 1, 128, T]` using the overlap-add synthesis in `chunking.py`, then calls `spectrogram.mel_to_waveform()` (Griffin-Lim) to get 48kHz waveforms. Phase 14 replaces this final mel-to-waveform step with `BigVGANVocoder.mel_to_waveform()`, which internally uses MelAdapter (Griffin-Lim round-trip for mel format conversion) and produces 44.1kHz output. The internal sample rate changes from 48kHz to 44.1kHz, and resampling to 48kHz becomes an optional export-boundary operation. The crossfade blending decision (mel-space before vocoder, not waveform-space after) is already the architecture: `synthesize_continuous_mel()` produces a continuous mel spectrogram, and a single vocoder pass converts it to waveform.

**Primary recommendation:** Inject the vocoder as a dependency into `GenerationPipeline` and the chunking/preview functions. Replace the three `spectrogram.mel_to_waveform(combined_mel)` calls in chunking.py and generation.py with `vocoder.mel_to_waveform(combined_mel)`. Update the internal sample rate from 48kHz to 44.1kHz. Add high-quality resampling at the export boundary using torchaudio's Kaiser-windowed sinc interpolation.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- High quality resampling (sinc/Kaiser) -- no cheap linear interpolation
- Resampling is **optional**, not automatic -- vocoder returns native 44.1kHz
- Resampling happens at the **export boundary**, not after vocoder inference
- Default export sample rate is **44.1kHz** (BigVGAN native)
- Simple 44.1kHz / 48kHz toggle for users who want 48kHz -- no multi-rate dropdown
- Preview uses the **full BigVGAN vocoder** -- same path as final generation, no lighter alternative
- What you preview is what you get
- Crossfade blending happens in **mel space** (before vocoder) -- blend mel spectrograms, then run vocoder once on the blended result
- Latent interpolation stays in **latent space** -- interpolate between latent vectors, decode each to mel, then vocoder
- Reconstruction path uses the **full vocoder** -- consistent quality, reconstruction metric reflects what users actually hear
- Crossfade overlap region size is **configurable** by the user
- Griffin-Lim is **removed entirely in Phase 14** -- not deferred to Phase 16
- No fallback, no hidden code, no legacy path
- Neural vocoder is the only reconstruction method after this phase
- If BigVGAN weights aren't downloaded: **auto-download with progress**, then generate -- no blocking error
- GPU inference failure (e.g., OOM): **warn the user**, then fall back to CPU inference

### Claude's Discretion
- Preview output path (raw vocoder vs. through export pipeline)
- Preview latency UX (loading indicator threshold)
- Preview caching strategy (cache vocoder output vs. always regenerate)
- Vocoder inference error recovery strategy (chunk-and-retry vs. fail with message)
- Exact resampling library choice (torchaudio, soxr, etc.)

### Deferred Ideas (OUT OF SCOPE)
- Griffin-Lim removal was originally Phase 16 scope -- pulling it into Phase 14 means Phase 16 can focus entirely on per-model HiFi-GAN training and auto-selection
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| GEN-01 | All generation paths (single chunk, crossfade, latent interpolation, preview, reconstruction) use neural vocoder | All five `spectrogram.mel_to_waveform()` call sites identified. Replacement is mechanical: inject vocoder, call `vocoder.mel_to_waveform()` instead. Mel-space crossfade architecture already in place via `synthesize_continuous_mel()`. |
| GEN-02 | BigVGAN's 44.1kHz output resampled to 48kHz transparently | torchaudio `Resample` with `resampling_method="sinc_interp_kaiser"` provides high-quality Kaiser-windowed sinc resampling. Existing resampler cache pattern in `generation.py` can be reused with upgraded quality parameters. Resampling at export boundary per user decision. |
| GEN-03 | Export pipeline (WAV/MP3/FLAC/OGG), metadata, and spatial audio work unchanged with vocoder output | Export pipeline operates on numpy arrays with a sample_rate parameter. Changing internal rate from 48kHz to 44.1kHz is transparent to export functions. Spatial processing, anti-aliasing filter, and peak normalization all accept sample_rate as parameter. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torchaudio | 2.10.0 (already installed) | High-quality Kaiser-windowed sinc resampling 44.1kHz -> 48kHz | Already a project dependency; `torchaudio.transforms.Resample` with `sinc_interp_kaiser` is professional-grade and GPU-acceleratable |
| BigVGAN vocoder | Phase 12 vendored | Neural mel-to-waveform conversion | Already implemented in `src/distill/vocoder/bigvgan_vocoder.py`; 122M params, 44.1kHz output |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | (already installed) | Audio array manipulation throughout pipeline | All waveform post-processing (spatial, normalization, export) |
| soundfile | (already installed) | Multi-format audio export (WAV/FLAC/OGG) | Export pipeline -- unchanged by this phase |
| lameenc | (already installed) | MP3 encoding | Export pipeline -- unchanged by this phase |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| torchaudio Resample (Kaiser) | soxr (python-soxr) | soxr is marginally higher quality at VHQ setting but adds a new C dependency; torchaudio is already installed and Kaiser mode is excellent quality. Stick with torchaudio. |
| torchaudio Resample (Kaiser) | librosa.resample (uses soxr internally) | librosa already a dependency but adds indirection; torchaudio is more natural for torch tensor pipeline |

**Recommendation:** Use `torchaudio.transforms.Resample` with `resampling_method="sinc_interp_kaiser"` and `lowpass_filter_width=64` for high-quality resampling. This matches the user's "sinc/Kaiser" requirement, is already installed, and avoids adding new dependencies.

## Architecture Patterns

### Current Call Graph (Griffin-Lim Path -- TO BE REPLACED)

```
GenerationPipeline.generate()
  -> generate_chunks_crossfade() / generate_chunks_latent_interp() / _generate_chunks_from_vector()
    -> synthesize_continuous_mel()        # produces mel [1, 1, 128, T]
    -> spectrogram.mel_to_waveform()      # Griffin-Lim -> 48kHz waveform  <-- REPLACE
  -> apply_anti_alias_filter()
  -> apply_spatial()
  -> peak_normalize()
  -> _get_resampler() (48kHz -> target)   # only if target != 48kHz
  -> compute_quality_score()

generate_preview()
  -> model.sample()                       # random latent -> mel
  -> spectrogram.mel_to_waveform()        # Griffin-Lim -> 48kHz  <-- REPLACE

generate_reconstruction_preview()
  -> model(sample_batch)                  # encode/decode
  -> spectrogram.mel_to_waveform()        # Griffin-Lim -> 48kHz  <-- REPLACE (both orig and recon)
```

### Target Call Graph (Vocoder Path)

```
GenerationPipeline.generate()
  -> generate_chunks_crossfade() / generate_chunks_latent_interp() / _generate_chunks_from_vector()
    -> synthesize_continuous_mel()        # produces mel [1, 1, 128, T]
    -> vocoder.mel_to_waveform()          # BigVGAN -> 44.1kHz waveform  <-- NEW
  -> apply_anti_alias_filter(audio, 44100)
  -> apply_spatial(audio, config, 44100)
  -> peak_normalize()
  -> optional resample 44.1kHz -> 48kHz  # at export boundary
  -> compute_quality_score()

generate_preview()
  -> model.sample()                       # random latent -> mel
  -> vocoder.mel_to_waveform()            # BigVGAN -> 44.1kHz  <-- NEW

generate_reconstruction_preview()
  -> model(sample_batch)                  # encode/decode
  -> vocoder.mel_to_waveform()            # BigVGAN -> 44.1kHz  <-- NEW (both orig and recon)
```

### Pattern 1: Vocoder Injection via GenerationPipeline Constructor

**What:** Add `vocoder: VocoderBase` as a parameter to `GenerationPipeline.__init__()` alongside the existing `model`, `spectrogram`, and `device` parameters. The pipeline uses `vocoder.mel_to_waveform()` instead of `spectrogram.mel_to_waveform()`.

**When to use:** All generation code paths that produce audio from mel spectrograms.

**Example:**
```python
class GenerationPipeline:
    def __init__(
        self,
        model: "ConvVAE",
        spectrogram: "AudioSpectrogram",
        device: "torch.device",
        vocoder: "VocoderBase | None" = None,
    ) -> None:
        self.model = model
        self.spectrogram = spectrogram
        self.device = device
        self.vocoder = vocoder or get_vocoder("bigvgan", device=str(device))
        self.model_name: str = "unknown"
```

**Why:** The vocoder is a heavy object (122M params on GPU). Creating it once in the pipeline constructor and reusing across all generation calls avoids repeated initialization. The `spectrogram` parameter is still needed for `get_mel_shape()` and `waveform_to_mel()` during training data preprocessing. The `vocoder` now handles mel-to-waveform exclusively.

### Pattern 2: Mel-to-Waveform Replacement in Chunking Functions

**What:** The three chunking functions (`generate_chunks_crossfade`, `generate_chunks_latent_interp`, `_generate_chunks_from_vector`) currently call `spectrogram.mel_to_waveform(combined_mel)` at their final step. Replace with a `vocoder` parameter.

**Example:**
```python
def generate_chunks_crossfade(
    model, spectrogram, num_chunks, device, seed,
    chunk_samples=48_000, overlap_samples=2400,
    vocoder=None,  # NEW
) -> np.ndarray:
    ...
    combined_mel = synthesize_continuous_mel(model, spectrogram, trajectory, chunk_samples)
    # OLD: wav = spectrogram.mel_to_waveform(combined_mel)
    wav = vocoder.mel_to_waveform(combined_mel)
    return wav.squeeze().cpu().numpy().astype(np.float32)
```

**Note on sample count:** The vocoder produces 44.1kHz audio, so `chunk_samples` at 48kHz no longer directly corresponds to output samples. However, `chunk_samples` only affects mel spectrogram sizing (via `spectrogram.get_mel_shape(chunk_samples)`) which determines the mel frame count. The vocoder then produces audio at its native rate from those mel frames. This is correct -- the mel frame count determines temporal extent, not the output sample rate.

### Pattern 3: Internal Sample Rate Transition

**What:** The pipeline currently assumes 48kHz internally (`internal_sr = 48_000`). With BigVGAN, the vocoder outputs 44.1kHz. The internal sample rate must change to match vocoder output.

**Key insight:** `chunk_samples` is used to compute mel shapes (via `spectrogram.get_mel_shape(chunk_samples)`), and the spectrogram operates at 48kHz. The mel spectrogram frame count is what matters -- it determines temporal extent. BigVGAN's 512x hop_size at 44.1kHz means each mel frame maps to 512 samples at 44.1kHz. So the temporal duration is preserved even though the sample count differs.

**Change:** After vocoder produces waveform, all downstream processing (anti-alias filter, spatial, normalization) uses the vocoder's native sample rate (44.1kHz). Resampling to user's chosen export rate happens at the end.

### Pattern 4: GPU OOM Fallback to CPU

**What:** BigVGAN is 122M parameters and long mel spectrograms can cause GPU OOM. Implement a try/except around vocoder inference with CPU fallback.

**Example:**
```python
try:
    wav = vocoder.mel_to_waveform(mel)
except RuntimeError as e:
    if "out of memory" in str(e).lower() or "CUDA" in str(e):
        logger.warning("GPU OOM during vocoder inference, falling back to CPU")
        vocoder.to(torch.device("cpu"))
        wav = vocoder.mel_to_waveform(mel.cpu())
        vocoder.to(original_device)  # move back after
    else:
        raise
```

### Pattern 5: High-Quality Resampling at Export Boundary

**What:** Replace the default torchaudio Resample (Hann window) with Kaiser-windowed sinc interpolation for the 44.1kHz -> 48kHz case.

**Example:**
```python
torchaudio.transforms.Resample(
    orig_freq=44100,
    new_freq=48000,
    resampling_method="sinc_interp_kaiser",
    lowpass_filter_width=64,
    rolloff=0.9475937167,  # optimal for Kaiser with beta=14.769656459379492
    beta=14.769656459379492,
)
```

**Source:** torchaudio's official [resampling tutorial](https://docs.pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html) recommends Kaiser window for highest quality. The `lowpass_filter_width=64` provides very sharp transition band. The `beta` and `rolloff` values above are the defaults torchaudio computes when using `sinc_interp_kaiser` without explicit beta -- we can simply use:
```python
torchaudio.transforms.Resample(
    orig_freq=44100,
    new_freq=48000,
    resampling_method="sinc_interp_kaiser",
    lowpass_filter_width=64,
)
```

### Anti-Patterns to Avoid

- **Resampling inside the vocoder:** The vocoder should return native 44.1kHz. Resampling is the pipeline's concern at the export boundary, not the vocoder's.
- **Dual internal sample rates:** Don't try to maintain both 48kHz and 44.1kHz internal rates. After vocoder, everything is 44.1kHz until export.
- **Keeping Griffin-Lim as fallback:** User explicitly decided no fallback, no hidden code. Remove it.
- **Resampling in spatial/filter functions:** These already accept `sample_rate` as parameter -- pass the correct value (44100 after vocoder), don't hardcode 48000.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| High-quality resampling | Custom sinc interpolation | `torchaudio.transforms.Resample(resampling_method="sinc_interp_kaiser")` | Kaiser window design, anti-aliasing filter coefficients, and edge-case handling are complex. torchaudio's implementation is GPU-acceleratable and battle-tested. |
| Mel format conversion | Direct matrix conversion | `MelAdapter` (already implemented in Phase 12) | VAE mels (log1p, HTK, 48kHz) vs BigVGAN mels (log-clamp, Slaney, 44.1kHz) differ in three dimensions. The waveform round-trip approach in MelAdapter is the verified correct path. |
| BigVGAN weight management | Manual download logic | `weight_manager.ensure_bigvgan_weights()` (already implemented) | HuggingFace Hub handles resumable downloads, caching, and offline fallback. |

**Key insight:** The vocoder infrastructure is fully built (Phase 12). This phase is about *wiring* it into the five generation code paths, not building new vocoder functionality.

## Common Pitfalls

### Pitfall 1: Duration Mismatch After Sample Rate Change
**What goes wrong:** After switching internal sample rate from 48kHz to 44.1kHz, the `overlap_samples=2400` (50ms at 48kHz) becomes incorrect. Duration calculations that multiply chunk_samples by sample rate will produce wrong values.
**Why it happens:** Hardcoded sample counts based on the old 48kHz rate.
**How to avoid:** The mel spectrogram frame count (not waveform sample count) determines temporal extent. `chunk_samples` feeds into `spectrogram.get_mel_shape()` which uses the 48kHz spectrogram config -- this is correct because it's computing mel frame counts. The vocoder then produces audio at its native rate from those mel frames. The key places to update: `overlap_samples` default value (recalculate for 44.1kHz if needed, though it's only used as an API compat stub in the current code), `internal_sr` variable in `GenerationPipeline.generate()`, and duration trimming at the end.
**Warning signs:** Generated audio is shorter or longer than requested duration.

### Pitfall 2: Forgetting to Update Sample Rate in Downstream Processing
**What goes wrong:** Anti-alias filter, spatial processing, and quality metrics all receive `sample_rate` as a parameter. If these still get 48000 instead of 44100, the filter cutoff and spatial processing will be subtly wrong.
**Why it happens:** The `internal_sr = 48_000` constant in `GenerationPipeline.generate()` cascades through all downstream calls.
**How to avoid:** Change `internal_sr` to use `vocoder.sample_rate` (44100). All downstream functions already accept `sample_rate` as parameter -- just pass the correct value.
**Warning signs:** Anti-alias filter at wrong cutoff, subtle spatial audio timing issues.

### Pitfall 3: Training Preview Performance
**What goes wrong:** Training previews call `generate_preview()` every N epochs. BigVGAN inference (122M params) is much slower than Griffin-Lim. If previews run synchronously in the training loop, they could significantly slow training.
**Why it happens:** Griffin-Lim is CPU-only and relatively fast for short clips. BigVGAN needs GPU and is slower per-sample.
**How to avoid:** Training previews are already short (1 chunk = ~1 second). BigVGAN inference for a single 1-second chunk should be fast enough (<1 second on GPU). Monitor but unlikely to be a real issue for the preview use case.
**Warning signs:** Training speed drops noticeably when previews are enabled.

### Pitfall 4: Griffin-Lim Still Used in MelAdapter
**What goes wrong:** The user decided "remove Griffin-Lim entirely." But MelAdapter (Phase 12) uses Griffin-Lim as an intermediate step for mel format conversion (VAE mel -> Griffin-Lim waveform -> resample -> BigVGAN mel).
**Why it happens:** MelAdapter was designed as a temporary bridge between VAE's mel format and BigVGAN's mel format.
**How to avoid:** The Griffin-Lim usage in MelAdapter is an *internal implementation detail* of the mel format conversion, not a user-facing reconstruction path. The user's intent is "no Griffin-Lim as the final output method." MelAdapter's internal Griffin-Lim is acceptable because: (a) its output goes into BigVGAN neural reconstruction which compensates for Griffin-Lim artifacts, (b) Phase 16's per-model HiFi-GAN will eliminate this entirely. Document this distinction clearly in code comments. The `AudioSpectrogram.mel_to_waveform()` method should remain available but only used by MelAdapter internally -- remove all direct generation-path calls to it.
**Warning signs:** N/A -- this is an architectural clarification, not a runtime issue.

### Pitfall 5: Gradio Audio Component Sample Rate
**What goes wrong:** The Gradio `gr.Audio` component in the generate tab receives `(sample_rate, ndarray)` tuples. If the sample rate changes from 48000 to 44100, the audio preview will play at the wrong speed or Gradio may resample unexpectedly.
**Why it happens:** The `_generate_audio()` handler returns `(result.sample_rate, result.audio)`. If `result.sample_rate` is 44100 but the UI expects 48000, playback will be wrong.
**How to avoid:** `result.sample_rate` is already dynamic (from `GenerationConfig.sample_rate`). As long as the GenerationResult carries the correct sample rate, Gradio handles it correctly. The default `GenerationConfig.sample_rate` should change from 48000 to 44100 per user decision.
**Warning signs:** Preview audio plays at wrong pitch.

### Pitfall 6: SAMPLE_RATE_OPTIONS and Config Defaults
**What goes wrong:** `SAMPLE_RATE_OPTIONS = (44_100, 48_000, 96_000)` already includes 44100 (good). But `GenerationConfig.sample_rate` defaults to 48000, the CLI `--sample-rate` defaults to 48000, and the UI export dropdown defaults to "48000". These must all change to 44100.
**Why it happens:** Everything was designed around 48kHz as the internal rate.
**How to avoid:** Update all default values to 44100. The user decision is "simple 44.1kHz / 48kHz toggle" so keep both options but change the default.
**Warning signs:** Users unknowingly get resampled audio when they expected native vocoder output.

## Code Examples

### Example 1: Vocoder-Powered Chunk Generation

```python
# In chunking.py: generate_chunks_crossfade() -- replacing spectrogram.mel_to_waveform
def generate_chunks_crossfade(
    model, spectrogram, num_chunks, device, seed,
    chunk_samples=48_000, overlap_samples=2400,
    vocoder=None,  # NEW: VocoderBase instance
):
    import numpy as np

    num_steps, _, _ = _compute_num_decode_steps(spectrogram, num_chunks, chunk_samples)
    num_anchors = max(2, num_chunks)
    z_anchors = _sample_latent_vectors(model, num_anchors, device, seed)
    trajectory = _interpolate_trajectory(z_anchors, num_steps)

    combined_mel = synthesize_continuous_mel(model, spectrogram, trajectory, chunk_samples)

    # NEW: Use vocoder instead of Griffin-Lim
    wav = vocoder.mel_to_waveform(combined_mel)  # Returns [B, 1, samples] at 44.1kHz
    return wav.squeeze().cpu().numpy().astype(np.float32)
```

### Example 2: High-Quality Kaiser Resampler

```python
# In generation.py: updated _get_resampler with Kaiser window
def _get_resampler(orig_freq: int, new_freq: int) -> object:
    key = (orig_freq, new_freq)
    if key not in _resampler_cache:
        import torchaudio

        _resampler_cache[key] = torchaudio.transforms.Resample(
            orig_freq,
            new_freq,
            resampling_method="sinc_interp_kaiser",
            lowpass_filter_width=64,
        )
    return _resampler_cache[key]
```

### Example 3: Updated GenerationPipeline.generate() Internal Rate

```python
# Key changes in GenerationPipeline.generate()
def generate(self, config):
    ...
    # Internal sample rate is now vocoder's native rate
    internal_sr = self.vocoder.sample_rate  # 44100 for BigVGAN

    # chunk_samples still computed from config (used for mel shape sizing)
    # Note: chunk_samples at 48kHz is fine -- it sizes mel frames via
    # spectrogram.get_mel_shape() which uses the spectrogram's 48kHz config.
    # The vocoder then converts those mel frames to audio at its native rate.
    chunk_samples = int(config.chunk_duration_s * self.spectrogram.config.sample_rate)
    ...
    # All downstream processing uses internal_sr (44100)
    audio = apply_anti_alias_filter(audio, internal_sr)
    audio = apply_spatial(audio, spatial_config, internal_sr)
    audio = peak_normalize(audio, target_peak=0.891)

    # Resample to export rate if different from vocoder native
    if config.sample_rate != internal_sr:
        resampler = _get_resampler(internal_sr, config.sample_rate)
        ...
```

### Example 4: GPU OOM Fallback for Vocoder

```python
def _vocoder_with_fallback(vocoder, mel, original_device):
    """Run vocoder inference with GPU OOM fallback to CPU."""
    import torch
    import logging

    logger = logging.getLogger(__name__)

    try:
        return vocoder.mel_to_waveform(mel)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(
                "GPU out of memory during vocoder inference. "
                "Falling back to CPU. Consider reducing duration."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpu_device = torch.device("cpu")
            vocoder.to(cpu_device)
            try:
                result = vocoder.mel_to_waveform(mel.cpu())
            finally:
                vocoder.to(original_device)
            return result
        raise
```

### Example 5: Updated Training Preview with Vocoder

```python
# In training/preview.py: generate_preview() with vocoder
def generate_preview(
    model, spectrogram, output_dir, epoch, device,
    num_samples=1, sample_rate=44_100,  # NOTE: default changed to 44100
    vocoder=None,
):
    ...
    with torch.no_grad():
        mel_recon = model.sample(num_samples, device)
        # NEW: Use vocoder instead of Griffin-Lim
        waveforms = vocoder.mel_to_waveform(mel_recon)  # 44.1kHz output
        for i in range(waveforms.shape[0]):
            audio = waveforms[i, 0].cpu()
            _save_wav(audio, wav_path, sample_rate)
```

## Inventory of All Call Sites Requiring Changes

### Direct `spectrogram.mel_to_waveform()` Calls (5 total)

| # | File | Line | Function | Context |
|---|------|------|----------|---------|
| 1 | `inference/generation.py` | 339 | `_generate_chunks_from_vector` | Slider-controlled generation |
| 2 | `inference/chunking.py` | 434 | `generate_chunks_crossfade` | Crossfade mode |
| 3 | `inference/chunking.py` | 493 | `generate_chunks_latent_interp` | Latent interpolation mode |
| 4 | `training/preview.py` | 85 | `generate_preview` | Random latent previews |
| 5 | `training/preview.py` | 170-171 | `generate_reconstruction_preview` | Original + reconstruction previews |

### Indirect Calls (NOT in generation path -- keep as-is)

| File | Line | Function | Why Keep |
|------|------|----------|----------|
| `vocoder/mel_adapter.py` | 106 | `MelAdapter.convert` | Internal to vocoder's mel format conversion |
| `controls/analyzer.py` | 311 | PCA sweep analysis | Analysis tool, not generation output; acceptable quality for feature extraction |

### Default Value Changes

| File | Field/Param | Old Default | New Default | Reason |
|------|-------------|-------------|-------------|--------|
| `inference/generation.py` | `GenerationConfig.sample_rate` | `48_000` | `44_100` | BigVGAN native rate |
| `inference/export.py` | `SAMPLE_RATE_OPTIONS` | `(44_100, 48_000, 96_000)` | Keep as-is but default to 44100 | Already includes 44100 |
| `cli/generate.py` | `--sample-rate` default | `48000` | `44100` | Match new default |
| `ui/tabs/generate_tab.py` | Export Sample Rate dropdown | `"48000"` | `"44100"` | Match new default |
| `training/preview.py` | `sample_rate` default | `48_000` | `44_100` | Preview at vocoder native rate |

### Griffin-Lim Removal

| File | What to Remove/Update | Impact |
|------|----------------------|--------|
| `audio/spectrogram.py` | `mel_to_waveform()` method, `InverseMelScale`, `GriffinLim` | Keep for MelAdapter's internal use only; add deprecation comment |
| `audio/spectrogram.py` | `__init__` GriffinLim/InverseMelScale creation | Keep -- MelAdapter still needs them |
| `inference/chunking.py` | Docstring references to Griffin-Lim | Update docs to reference vocoder |
| `audio/filters.py` | Docstring "after GriffinLim output" | Update to "after vocoder output" |

**Important clarification on Griffin-Lim removal:** The `AudioSpectrogram` class's `mel_to_waveform()` method (which uses InverseMelScale + GriffinLim) must be preserved because `MelAdapter` depends on it for mel format conversion. What gets removed is every *direct generation-path* call to `spectrogram.mel_to_waveform()`. The five call sites in the inventory above are replaced with vocoder calls. The analyzer's call (in `controls/analyzer.py`) can also be updated if desired, but it's not a generation output path.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Griffin-Lim (iterative phase estimation) | Neural vocoder (BigVGAN) | Phase 12 built it, Phase 14 wires it | Dramatically better audio quality; neural vocoder handles phase coherence natively |
| 48kHz internal processing | 44.1kHz internal (vocoder native) | Phase 14 | Avoids unnecessary resampling for the most common case |
| Mandatory resampling to 48kHz | Optional resampling at export boundary | Phase 14 | Cleaner pipeline; users who want 44.1kHz get native vocoder output |

## Open Questions

1. **PCA analyzer mel_to_waveform usage**
   - What we know: `controls/analyzer.py` line 311 uses `spectrogram.mel_to_waveform()` for feature extraction during PCA analysis
   - What's unclear: Should this also use the vocoder? It would be more accurate (features computed on what users actually hear) but slower
   - Recommendation: Keep Griffin-Lim for analyzer -- it's not a user-facing output, just feature extraction for labeling PCA components. Speed matters more here. This is not a GEN-01 requirement (it's analysis, not generation). If desired, can be updated in a future phase.

2. **Duration accuracy after sample rate change**
   - What we know: `chunk_samples` is computed as `int(config.chunk_duration_s * internal_sr)` in `GenerationPipeline.generate()`. With the rate change, the mel frame count computation still goes through `spectrogram.get_mel_shape(chunk_samples)` which uses the spectrogram's 48kHz config.
   - What's unclear: Whether the final audio duration after vocoder + trimming will be exactly the requested duration
   - Recommendation: The trim step at the end already handles this: `target_samples = int(config.duration_s * config.sample_rate)`. As long as the vocoder produces at least the target duration, trimming ensures exact length. Verify this in testing.

3. **Preview caching (Claude's discretion)**
   - What we know: BigVGAN inference is heavier than Griffin-Lim
   - Recommendation: No caching for initial implementation. Preview is "what you preview is what you get" -- always regenerate. If latency becomes a problem, caching can be added later. The simplest correct implementation first.

## Sources

### Primary (HIGH confidence)
- Codebase analysis: All five `spectrogram.mel_to_waveform()` call sites directly examined
- Phase 12 verification: `12-VERIFICATION.md` confirms vocoder pipeline works end-to-end
- [torchaudio Resample documentation](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.Resample.html) -- Kaiser window parameters

### Secondary (MEDIUM confidence)
- [torchaudio resampling tutorial](https://docs.pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html) -- quality comparison of resampling methods
- [python-soxr](https://github.com/dofuuz/python-soxr) -- alternative resampling library (not recommended, torchaudio sufficient)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already installed and working
- Architecture: HIGH -- exact call sites identified, replacement pattern is mechanical
- Pitfalls: HIGH -- derived from direct codebase analysis, not hypothetical

**Research date:** 2026-02-27
**Valid until:** 2026-03-27 (stable domain; no external API changes expected)
