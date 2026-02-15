# Phase 5: Musically Meaningful Controls - Research

**Researched:** 2026-02-13
**Domain:** PCA-based latent space analysis, audio feature extraction, parameter mapping, slider control logic
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Parameter mapping approach:**
- PCA discovery on trained latent space (not predefined targets)
- Adaptive slider count -- expose as many PCA components as exceed a variance threshold (could be 3 or 12 depending on model)
- Neutral axis labels by default ("Axis 1", "Axis 2") with suggested labels based on audio feature correlation; user can accept or rename
- User-triggered analysis (not automatic after training) -- user decides when to run "Analyze latent space"

**Slider behavior & ranges:**
- Stepped sliders (discrete positions, not continuous) for repeatability
- Soft warning zone at extreme values -- visual indicator but slider still allows full range; no hard clamp
- "Randomize all" button to set all sliders to random positions within safe bounds
- No per-slider randomize (global only)

**Parameter categories:**
- All parameter families weighted equally -- timbre, harmony, temporal, spatial all matter; no priority
- Acoustic terms for labels when accurate (spectral centroid, RMS energy, zero-crossing rate) -- precision over producer jargon
- Fully independent sliders -- each maps to exactly one PCA component, no coupling
- Global "reset to center" button (returns all sliders to latent space mean)
- No per-slider reset

**Discovery & calibration:**
- Analysis data: training data encodings + random prior samples combined for full coverage
- Graceful degradation -- show whatever dimensions exist, even if only 1-2; warn user that more data might improve variety
- Analysis results (PCA mapping, labels, safe ranges) saved with the model checkpoint -- instant slider restoration on load

### Claude's Discretion
- Slider visual feedback approach (label + value only, or label + descriptor, etc.)
- Whether to expose variance-explained percentages per slider
- Exact variance threshold for determining "meaningful" PCA components
- Compression algorithm for analysis data within checkpoint

### Deferred Ideas (OUT OF SCOPE)
- Coupled/correlated parameters (moving one slider subtly influences others for more natural musical control) -- revisit after v1
- Per-slider randomize buttons -- could add later if users want targeted exploration
- Per-slider reset-to-center -- could add alongside per-slider randomize

</user_constraints>

## Summary

Phase 5 builds the bridge between the VAE's opaque 64-dimensional latent space and human-understandable musical controls. The approach is: (1) encode training data through the trained VAE encoder to collect latent vectors, (2) run PCA on those vectors to find orthogonal directions of maximum variance, (3) correlate each PCA component with computed audio features (spectral centroid, RMS energy, zero-crossing rate, spectral rolloff, spectral flatness) to suggest human-readable labels, (4) determine safe ranges and step sizes for each component, and (5) persist all analysis results alongside the model checkpoint for instant restoration.

The technical foundation is solid. Research confirms that VAEs naturally pursue PCA-like directions (Rolinek et al., CVPR 2019), meaning PCA on a VAE latent space recovers explicit variance ordering and provides clean, orthogonal control axes. The MIDISpace paper (Valero-Mas et al., 2022) demonstrates that PCA applied to music VAE latent spaces finds "largely disentangled directions that change the style and characteristics" of generated content, with directions that are "often monotonic, global and encode fundamental musical characteristics." This validates the user's chosen approach of PCA discovery over predefined targets.

The stack decision is straightforward: scikit-learn 1.8.0 for PCA (already proven, well-documented API), numpy + scipy for audio feature computation (avoid adding librosa which drags in numba), and torchaudio's SpectralCentroid for that specific feature. The existing checkpoint system (`training/checkpoint.py`) already saves arbitrary dict data via `torch.save`, so persisting PCA results is a matter of adding analysis data to the checkpoint dict. No new heavy dependencies are needed -- scikit-learn is the only addition.

**Primary recommendation:** Build a `LatentSpaceAnalyzer` class in `src/small_dataset_audio/controls/` that encapsulates PCA fitting, audio feature correlation, safe range computation, and serialization. Use scikit-learn PCA for dimensionality analysis, compute audio features with numpy/scipy/torchaudio (NOT librosa), and store analysis results as numpy arrays within the existing checkpoint format. The slider-to-latent mapping is: `z = mean + sum(slider_value_i * component_i)` where components come from PCA.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-learn | >=1.8.0 | PCA decomposition, variance analysis | Standard ML library; PCA is its bread and butter; `explained_variance_ratio_` provides exactly what we need for adaptive slider count |
| numpy | >=1.26 | Audio feature computation, correlation matrices, array operations | Already in project; sufficient for RMS, zero-crossing rate, spectral rolloff, spectral flatness |
| scipy | >=1.12 | `scipy.stats.pearsonr` for feature-component correlation, `scipy.signal.stft` for spectral analysis | Already in project; provides statistical correlation with p-values |
| torch | >=2.10.0 | Model inference (encoding training data), checkpoint persistence | Already in project |
| torchaudio | >=2.10.0 | `SpectralCentroid` transform for centroid computation | Already in project; verified available in 2.10 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| (none -- no new supporting libraries needed) | | | |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| scikit-learn PCA | Manual SVD with numpy.linalg.svd | sklearn wraps SVD with `explained_variance_ratio_`, `inverse_transform`, and `mean_` -- reimplementing these is error-prone and gains nothing |
| numpy/scipy for audio features | librosa 0.11.0 | librosa provides spectral_centroid, rms, zero_crossing_rate etc. but requires numba (heavy JIT compiler) as transitive dependency; the features are simple formulas we can compute directly |
| Pearson correlation (scipy.stats.pearsonr) | Spearman rank correlation | Pearson is correct for linear relationships, which is what PCA produces; Spearman would overestimate correlations for this use case |

**Installation:**
```bash
uv add "scikit-learn>=1.8"
```

## Architecture Patterns

### Recommended Project Structure
```
src/small_dataset_audio/
├── controls/
│   ├── __init__.py           # Public API exports
│   ├── analyzer.py           # LatentSpaceAnalyzer class (PCA + feature correlation)
│   ├── features.py           # Audio feature extraction (spectral centroid, RMS, ZCR, etc.)
│   ├── mapping.py            # Slider-to-latent vector conversion logic
│   └── serialization.py      # Save/load analysis results to/from checkpoint
├── inference/
│   └── generation.py         # [MODIFY] Accept latent vector from slider mapping
├── training/
│   └── checkpoint.py         # [MODIFY] Include analysis data in checkpoint dict
```

### Pattern 1: LatentSpaceAnalyzer (Core Analysis Class)
**What:** A class that orchestrates the full analysis pipeline: encode data, fit PCA, compute audio features, correlate features with components, determine safe ranges, and produce a serializable result.
**When to use:** When user triggers "Analyze latent space" in the UI.
**Example:**
```python
# Source: scikit-learn PCA docs + project pattern
from dataclasses import dataclass, field
import numpy as np

@dataclass
class AnalysisResult:
    """Serializable result of latent space analysis."""
    # PCA data
    pca_components: np.ndarray       # [n_components, latent_dim] -- principal axes
    pca_mean: np.ndarray             # [latent_dim] -- latent space mean
    explained_variance_ratio: np.ndarray  # [n_components] -- % variance per axis
    n_active_components: int          # Number exceeding variance threshold

    # Per-component metadata
    component_labels: list[str]       # ["Axis 1 (spectral centroid)", ...]
    suggested_labels: list[str]       # ["spectral centroid", "RMS energy", ...]
    user_labels: list[str]            # User-overridden labels (initially empty)

    # Range data
    safe_min: np.ndarray             # [n_components] -- safe lower bound per axis
    safe_max: np.ndarray             # [n_components] -- safe upper bound per axis
    warning_min: np.ndarray          # [n_components] -- warning zone lower bound
    warning_max: np.ndarray          # [n_components] -- warning zone upper bound
    step_size: np.ndarray            # [n_components] -- discrete step size per axis

    # Correlation data
    feature_correlations: dict       # {feature_name: [corr_per_component]}

    def to_dict(self) -> dict:
        """Convert to dict of numpy arrays + primitives for checkpoint storage."""
        ...

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisResult":
        """Reconstruct from checkpoint dict."""
        ...
```

### Pattern 2: Slider-to-Latent Vector Mapping
**What:** Convert discrete slider positions to a latent vector in the original 64-dimensional space using PCA inverse transform. Each slider controls exactly one PCA component; the result is reconstructed via `z = mean + sum(value_i * component_i)`.
**When to use:** Every time a slider value changes or generation is triggered.
**Example:**
```python
def sliders_to_latent(
    slider_values: np.ndarray,          # [n_active_components] -- current slider positions
    analysis: AnalysisResult,
) -> np.ndarray:
    """Convert slider positions to a 64-dimensional latent vector.

    Each slider value is a position along one PCA component.
    The result is the mean plus the weighted sum of components.
    """
    # slider_values are in PCA space (centered, scaled)
    # Reconstruct to original latent space
    components = analysis.pca_components[:analysis.n_active_components]
    z = analysis.pca_mean + slider_values @ components
    return z  # shape [latent_dim]
```

### Pattern 3: Audio Feature Correlation for Label Suggestion
**What:** For each PCA component, generate audio at multiple points along that axis (sweeping from min to max), extract audio features from each sample, and compute Pearson correlation between the component value and each audio feature. The feature with highest absolute correlation becomes the suggested label.
**When to use:** During the "Analyze latent space" operation, after PCA fitting.
**Example:**
```python
def correlate_component_with_features(
    model: "ConvVAE",
    spectrogram: "AudioSpectrogram",
    analysis_result: "AnalysisResult",
    component_idx: int,
    n_samples: int = 20,
    device: "torch.device" = None,
) -> dict[str, float]:
    """Sweep one PCA component, decode audio, compute feature correlations.

    Returns {feature_name: pearson_r} for each audio feature.
    """
    import torch
    from scipy.stats import pearsonr

    positions = np.linspace(
        analysis_result.safe_min[component_idx],
        analysis_result.safe_max[component_idx],
        n_samples,
    )

    feature_values = {name: [] for name in FEATURE_NAMES}

    for pos in positions:
        # Build latent vector with only this component varying
        slider_vals = np.zeros(analysis_result.n_active_components)
        slider_vals[component_idx] = pos
        z = sliders_to_latent(slider_vals, analysis_result)

        # Decode to audio
        z_tensor = torch.from_numpy(z).float().unsqueeze(0).to(device)
        with torch.no_grad():
            mel = model.decode(z_tensor, target_shape=mel_shape)
        wav = spectrogram.mel_to_waveform(mel.cpu()).squeeze().numpy()

        # Extract features
        features = extract_audio_features(wav, sample_rate=48_000)
        for name, val in features.items():
            feature_values[name].append(val)

    # Compute Pearson correlation for each feature
    correlations = {}
    for name, values in feature_values.items():
        r, p = pearsonr(positions, values)
        if p < 0.05:  # Only include statistically significant correlations
            correlations[name] = float(r)

    return correlations
```

### Pattern 4: Encoding Training Data for PCA
**What:** Pass all training data through the VAE encoder to collect mu vectors (the latent means), plus sample random vectors from the prior (standard normal). Combine both sets as input to PCA.
**When to use:** First step of the "Analyze latent space" operation.
**Example:**
```python
def collect_latent_vectors(
    model: "ConvVAE",
    dataset: "torch.utils.data.Dataset",
    device: "torch.device",
    n_prior_samples: int = 500,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode training data + sample prior to build PCA input matrix.

    Returns array of shape [n_total, latent_dim].
    """
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    all_mu = []

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            mel = batch.to(device)  # [B, 1, n_mels, time]
            mu, _ = model.encode(mel)
            all_mu.append(mu.cpu().numpy())

    # Add random prior samples for coverage
    prior_samples = np.random.randn(n_prior_samples, model.latent_dim)
    training_mu = np.concatenate(all_mu, axis=0)

    return np.concatenate([training_mu, prior_samples], axis=0)
```

### Pattern 5: Safe Range Computation
**What:** Determine the operational range for each PCA component based on the distribution of training data projections. Use percentiles (e.g., 2nd-98th) for safe bounds and (0.5th-99.5th) for warning zones. Step size is derived from the range divided by a target number of discrete steps.
**When to use:** After PCA fitting, before building slider metadata.
**Example:**
```python
def compute_safe_ranges(
    projected: np.ndarray,   # [n_samples, n_components] -- PCA-projected data
    n_steps: int = 21,       # Number of discrete slider positions (odd for center)
    safe_percentile: float = 2.0,
    warn_percentile: float = 0.5,
) -> dict:
    """Compute safe and warning ranges for each PCA component."""
    n_components = projected.shape[1]

    safe_min = np.percentile(projected, safe_percentile, axis=0)
    safe_max = np.percentile(projected, 100 - safe_percentile, axis=0)
    warn_min = np.percentile(projected, warn_percentile, axis=0)
    warn_max = np.percentile(projected, 100 - warn_percentile, axis=0)

    # Step size: divide safe range into n_steps discrete positions
    step_size = (safe_max - safe_min) / (n_steps - 1)

    return {
        "safe_min": safe_min,
        "safe_max": safe_max,
        "warning_min": warn_min,
        "warning_max": warn_max,
        "step_size": step_size,
    }
```

### Anti-Patterns to Avoid
- **Running PCA on raw training audio instead of latent vectors:** PCA should operate on the VAE's encoded mu vectors, not on mel spectrograms or raw audio. The latent space is already a compressed, structured representation.
- **Using continuous sliders when user requested stepped:** The decision locks in discrete positions for repeatability. The slider logic must snap to discrete step values.
- **Hard-clamping slider range:** The user decided on soft warning zones, not hard limits. Sliders must allow the full range (including warning zones and beyond), just with visual indicators.
- **Fitting PCA on prior samples only:** The analysis data must include both training data encodings AND prior samples. Training encodings show where the model has learned structure; prior samples fill in coverage gaps.
- **Correlating features across all components simultaneously:** Correlate one component at a time by sweeping it while holding others at zero (the mean). This isolates each component's effect.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PCA decomposition | Manual SVD + mean centering + variance calculation | `sklearn.decomposition.PCA` | sklearn handles numerical stability (svd_solver selection), provides `explained_variance_ratio_`, `mean_`, `components_`, `inverse_transform` -- reimplementing all of these correctly is non-trivial |
| Spectral centroid | Custom FFT + weighted average | `torchaudio.transforms.SpectralCentroid` | Already in project; handles windowing, hop_length, and frequency bin computation correctly |
| Variance threshold selection | Manual cumulative sum loop | `pca.explained_variance_ratio_` with `np.cumsum` | One-liner vs error-prone loop; the ratio is pre-computed by sklearn |
| Latent vector reconstruction | Manual matrix multiply from slider values | `pca.inverse_transform()` or `mean + values @ components` | sklearn's inverse_transform handles the centering/uncentering automatically |

**Key insight:** The PCA + feature correlation pipeline has many moving parts (mean centering, variance normalization, orthogonal decomposition, statistical testing, range computation) where small errors compound. Using sklearn's PCA eliminates an entire class of numerical issues. Audio features like RMS and zero-crossing rate ARE simple enough to hand-compute with numpy, but PCA itself is not.

## Common Pitfalls

### Pitfall 1: Posterior Collapse Masking Useful Dimensions
**What goes wrong:** If the VAE suffered posterior collapse during training, many latent dimensions will be near-identical to the prior (standard normal). PCA will find variance only in the non-collapsed dimensions, potentially yielding very few (1-2) active components.
**Why it happens:** KL annealing not aggressive enough, or dataset too small for the 64-dimensional latent space.
**How to avoid:** This is actually the CORRECT behavior -- PCA should surface only the dimensions where the model learned meaningful structure. The graceful degradation requirement handles this: show whatever dimensions exist, warn the user. The threshold should be conservative (not too aggressive) to avoid showing noise dimensions.
**Warning signs:** PCA finds < 3 components above the variance threshold; explained variance is dominated by a single component (>80%).

### Pitfall 2: Feature Correlation With Insufficient Samples
**What goes wrong:** Pearson correlation computed from too few samples (e.g., 5 sweep points) gives unreliable r-values with high p-values. Labels end up misleading.
**Why it happens:** Each sweep point requires a full decode + mel inversion + feature extraction, which is slow. Temptation to reduce sample count.
**How to avoid:** Use at least 15-20 sweep points per component. The decode path is fast (model inference), but mel-to-waveform via GriffinLim is slower. For correlation, 20 points gives reasonable power for detecting |r| > 0.5. Require p < 0.05 for statistical significance before suggesting a label.
**Warning signs:** Suggested labels change drastically when rerunning analysis; p-values near or above 0.05.

### Pitfall 3: Serialization Version Mismatch
**What goes wrong:** Analysis results saved with one checkpoint format fail to load when the checkpoint format evolves (e.g., adding new fields to AnalysisResult).
**Why it happens:** Using pickle/joblib for sklearn PCA objects ties the saved data to specific sklearn versions. The existing checkpoint system uses torch.save which is essentially pickle.
**How to avoid:** Do NOT save the sklearn PCA object. Instead, extract the numpy arrays (`components_`, `mean_`, `explained_variance_ratio_`) and save those as plain numpy arrays within the checkpoint dict. The PCA object can be reconstructed from these arrays if needed, but the slider mapping only needs `mean + values @ components` which is pure numpy.
**Warning signs:** "Incompatible sklearn version" errors on checkpoint load; corrupted analysis state.

### Pitfall 4: Slider Discretization Causing Non-Reproducibility
**What goes wrong:** Floating-point slider values that should be "the same" differ by tiny amounts, producing different latent vectors.
**Why it happens:** Continuous slider values are snapped to steps inconsistently between UI and backend.
**How to avoid:** Define discrete step indices (integers: 0, 1, 2, ..., n_steps-1) as the ground truth. Convert to PCA-space values via `value = safe_min + step_idx * step_size`. Never store or compare floating-point slider values -- always convert back to step indices.
**Warning signs:** "Same settings" producing audibly different output; seed + slider positions not reproducing identical audio.

### Pitfall 5: PCA on Unnormalized Data Distorting Variance
**What goes wrong:** If encoding plus prior samples have very different scales across latent dimensions (e.g., some dimensions have std=0.1, others std=5.0), PCA will be dominated by high-variance dimensions regardless of musical importance.
**Why it happens:** VAE training may not perfectly regularize all dimensions equally.
**How to avoid:** sklearn's PCA automatically centers data (subtracts mean). For this use case, do NOT whiten the data (whiten=False) because we want PCA to reflect the actual variance structure of the latent space. The variance differences ARE the signal we want to capture -- high-variance dimensions are likely where the model learned meaningful variation.
**Warning signs:** This is actually working as intended when PCA captures high-variance dimensions. Problems only arise if you accidentally whiten or normalize before PCA.

### Pitfall 6: Memory Pressure During Batch Encoding
**What goes wrong:** Encoding the entire training dataset at once (thousands of mel spectrograms) exhausts GPU memory.
**Why it happens:** Loading all mel spectrograms to device simultaneously.
**How to avoid:** Use DataLoader with modest batch_size (32) and process batches sequentially, collecting mu vectors (which are small: [B, 64]) on CPU. The encoding step uses the existing `model.encode()` method which returns mu and logvar; only mu is needed for PCA.
**Warning signs:** OOM errors during analysis; MPS device running out of memory.

## Code Examples

Verified patterns from official sources:

### PCA Fitting with Adaptive Component Count
```python
# Source: scikit-learn 1.8.0 PCA docs
from sklearn.decomposition import PCA
import numpy as np

def fit_pca_adaptive(
    latent_vectors: np.ndarray,   # [n_samples, latent_dim]
    variance_threshold: float = 0.02,  # Minimum variance ratio per component
    min_components: int = 1,
    max_components: int = 20,
) -> tuple[PCA, int]:
    """Fit PCA and determine number of meaningful components.

    A component is "meaningful" if its explained_variance_ratio_
    exceeds variance_threshold.
    """
    # Fit full PCA first to see all variance ratios
    pca = PCA(n_components=min(max_components, latent_vectors.shape[1]))
    pca.fit(latent_vectors)

    # Count components exceeding threshold
    n_active = int(np.sum(pca.explained_variance_ratio_ >= variance_threshold))
    n_active = max(n_active, min_components)
    n_active = min(n_active, max_components)

    # Refit with exact number for clean components_ shape
    if n_active < pca.n_components_:
        pca_final = PCA(n_components=n_active)
        pca_final.fit(latent_vectors)
        return pca_final, n_active

    return pca, n_active
```

### Audio Feature Extraction with numpy/scipy (No librosa)
```python
# Source: Standard DSP formulas + torchaudio SpectralCentroid docs
import numpy as np

def extract_audio_features(
    audio: np.ndarray,      # 1-D float32 waveform
    sample_rate: int = 48_000,
    frame_size: int = 2048,
    hop_length: int = 512,
) -> dict[str, float]:
    """Extract summary audio features from a waveform.

    Returns single scalar per feature (mean across frames).
    Uses numpy/scipy only -- no librosa dependency.
    """
    features = {}

    # RMS Energy: sqrt(mean(x^2)) per frame, then mean across frames
    n_frames = 1 + (len(audio) - frame_size) // hop_length
    rms_values = []
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + frame_size]
        rms_values.append(np.sqrt(np.mean(frame ** 2)))
    features["rms_energy"] = float(np.mean(rms_values))

    # Zero Crossing Rate: count sign changes / frame_size
    zcr_values = []
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + frame_size]
        crossings = np.sum(np.abs(np.diff(np.sign(frame))) > 0)
        zcr_values.append(crossings / frame_size)
    features["zero_crossing_rate"] = float(np.mean(zcr_values))

    # Spectral features from magnitude STFT
    from scipy.signal import stft as scipy_stft
    freqs, times, Zxx = scipy_stft(
        audio, fs=sample_rate, nperseg=frame_size, noverlap=frame_size - hop_length,
    )
    magnitude = np.abs(Zxx)

    # Spectral Centroid: weighted mean of frequencies
    power = magnitude ** 2
    centroid_per_frame = np.sum(freqs[:, None] * power, axis=0) / (np.sum(power, axis=0) + 1e-10)
    features["spectral_centroid"] = float(np.mean(centroid_per_frame))

    # Spectral Rolloff (85th percentile): freq below which 85% of energy lies
    cumulative_energy = np.cumsum(power, axis=0)
    total_energy = cumulative_energy[-1:, :] + 1e-10
    rolloff_mask = cumulative_energy / total_energy >= 0.85
    rolloff_idx = np.argmax(rolloff_mask, axis=0)
    features["spectral_rolloff"] = float(np.mean(freqs[rolloff_idx]))

    # Spectral Flatness: geometric mean / arithmetic mean (in dB)
    log_magnitude = np.log(magnitude + 1e-10)
    geo_mean = np.exp(np.mean(log_magnitude, axis=0))
    arith_mean = np.mean(magnitude, axis=0) + 1e-10
    flatness = geo_mean / arith_mean
    features["spectral_flatness"] = float(np.mean(flatness))

    return features

# Full list of features for correlation
FEATURE_NAMES = [
    "rms_energy",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_rolloff",
    "spectral_flatness",
]
```

### Saving Analysis Results in Checkpoint
```python
# Source: Existing checkpoint.py pattern + numpy serialization
def analysis_to_checkpoint_dict(analysis: "AnalysisResult") -> dict:
    """Convert AnalysisResult to a dict suitable for torch.save.

    Stores only numpy arrays and Python primitives -- NO sklearn objects.
    This ensures checkpoint portability across sklearn versions.
    """
    return {
        "analysis_version": 1,
        "pca_components": analysis.pca_components,          # np.ndarray
        "pca_mean": analysis.pca_mean,                      # np.ndarray
        "explained_variance_ratio": analysis.explained_variance_ratio,
        "n_active_components": analysis.n_active_components,
        "component_labels": analysis.component_labels,
        "suggested_labels": analysis.suggested_labels,
        "user_labels": analysis.user_labels,
        "safe_min": analysis.safe_min,
        "safe_max": analysis.safe_max,
        "warning_min": analysis.warning_min,
        "warning_max": analysis.warning_max,
        "step_size": analysis.step_size,
        "feature_correlations": analysis.feature_correlations,
    }

# In checkpoint.py save_checkpoint, add:
# checkpoint["latent_analysis"] = analysis_to_checkpoint_dict(analysis)
#
# In checkpoint.py load_checkpoint, read:
# analysis_data = checkpoint.get("latent_analysis", None)
# if analysis_data is not None:
#     analysis = AnalysisResult.from_dict(analysis_data)
```

### Generating Audio from Slider Positions
```python
# Source: Project pattern from inference/generation.py + PCA math
import torch
import numpy as np

def generate_from_sliders(
    slider_step_indices: list[int],   # Integer step indices per active component
    analysis: "AnalysisResult",
    model: "ConvVAE",
    spectrogram: "AudioSpectrogram",
    device: "torch.device",
    seed: int | None = None,
) -> np.ndarray:
    """Generate audio from discrete slider positions.

    Steps:
    1. Convert step indices to PCA-space values
    2. Reconstruct 64-D latent vector from PCA components
    3. Decode latent vector to mel spectrogram
    4. Convert mel to waveform
    """
    # Step indices -> PCA-space values
    n = analysis.n_active_components
    pca_values = np.zeros(n, dtype=np.float32)
    for i in range(n):
        pca_values[i] = analysis.safe_min[i] + slider_step_indices[i] * analysis.step_size[i]

    # PCA-space -> original latent space
    z = analysis.pca_mean + pca_values @ analysis.pca_components[:n]

    # Set seed for any stochastic parts of generation
    if seed is not None:
        torch.manual_seed(seed)

    # Decode
    z_tensor = torch.from_numpy(z).float().unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        mel_shape = spectrogram.get_mel_shape(48_000)  # 1 second at 48kHz
        # Ensure decoder initialised
        if model.decoder.fc is None:
            n_mels, time_frames = mel_shape
            pad_h = (16 - n_mels % 16) % 16
            pad_w = (16 - time_frames % 16) % 16
            spatial = ((n_mels + pad_h) // 16, (time_frames + pad_w) // 16)
            model.decoder._init_linear(spatial)
        mel = model.decode(z_tensor, target_shape=mel_shape)

    wav = spectrogram.mel_to_waveform(mel.cpu())
    return wav.squeeze().numpy().astype(np.float32)
```

## Discretion Recommendations

These are areas marked as Claude's Discretion in CONTEXT.md:

### Slider Visual Feedback: Label + Value + Variance Percentage
**Recommendation:** Show each slider with: (1) the label (user-overridden or suggested), (2) the current step value as a numeric readout, and (3) a small "explains X% variance" indicator below each slider. This gives users both the intuitive control and the statistical context to understand which sliders matter most.
**Rationale:** Variance-explained percentages directly answer "which sliders have the most impact?" without requiring audio experimentation. A slider explaining 40% of variance is clearly more impactful than one explaining 2%. This also helps users understand graceful degradation -- if only 2 sliders exist, seeing "explains 45% + 35% = 80% total" communicates that most variation is captured.

### Expose Variance-Explained Percentages: Yes
**Recommendation:** Show variance percentages per slider. Display as small text under each slider label (e.g., "Axis 1 (spectral centroid) -- 34.2% variance"). Also show cumulative variance explained across all active sliders.
**Rationale:** This is the key piece of information that distinguishes a well-trained model (15 meaningful axes) from a poorly-trained one (2 axes capturing 90%). It also helps users prioritize exploration -- start with high-variance sliders for dramatic changes, low-variance sliders for subtle tweaks.

### Variance Threshold: 2% (0.02) of Total Variance
**Recommendation:** A PCA component is "meaningful" if its `explained_variance_ratio_` exceeds 0.02 (2% of total variance). This is deliberately conservative -- it will include components that explain even modest variation, supporting the user's goal of maximum exploration.
**Rationale:**
- Too high (e.g., 10%): Would only expose 2-4 sliders for most models, limiting exploration.
- Too low (e.g., 0.5%): Would expose noise dimensions that produce no audible change, confusing users.
- 2% balances coverage vs noise. For a 64-dim latent space where variance is evenly distributed, each dimension would explain ~1.6%. Setting the threshold at 2% filters out dimensions that are at or below the "uniform distribution" baseline.
- This threshold should be a configurable constant, not hard-coded, so it can be tuned after testing with real trained models.

### Compression of Analysis Data: None Required
**Recommendation:** Store analysis data as raw numpy arrays within the checkpoint dict. Do not compress.
**Rationale:** The analysis data is tiny. For n_active_components=12 and latent_dim=64: `pca_components` is 12x64 = 768 float64 values (6 KB), `pca_mean` is 64 values (0.5 KB), ranges are 12 values each (~0.1 KB). Total analysis overhead: well under 100 KB. The model checkpoint itself (3.1M parameters) is already tens of megabytes. The analysis data is negligible.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual latent dimension labeling | PCA-based automated discovery | MIDISpace (2022), RAVE research (2021-2023) | Removes human labeling bottleneck; labels based on data, not assumptions |
| Fixed number of control dimensions | Adaptive component count via variance threshold | Standard ML practice | Works with any quality of trained model; graceful degradation |
| Saving full sklearn PCA objects | Saving component arrays as numpy | Current best practice | Version-portable checkpoints; no sklearn version lock-in |
| Librosa for audio features | Numpy/scipy/torchaudio direct computation | Trend toward minimal dependencies | Avoids numba dependency; features are simple formulas |

**Deprecated/outdated:**
- Saving sklearn objects via pickle for production persistence: Version-sensitive; use numpy arrays instead
- Using whiten=True for PCA on VAE latent spaces: Whitening destroys the variance structure that tells us which dimensions matter

## Open Questions

1. **Optimal number of sweep points for correlation**
   - What we know: 20 sweep points per component is statistically reasonable for detecting |r| > 0.5 with p < 0.05.
   - What's unclear: For 12 active components x 20 points each = 240 model inferences + GriffinLim inversions. This could take 30-60 seconds. Acceptable for a user-triggered operation, but needs benchmarking.
   - Recommendation: Start with 20 points. If too slow, reduce to 15. The operation is user-triggered and can show a progress indicator. Parallelize across components if possible (each component sweep is independent).

2. **Feature list completeness**
   - What we know: Spectral centroid, RMS energy, zero-crossing rate, spectral rolloff, and spectral flatness cover the major perceptual dimensions (brightness, loudness, noisiness, frequency distribution, tonality).
   - What's unclear: Whether these 5 features are sufficient to produce meaningful labels for all PCA components, or whether additional features (e.g., spectral bandwidth, onset strength, temporal autocorrelation) would improve label quality.
   - Recommendation: Start with the 5 features listed. They cover the parameter categories in the requirements (timbre via centroid/flatness, temporal via ZCR, spatial/textural via rolloff/flatness, density via RMS). Add more features in future iterations based on testing.

3. **Interaction with GenerationPipeline for multi-chunk generation**
   - What we know: The current `GenerationPipeline.generate()` method uses random latent vectors for chunk generation. Slider-controlled generation needs to use the user-specified latent vector.
   - What's unclear: For multi-second generation (crossfade/interpolation), should all chunks use the same slider-derived latent vector, or should there be controlled variation?
   - Recommendation: For v1, use the slider-derived latent vector as the single generation point for 1-second output. For multi-chunk generation, the slider-derived vector becomes the anchor, with chunk-to-chunk variation controlled by the existing seed parameter. The seed introduces controlled randomness around the slider position. This is a Phase 5 + Phase 4 integration question that should be resolved during implementation.

4. **User label persistence across re-analysis**
   - What we know: Users can rename slider labels. Analysis results are saved in checkpoints.
   - What's unclear: If the user re-runs "Analyze latent space" (e.g., after more training), the PCA components may differ. User labels from the previous analysis may no longer apply.
   - Recommendation: When re-analyzing, warn the user that custom labels will be reset. Store previous user labels in the checkpoint for reference, but do not automatically apply them to new components.

## Sources

### Primary (HIGH confidence)
- [scikit-learn PCA 1.8.0 docs](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) -- constructor params, attributes (components_, explained_variance_ratio_, mean_), methods (fit, transform, inverse_transform)
- [torchaudio 2.10.0 SpectralCentroid](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.SpectralCentroid.html) -- confirmed available in torchaudio 2.10 with sample_rate, n_fft, hop_length params
- [torchaudio 2.10.0 transforms list](https://docs.pytorch.org/audio/stable/transforms.html) -- confirmed SpectralCentroid exists; RMS, ZCR, spectral_rolloff, spectral_flatness do NOT exist in torchaudio
- [torchaudio 2.10.0 functional list](https://docs.pytorch.org/audio/stable/functional.html) -- confirmed spectral_centroid function exists; no RMS or ZCR functions
- [scipy.stats.pearsonr docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html) -- Pearson r with p-value for correlation significance
- Existing codebase: `models/vae.py` (64-dim latent, encode method returns mu/logvar), `training/checkpoint.py` (torch.save dict format), `inference/generation.py` (GenerationPipeline), `audio/spectrogram.py` (mel-to-waveform)

### Secondary (MEDIUM confidence)
- [Rolinek et al., "Variational Autoencoders Pursue PCA Directions (by Accident)", CVPR 2019](https://ar5iv.labs.arxiv.org/html/1812.06775) -- VAEs naturally align with PCA directions; PCA on VAE latent space recovers explicit variance ordering. Verified via ArXiv full text.
- [Valero-Mas et al., "MIDISpace: Finding Linear Directions in Latent Space for Music Generation", 2022](https://dl.acm.org/doi/fullHtml/10.1145/3527927.3532790) -- PCA on music VAE latent space finds "largely disentangled directions" that are "monotonic, global and encode fundamental musical characteristics"
- [librosa 0.11.0 dependency list](https://deepwiki.com/librosa/librosa/1.1-installation) -- confirmed librosa requires numba as mandatory dependency; justifies using numpy/scipy instead
- [scikit-learn 1.8.0 release info](https://scikit-learn.org/stable/whats_new.html) -- confirmed 1.8.0 is latest stable release (December 2025)

### Tertiary (LOW confidence)
- Spectral centroid formula from musicinformationretrieval.com and Wikipedia -- standard DSP formulas, but implementation details (windowing, overlap) need validation against test audio
- PCA variance threshold (2%) recommendation -- based on statistical reasoning about 64-dim spaces, not empirical testing with this specific VAE architecture. Needs tuning after first trained model.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- scikit-learn PCA is the standard tool; all other libs already in project
- Architecture: HIGH -- PCA-on-latent-space is a well-researched pattern validated by multiple papers; code patterns follow existing project conventions
- Audio features: MEDIUM -- formulas are standard DSP but our numpy/scipy implementations need testing against librosa's reference implementations for correctness
- Correlation approach: MEDIUM -- Pearson correlation for PCA-feature labeling is logical but effectiveness depends on how disentangled this specific VAE's latent space is
- Slider discretization: HIGH -- integer step indices with deterministic conversion is a proven pattern for reproducibility
- Pitfalls: HIGH -- identified from both research literature and practical experience with the codebase

**Research date:** 2026-02-13
**Valid until:** 2026-03-15 (stable domain; PCA and audio feature extraction are mature techniques)
