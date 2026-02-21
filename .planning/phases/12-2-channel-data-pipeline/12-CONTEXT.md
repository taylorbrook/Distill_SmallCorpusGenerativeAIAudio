# Phase 12: 2-Channel Data Pipeline - Context

**Gathered:** 2026-02-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Transform the audio loading pipeline from 1-channel magnitude-only mel spectrograms to 2-channel (magnitude + instantaneous frequency) spectrograms. Compute IF in mel domain, normalize both channels independently, mask IF in low-amplitude bins, and cache preprocessed spectrograms to disk. Mel-scale pipeline and STFT parameters preserved from v1.0. VAE architecture changes and ISTFT reconstruction are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Training workflow
- Preprocessing happens automatically on first `distill train` if no cache exists — same UX as v1.0
- No separate `distill preprocess` command needed (training auto-detects and builds cache)
- Show per-file progress output during preprocessing (file count, progress bar)
- Auto-detect when audio files change (added/removed/replaced) by comparing file list and modification times against cache; rebuild only what changed

### Augmentation strategy
- Augmentation continues on waveforms (pitch shift, speed, noise, volume) — then the augmented waveforms are converted to 2-channel spectrograms and cached
- All augmented variants are pre-baked into the cache (10x expansion cached to disk)
- Show estimated disk usage before preprocessing begins; let user confirm if large
- Cache stores the augmentation config used; if settings change, cache is invalidated automatically

### Configuration exposure
- IF masking threshold: user-configurable in training config (power users can tune it)
- STFT parameters (n_fft, hop_length, n_mels): keep v1.0 defaults (2048, 512, 128) but make configurable for advanced users
- Config organization: Claude's discretion — organize based on existing code patterns

### Cache behavior
- Cache location: `.cache/` folder inside the audio dataset directory (cache lives with the data)
- Include a human-readable JSON manifest in the cache dir showing: file list, config/settings used, normalization statistics, timestamps
- Cache file format: Claude's discretion (likely .pt tensors given v1.0 precedent)
- Incremental vs full rebuild on file changes: Claude's discretion — balance simplicity with correctness
- `--no-cache` flag: Claude's discretion — decide based on whether dual code paths add meaningful complexity

### Claude's Discretion
- Normalization scope: per-dataset vs per-file (pick what works best for VAE training on small datasets)
- Config organization structure (extend TrainingConfig vs nested section)
- Cache file format (.pt vs .npz)
- Incremental rebuild strategy
- Whether to include `--no-cache` escape hatch

</decisions>

<specifics>
## Specific Ideas

- Training workflow should feel identical to v1.0 from the user's perspective — `distill train` just works, preprocessing is transparent
- Progress output during preprocessing so user knows what's happening (not a silent wait)
- Cache manifest should be inspectable for debugging — users can look at it to understand what was cached and with what settings
- Disk usage estimate before large cache builds gives user control over storage

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 12-2-channel-data-pipeline*
*Context gathered: 2026-02-21*
