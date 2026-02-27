# Phase 15: ISTFT Reconstruction - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Convert generated 2-channel spectrograms (magnitude + instantaneous frequency) into audio waveforms via ISTFT using phase reconstructed from IF. Remove all Griffin-Lim code and v1.0-only reconstruction paths. This phase delivers the reconstruction function and cleanup — full pipeline wiring is Phase 16.

</domain>

<decisions>
## Implementation Decisions

### Griffin-Lim removal
- Delete all Griffin-Lim code entirely — no debug flag, no fallback, no stub
- Full cleanup: remove from code, configs, comments, docstrings — no trace left
- Natural errors only: if anything references deleted Griffin-Lim code, let normal Python ImportError/AttributeError surface it
- Delete existing Griffin-Lim tests; write fresh ISTFT reconstruction tests from scratch

### v1.0 model compatibility
- Clean break: v2.0 pipeline only, no backward compatibility with v1.0 magnitude-only models
- No special handling for v1.0 model loading — let it fail naturally on channel dimension mismatch
- Clean up v1.0-only reconstruction code paths as part of this phase (one clean sweep, not deferred to Phase 16)
- All saved v1.0 checkpoints are considered disposable — no preservation needed

### Phase reconstruction behavior
- Initial phase value: zero at time step 0 for cumulative sum
- Leave phase unwrapped after cumulative sum (no wrapping to [-pi, pi])
- Mel-to-linear frequency conversion before applying ISTFT
- Denormalization placement: Claude's discretion — put it where it fits cleanest in the pipeline

### Quality validation
- Both round-trip tests and waveform sanity checks
- Round-trip: encode real audio → spectrogram → reconstruct via ISTFT → compare to original; Claude determines appropriate quality thresholds for mel-domain round-trip
- Sanity checks: valid audio (correct length, no NaN/inf, reasonable amplitude)
- Test signals: synthetic (sine waves, noise) for fast unit tests + real audio files for integration/quality tests
- No meta-test for Griffin-Lim removal verification — trust the cleanup

### Claude's Discretion
- Whether denormalization happens inside the reconstruction function or is the caller's responsibility
- Appropriate SNR/MSE thresholds for round-trip quality tests (realistic for mel-domain reconstruction)
- Exact mel-to-linear inversion approach

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches for ISTFT reconstruction. The key constraint is that the pipeline must be clean: no Griffin-Lim anywhere, no v1.0 compatibility shims, and fresh tests for the new path.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 15-istft-reconstruction*
*Context gathered: 2026-02-27*
