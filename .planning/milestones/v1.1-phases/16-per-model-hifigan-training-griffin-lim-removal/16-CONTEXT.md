# Phase 16: Per-Model HiFi-GAN Training & Griffin-Lim Removal - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Users can train a small per-model HiFi-GAN V2 vocoder on their specific training audio for maximum fidelity. The system auto-selects the best available vocoder (per-model > BigVGAN universal). Griffin-Lim reconstruction code is fully removed — neural vocoder is the only reconstruction method. Training supports cancel/resume and works via both the Gradio UI Train tab and the `train-vocoder` CLI command.

</domain>

<decisions>
## Implementation Decisions

### Training UX in UI
- Vocoder training controls live in a **section below the existing VAE training controls** in the Train tab
- Section only enabled/visible after VAE training is complete for the selected model
- **Full parameter control** exposed: epochs, learning rate, batch size, checkpoint interval
- Training progress shows **live-updating loss curve chart** (generator + discriminator loss) plus text stats (epoch, current loss, ETA)
- **Periodic audio preview samples** generated during training so the user can hear improvement over time (every N epochs — Claude decides frequency)

### CLI train-vocoder command
- Command structure at Claude's discretion (subcommand of `train` or top-level — pick based on existing CLI patterns)
- **Mirror UI parameters exactly**: --epochs, --lr, --batch-size, --checkpoint-interval
- Training output uses **Rich live table** display showing epoch, loss values, ETA, and progress bar (consistent with existing Rich console usage)
- CLI audio preview samples: Claude's discretion on whether to save periodic WAV previews to disk during CLI training

### Cancel & resume behavior
- On cancel: **immediately save checkpoint** at current epoch, confirm with "Checkpoint saved at epoch N. Resume anytime." — graceful stop, no data loss
- On restart with existing checkpoint: **ask the user** "Resume from epoch N or start fresh?" — explicit choice every time
- Checkpoints stored **inside the .distillgan model file** itself — self-contained, no external checkpoint files to manage
- **Retrain with confirmation**: if model already has a trained vocoder, warn "This model already has a trained vocoder. Replace it?" before proceeding

### Auto-selection policy
- Auto-selection logic (whether to always prefer per-model or use a quality threshold) at **Claude's discretion** — pick a sensible policy based on how HiFi-GAN training converges
- When user manually selects per-model HiFi-GAN with minimal training: show a **subtle warning** like "Trained for 5 epochs — quality may be limited" but allow the selection
- After Griffin-Lim removal, if no vocoder is available: **auto-download BigVGAN** transparently on first generate (consistent with Phase 15 lazy download design)
- Quality badge after generation unchanged from Phase 15 decisions — shows vocoder name alongside seed/sample-rate/bit-depth

### Griffin-Lim removal
- Fully remove all Griffin-Lim reconstruction code — no fallback, no legacy path
- Neural vocoder (BigVGAN universal or per-model HiFi-GAN) is the only mel-to-waveform method

### Claude's Discretion
- CLI command structure (subcommand vs top-level)
- Audio preview frequency during training (every N epochs)
- Whether CLI saves periodic audio preview WAVs to disk
- Auto-selection quality threshold policy
- Training defaults for each parameter (sensible defaults for small datasets)
- Data augmentation specifics (TRAIN-03)
- Adversarial loss implementation details (TRAIN-02)

</decisions>

<specifics>
## Specific Ideas

- Loss curve should show both generator and discriminator loss — user can see the adversarial training dynamic
- Audio preview lets user hear the vocoder improving — critical for knowing "when to stop" since loss numbers alone aren't intuitive for audio quality
- Checkpoint-inside-model design means users never deal with loose checkpoint files — just their .distillgan file
- Resume choice is important because sometimes the user wants to start fresh with different hyperparameters

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 16-per-model-hifigan-training-griffin-lim-removal*
*Context gathered: 2026-02-27*
