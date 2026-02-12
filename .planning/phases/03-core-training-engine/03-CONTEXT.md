# Phase 3: Core Training Engine - Context

**Gathered:** 2026-02-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Train a generative VAE model on small datasets (5-500 audio files) with overfitting prevention, progress monitoring, and checkpoint recovery. Users can monitor training, hear previews, cancel/resume, and the system prevents overfitting automatically. Generation quality, export, and parameter controls are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Training progress feedback
- Live dashboard (not scrolling logs) showing real-time metrics
- Detailed metrics: loss curves (train + val), KL divergence, reconstruction loss, learning rate, epoch, step, ETA, GPU/device memory usage
- Hybrid update frequency: step-level updates for training loss, epoch-level updates for validation metrics
- Claude's discretion on whether to include an intuitive quality indicator vs raw metrics only

### Audio preview behavior
- Generate 1 audio preview every N epochs (configurable interval)
- Manual playback only — preview appears with play button, no auto-play
- Keep all previews visible in a scrollable list with epoch labels — user can hear the model improve over time

### Overfitting controls
- Layered control system: fully automatic defaults, 2-3 presets (Conservative/Balanced/Aggressive), and advanced toggles (dropout, weight decay, augmentation strength) for power users
- On overfitting detection (val loss diverging from train loss): warn visually on dashboard but continue training — user decides when to stop
- Auto-adapt regularization strategy based on dataset size — smaller datasets (5-50 files) get stronger regularization, more augmentation, fewer default epochs
- Automatic validation split based on dataset size (e.g., 80/20 for larger sets, adaptive for tiny sets)

### Checkpoint & resume UX
- Automatic checkpoint saving at regular intervals
- Claude's discretion on: checkpoint retention count, cancel behavior (immediate save vs finish-epoch), resume flow (summary vs auto-continue), and whether to include a manual "save now" button

### Claude's Discretion
- Quality indicator approach (intuitive score vs raw metrics only)
- Checkpoint retention count (balancing disk space vs rollback flexibility)
- Cancel behavior (immediate checkpoint save vs finish current epoch)
- Resume flow (summary screen vs seamless auto-continue)
- Manual checkpoint save button (include or omit)
- Exact preview interval default (every N epochs)
- Specific preset parameter values for overfitting presets
- Validation split ratios for different dataset sizes

</decisions>

<specifics>
## Specific Ideas

- Dashboard should feel like a proper ML training monitor — loss curves, not just numbers
- Preview list creates a timeline of model quality evolution — important for small-dataset training where you want to catch the "sweet spot" before overfitting
- Layered overfitting controls let beginners just train while power users can fine-tune regularization

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-core-training-engine*
*Context gathered: 2026-02-12*
