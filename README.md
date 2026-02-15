# Distill

*Small Custom-Corpus Generative AI Audio*

A generative audio tool that trains VAE models on your own small audio collections (5-500 files) and lets you explore the learned sound space through musically meaningful parameter controls.

Train on your recordings, twist the sliders, discover sounds that are unmistakably *yours* but entirely new.

## Why This Exists

Existing tools either give you opaque latent dimensions that don't map to musical concepts (RAVE), or lean on massive datasets of popular styles that strip away your personal voice (Suno, etc.). Distill sits in between: **controllable exploration of your own sound with musically meaningful parameters**.

The core innovation is PCA-based latent space analysis that maps the VAE's internal dimensions to parameters you actually think in — brightness, warmth, roughness, harmonic tension, temporal character, spatial texture.

## Features

- **Train on tiny datasets** — meaningful results from as few as 5-20 audio files
- **Musically meaningful sliders** — PCA-mapped controls (brightness, warmth, roughness, harmonic tension, etc.) instead of opaque latent dimensions
- **Multi-format export** — WAV (48kHz/24bit default), MP3, FLAC, OGG with embedded metadata
- **Spatial audio** — mono, stereo field, and binaural (HRTF) output
- **Multi-model blending** — load multiple trained models and blend between them
- **Presets** — save and recall slider configurations per model
- **Generation history** — browse past outputs with A/B comparison
- **Model library** — save, load, search, and manage trained models
- **Web UI** — Gradio-based interface with tabs for data, training, generation, and library
- **CLI** — full command-line access for batch generation and scripting
- **Hardware flexible** — Apple Silicon (MPS), NVIDIA (CUDA), and CPU fallback

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)

## Setup

```bash
git clone <repo-url>
cd "Small DataSet Audio"
make setup
```

This installs all dependencies (PyTorch, TorchAudio, Gradio, etc.) into a local `.venv` via uv.

## Usage

### Web UI

```bash
distill
# or
make run
```

Launches a Gradio web interface with four tabs:

| Tab | What it does |
|-----|-------------|
| **Data** | Upload and manage audio datasets |
| **Train** | Train VAE models with live progress and loss charts |
| **Generate** | Explore the latent space with sliders, preview and export audio |
| **Library** | Browse, search, and manage saved models |

### CLI

#### Train a model

```bash
distill train ./my-recordings/

# With options
distill train ./my-recordings/ --preset balanced --epochs 200 --model-name "field-recordings-v1"
```

Training presets: `auto` (adapts to dataset size), `conservative`, `balanced`, `aggressive`.

#### Generate audio

```bash
# Basic generation
distill generate my-model --duration 5.0

# Batch generation with slider control
distill generate my-model -n 10 -d 3.0 --slider '0:7' --slider '1:-3'

# Export as FLAC with stereo spatial processing
distill generate my-model -d 10.0 --format flac --spatial-mode stereo

# Use a saved preset
distill generate my-model -d 5.0 --preset "bright-textures"

# Blend two models
distill generate model-a --blend 'model-b:40' -d 5.0

# Machine-readable output (pipe to other tools)
distill generate my-model -n 5 --json
```

Slider values range from -10 to 10. The number of active sliders depends on the model (typically 4-8, determined by PCA analysis).

#### Manage models

```bash
distill model list              # table of all saved models
distill model info my-model     # detailed model info
distill model delete my-model   # remove a model (with confirmation)
distill model list --json       # JSON output for scripting
```

Models can be referenced by name, ID, or `.distill` file path.

### Global Options

```bash
distill --device cpu        # force CPU (default: auto-detect)
distill --device mps        # force Apple Silicon
distill --device cuda       # force NVIDIA GPU
distill --verbose           # detailed startup info
distill --config path.toml  # custom config file
```

## Project Structure

```
src/distill/
├── audio/          # I/O, preprocessing, spectrograms, filters, spatial, metadata
├── cli/            # Typer CLI (train, generate, model, ui subcommands)
├── config/         # Settings and defaults
├── controls/       # PCA-based latent space analysis and slider mapping
├── data/           # Dataset loading and summary
├── hardware/       # Device detection, memory management, benchmarking
├── history/        # Generation history store and A/B comparison
├── inference/      # Generation pipeline, chunking, blending, spatial, export
├── library/        # Model catalog (JSON index)
├── models/         # VAE architecture, losses, persistence
├── presets/        # Preset save/load manager
├── training/       # Training loop, config, metrics, checkpointing, previews
├── ui/             # Gradio web interface (tabs, components, state)
└── validation/     # Environment and startup validation
```

## Configuration

On first run, a `config.toml` is created:

```toml
[general]
project_name = "Distill"

[paths]
datasets = "data/datasets"
models = "data/models"
generated = "data/generated"

[hardware]
device = "mps"          # auto, mps, cuda, cpu
max_batch_size = 32
memory_limit_gb = 64.0
```

## Development

```bash
make test       # run test suite
make lint       # ruff linter
make format     # auto-format with ruff
make benchmark  # hardware benchmark
make clean      # remove caches and build artifacts
```

## How It Works

1. **Audio in** — Your audio files are converted to mel spectrograms
2. **Training** — A convolutional VAE learns a compressed latent representation of your sound
3. **Analysis** — PCA extracts the most meaningful axes of variation in the latent space and maps them to musically interpretable parameters
4. **Generation** — Navigate the latent space via sliders, generating new spectrograms that are converted back to audio using chunk-based synthesis with crossfade for arbitrary duration
5. **Export** — Output at professional quality (48kHz/24bit+) in your preferred format with metadata

## License

MIT
