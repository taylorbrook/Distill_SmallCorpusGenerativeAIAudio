# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-21)

**Core value:** Controllable exploration — users can reliably navigate between sound worlds using discrete audio codes and generative priors
**Current focus:** v1.1 VQ-VAE — replacing continuous VAE with RVQ-VAE architecture

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-02-21 — Milestone v1.1 started

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- v1.1: Replace continuous VAE with RVQ-VAE (discrete codes for sharper reconstructions)
- v1.1: Include autoregressive prior model (full two-stage generation pipeline)
- v1.1: Rethink UI around discrete codes (encode + manipulate + decode, not PCA sliders)
- v1.1: Clean break from v1.0 (old models won't load, no backward compatibility)
- v1.1: Use lucidrains/vector-quantize-pytorch for VQ/RVQ layers

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-21
Stopped at: Defining milestone v1.1 requirements
Resume file: None
