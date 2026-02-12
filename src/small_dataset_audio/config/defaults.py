"""Default configuration values for Small Dataset Audio.

These defaults are used when no config.toml exists, and as fallback
values when loading a config file that is missing newer keys (forward
compatibility).

Note: TOML has no null type, so we use 0 / 0.0 to mean "not yet set"
for numeric fields that get populated by the hardware benchmark.
"""

DEFAULT_CONFIG: dict = {
    "general": {
        "project_name": "Small Dataset Audio",
        "first_run_complete": False,
    },
    "paths": {
        "datasets": "data/datasets",
        "models": "data/models",
        "generated": "data/generated",
    },
    "hardware": {
        "device": "auto",          # "auto", "mps", "cuda", "cpu"
        "max_batch_size": 0,       # 0 = not yet benchmarked, set by benchmark
        "memory_limit_gb": 0.0,    # 0.0 = not yet measured
    },
}
