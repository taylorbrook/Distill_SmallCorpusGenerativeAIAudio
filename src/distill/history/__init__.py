"""Generation history tracking and A/B comparison.

Public API:
- HistoryEntry: dataclass for a single generation history entry
- GenerationHistory: CRUD for generation history with WAV + thumbnail storage
- ABComparison: runtime state for A/B comparison between two generations
"""

from distill.history.comparison import ABComparison
from distill.history.store import GenerationHistory, HistoryEntry

__all__ = ["ABComparison", "GenerationHistory", "HistoryEntry"]
