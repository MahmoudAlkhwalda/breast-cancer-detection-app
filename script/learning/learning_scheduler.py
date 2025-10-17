"""
Learning Scheduler (minimal no-op implementation)

Provides the interfaces expected by app.py without introducing
any new runtime behavior. This avoids import errors while
preserving existing application functionality.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


_SCHEDULER: Optional["LearningScheduler"] = None


class LearningScheduler:
    def __init__(self, db_session: Any = None) -> None:
        self.db_session = db_session
        self._running: bool = False
        self._learning_enabled: bool = True

    def start_scheduler(self) -> None:
        self._running = True

    def stop_scheduler(self) -> None:
        self._running = False

    def enable_learning(self) -> None:
        self._learning_enabled = True

    def disable_learning(self) -> None:
        self._learning_enabled = False

    def get_scheduler_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "learning_enabled": self._learning_enabled,
        }


def initialize_learning_scheduler(db_session: Any = None) -> LearningScheduler:
    """Create the global scheduler instance if not already created."""
    global _SCHEDULER
    if _SCHEDULER is None:
        _SCHEDULER = LearningScheduler(db_session)
    return _SCHEDULER


def get_learning_scheduler() -> Optional[LearningScheduler]:
    """Return the global scheduler instance (may be None if not initialized)."""
    return _SCHEDULER
