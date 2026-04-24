"""
progress_tracker.py — Thread-safe global state for pipeline progress.
Exposes a singleton ProgressTracker used by all processing modules.
"""

from __future__ import annotations

import threading
from typing import Optional

from backend.models.schema import ProcessingStatus
from backend.utils.logger import get_logger

logger = get_logger("progress_tracker")

_MAX_LOG_LINES = 500   # keep the last N log messages in memory


class ProgressTracker:
    """
    Thread-safe progress tracker.
    All mutations go through lock-guarded methods so that the FastAPI
    background task and the /status polling endpoint never race.
    """

    def __init__(self) -> None:
        self._lock  = threading.Lock()
        self._state = ProcessingStatus()
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def start(self, total_files: int) -> None:
        with self._lock:
            self._state = ProcessingStatus(
                total_files=total_files,
                is_running=True,
                is_complete=False,
            )
        logger.info("Pipeline started — %d files queued.", total_files)

    def set_current_file(self, name: str) -> None:
        with self._lock:
            self._state.current_file = name
        logger.info("Processing: %s", name)

    def file_done(self) -> None:
        with self._lock:
            self._state.processed_files += 1
            self._state.current_file = None

    def add_rows(self, count: int) -> None:
        with self._lock:
            self._state.extracted_rows += count

    def add_error(self, message: str) -> None:
        with self._lock:
            self._state.errors += 1
        self.log(f"❌ ERROR: {message}")
        logger.error(message)

    def set_excel_url(self, url: str) -> None:
        with self._lock:
            self._state.excel_url = url

    def add_records(self, records: list[QuestionPaperRecord]) -> None:
        """Add new records to the live preview list (keep last 10)."""
        with self._lock:
            # Convert to dict for JSON serialization
            rows = [r.to_excel_row() for r in records]
            self._state.recent_records = (rows + self._state.recent_records)[:10]

    def log(self, message: str) -> None:
        with self._lock:
            self._state.logs.append(message)
            if len(self._state.logs) > _MAX_LOG_LINES:
                self._state.logs = self._state.logs[-_MAX_LOG_LINES:]

    def complete(self) -> None:
        with self._lock:
            self._state.is_running  = False
            self._state.is_complete = True
            self._state.current_file = None
        logger.info(
            "Pipeline complete — %d rows extracted, %d errors.",
            self._state.extracted_rows,
            self._state.errors,
        )

    def reset(self) -> None:
        """Reset to initial state (called before each new run)."""
        with self._lock:
            self._state = ProcessingStatus()
            self._stop_event.clear()

    def stop(self) -> None:
        """Signals the background thread to stop."""
        self._stop_event.set()
        self.log("🛑 Stop signal received. Finishing current file...")

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def snapshot(self) -> ProcessingStatus:
        """Return a deep-copy snapshot safe to serialize."""
        with self._lock:
            return self._state.model_copy(deep=True)

    def is_running(self) -> bool:
        with self._lock:
            return self._state.is_running


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

tracker = ProgressTracker()
