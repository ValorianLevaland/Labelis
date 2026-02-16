from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional


@dataclass(frozen=True)
class DebugEvent:
    """One debug event (stage-tagged, timestamped)."""

    t_iso: str
    stage: int
    message: str


class DebugTrace:
    """Collects Debug(n) messages and can persist them.

    This is the Python equivalent of the MATLAB script's staged DEBUG prints.
    """

    def __init__(self, log_cb: Optional[Callable[[str], None]] = None):
        self._log_cb = log_cb
        self.events: List[DebugEvent] = []

    def log(self, stage: int, message: str) -> None:
        stage_i = int(stage)
        msg = str(message)
        line = f"DEBUG({stage_i}): {msg}"
        if self._log_cb is not None:
            self._log_cb(line)
        self.events.append(DebugEvent(t_iso=datetime.now().isoformat(timespec="seconds"), stage=stage_i, message=msg))

    def save_text(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{e.t_iso} | DEBUG({e.stage}): {e.message}" for e in self.events]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def as_lines(self) -> List[str]:
        return [f"{e.t_iso} | DEBUG({e.stage}): {e.message}" for e in self.events]
