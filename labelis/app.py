from __future__ import annotations

import sys
import traceback
import logging
from datetime import datetime
from pathlib import Path

from .utils import ensure_log_dir


def _setup_logging() -> tuple[logging.Logger, Path]:
    log_dir = ensure_log_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"labelis_{ts}.log"

    logger = logging.getLogger("labelis")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if main() called twice
    if not any(isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == log_path for h in logger.handlers):
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger, log_path


def _qt_critical(title: str, message: str) -> None:
    # Safe message box if Qt is available; otherwise stderr.
    try:
        from PyQt5 import QtWidgets
        QtWidgets.QMessageBox.critical(None, title, message)
    except Exception:
        print(f"[{title}] {message}", file=sys.stderr)


def main() -> int:
    logger, log_path = _setup_logging()
    logger.info("[Labelis] Starting...")
    logger.info(f"[Labelis] Log file: {log_path}")

    try:
        from PyQt5 import QtWidgets  # noqa: F401
    except Exception as e:
        msg = "PyQt5 is not available in this environment.\n\n" + str(e)
        logger.exception(msg)
        _qt_critical("Labelis", msg)
        return 1

    try:
        import napari  # noqa: F401
    except Exception as e:
        msg = "Napari is not available in this environment.\n\n" + str(e)
        logger.exception(msg)
        _qt_critical("Labelis", msg)
        return 1

    try:
        from PyQt5 import QtWidgets
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        # Keep running as long as the Napari window exists
        app.setQuitOnLastWindowClosed(True)

        from .gui.controller import LabelisController

        ctrl = LabelisController(logger=logger)
        ctrl.show()

        ret = app.exec_()
        logger.info(f"[Labelis] Exiting with code {ret}")
        return int(ret)

    except Exception:
        tb = traceback.format_exc()
        logger.error("[Labelis] Fatal error:\n" + tb)
        _qt_critical("Labelis - Fatal error", f"A fatal error occurred.\n\nLog: {log_path}\n\n{tb}")
        return 1
