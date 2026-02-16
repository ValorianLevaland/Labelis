from __future__ import annotations

import os
import platform
from pathlib import Path


def user_data_dir(app_name: str = "labelis") -> Path:
    """Return a per-user data directory.

    We avoid external dependencies (e.g. *appdirs*) to keep the distribution
    lightweight.

    Best-effort platform mapping:
    - Windows: %LOCALAPPDATA%/<app_name> (fallback to %APPDATA%)
    - macOS:   ~/Library/Application Support/<app_name>
    - Linux/Unix: $XDG_DATA_HOME/<app_name> (fallback ~/.local/share/<app_name>)
    """
    system = platform.system().lower()

    if system.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / app_name
        # Conservative fallback
        return Path.home() / "AppData" / "Local" / app_name

    if system == "darwin":
        return Path.home() / "Library" / "Application Support" / app_name

    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / app_name
    return Path.home() / ".local" / "share" / app_name


def ensure_log_dir(app_name: str = "labelis") -> Path:
    """Create (if needed) and return the Labelis log directory."""
    d = user_data_dir(app_name) / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d
