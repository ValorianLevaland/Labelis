"""Labelis launcher (double-click friendly).

Run with:
    python _app.py

Or just double-click this file (Windows file association must point to the
Python interpreter of your desired environment).
"""

from __future__ import annotations

from labelis.app import main

if __name__ == "__main__":
    raise SystemExit(main())
