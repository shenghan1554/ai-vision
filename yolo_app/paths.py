"""Paths that work both from source and from a PyInstaller build."""

from __future__ import annotations

import sys
from pathlib import Path


def app_dir() -> Path:
    """Return the writable application directory."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_path(relative: str | Path) -> Path:
    """Return a bundled resource path, also accepting an absolute path."""
    path = Path(relative)
    if path.is_absolute():
        return path
    base = Path(getattr(sys, "_MEIPASS", app_dir()))
    return base / path


def ensure_user_dirs() -> None:
    """Create conventional portable input/output directories."""
    for name in ("input", "output"):
        (app_dir() / name).mkdir(parents=True, exist_ok=True)
