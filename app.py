"""Windows desktop entry point; pass --cli to use the command-line interface."""

from __future__ import annotations

import multiprocessing
import os
import sys


def main() -> int:
    multiprocessing.freeze_support()
    if "--cli" in sys.argv:
        # A windowed PyInstaller executable has no console streams. Redirect
        # them so its optional CLI mode still creates result files reliably.
        if getattr(sys, "frozen", False):
            if sys.stdout is None:
                sys.stdout = open(os.devnull, "w", encoding="utf-8")
            if sys.stderr is None:
                sys.stderr = open(os.devnull, "w", encoding="utf-8")
        from detect import main as cli_main

        return cli_main(sys.argv[2:])

    from yolo_app.gui import run_gui

    run_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
