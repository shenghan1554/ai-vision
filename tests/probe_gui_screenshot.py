"""Capture the AI Vision window layout for visual verification."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo_app.gui import YOLOv8DesktopApp


def main() -> None:
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "layout.png")
    app = YOLOv8DesktopApp()
    app.root.update_idletasks()
    app.root.update()
    # Let the theme/layout settle before grabbing the window.
    for _ in range(30):
        app.root.update()
        time.sleep(0.05)
    app.root.update_idletasks()

    from PIL import ImageGrab

    x = app.root.winfo_rootx()
    y = app.root.winfo_rooty()
    w = app.root.winfo_width()
    h = app.root.winfo_height()
    image = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"saved {output} ({image.width}x{image.height})")
    print(f"preview={app.preview_canvas.winfo_width()}x{app.preview_canvas.winfo_height()}")
    print(f"log={app.log_text.winfo_width()}x{app.log_text.winfo_height()}")
    app._on_close()


if __name__ == "__main__":
    main()