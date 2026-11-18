"""Manual real-UVC + modern-GUI integration probe (not unit-discovered)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo_app.gui import YOLOv8DesktopApp


def main() -> None:
    parser = argparse.ArgumentParser(description="现代 GUI + 真实相机单次拍照联调")
    parser.add_argument("--source", default="usb://0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["photo", "live"], default="photo")
    parser.add_argument("--seconds", type=float, default=4.0)


    args = parser.parse_args()
    root_dir = Path(__file__).resolve().parent.parent
    errors: list[str] = []
    app = YOLOv8DesktopApp()
    app.root.withdraw()
    app._settings_path = Path(args.output) / "settings.json"
    app.model_var.set(str(root_dir / "models" / "yolov8n.onnx"))
    app.source_var.set(args.source)
    app.output_var.set(args.output)
    app.device_var.set("cpu")
    app.unique_output_var.set(True)
    app.save_json_var.set(True)
    try:
        with patch(
            "yolo_app.gui.messagebox.showerror",
            side_effect=lambda _title, message, **_kwargs: errors.append(str(message)),
        ):
            app.camera_mode_var.set("拍照识别" if args.mode == "photo" else "实时识别")
            app.start_processing()
            if args.mode == "live":
                app.root.after(int(args.seconds * 1000), app.stop_processing)
            deadline = time.monotonic() + 45
            while app._running and time.monotonic() < deadline:
                app.root.update()
                time.sleep(0.01)
            app.root.update()
        if app._running or app._run_failed or errors or app.last_output_dir is None:
            raise RuntimeError(f"GUI camera probe failed: {errors}\n{app.log_text.get('1.0', 'end')}")
        photos = list(app.last_output_dir.glob("*.jpg"))
        if len(photos) != 1:
            raise AssertionError(f"Expected exactly one saved image, found: {photos}")
        if not photos[0].with_suffix(".json").is_file():
            raise AssertionError("JSON result missing")
        log = app.log_text.get("1.0", "end")
        expected = "最后一帧" if args.mode == "live" else "拍照检测"
        if expected not in log:
            raise AssertionError(f"Log missing {expected!r}: {log[-500:]}")
        print(f"Real GUI {args.mode} probe passed: {photos[0]}")
    finally:
        if app._cancel_event is not None:
            app._cancel_event.set()
        app._on_close()


if __name__ == "__main__":
    main()
