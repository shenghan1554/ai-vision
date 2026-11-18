"""Measure repeated photo runs through the real GUI worker lifecycle."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo_app.gui import YOLOv8DesktopApp


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI 连续拍照速度实测")
    parser.add_argument("--source", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    errors: list[str] = []
    app = YOLOv8DesktopApp()
    app.root.withdraw()
    app._settings_path = Path(args.output) / "settings.json"
    root_dir = Path(__file__).resolve().parent.parent
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
            for index in range(max(1, args.rounds)):
                started = time.perf_counter()
                app.start_processing()
                deadline = time.monotonic() + 60
                while app._running and time.monotonic() < deadline:
                    app.root.update()
                    time.sleep(0.01)
                app.root.update()
                elapsed = time.perf_counter() - started
                if app._running or app._run_failed or errors:
                    raise RuntimeError(
                        f"GUI photo run {index + 1} failed: {errors}\n"
                        f"{app.log_text.get('1.0', 'end')}"
                    )
                print(
                    f"GUI photo {index + 1}: {elapsed:.3f}s, "
                    f"persistent={app._camera_stream is not None}, "
                    f"output={app.last_output_dir}"
                )
        if app._camera_stream is None:
            raise AssertionError("Persistent camera stream was not retained")
    finally:
        if app._cancel_event is not None:
            app._cancel_event.set()
        app._on_close()
        if app._camera_stream is not None:
            raise AssertionError("Persistent stream was not released on close")
        print("GUI stream released on close: OK")


if __name__ == "__main__":
    main()
