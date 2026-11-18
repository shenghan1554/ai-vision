"""Measure persistent-camera photo latency on a real USB/Hikvision device."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo_app.camera import CameraOptions, open_camera
from yolo_app.engine import YOLOv8Engine
from yolo_app.output import create_run_output_directory
from yolo_app.processor import process_source


def main() -> None:
    parser = argparse.ArgumentParser(description="持久连接拍照速度实测")
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", default="models/yolov8n.onnx")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    engine = YOLOv8Engine(args.model, device="cpu")
    engine.warmup()
    output = create_run_output_directory(args.output)
    options = CameraOptions(warmup_frames=0)
    opened = time.perf_counter()
    stream = open_camera(args.source, options)
    open_seconds = time.perf_counter() - opened
    print(f"initial open: {open_seconds:.3f}s ({stream.description})")
    try:
        for index in range(max(1, args.rounds)):
            started = time.perf_counter()
            events = list(
                process_source(
                    engine,
                    args.source,
                    output,
                    save_json=False,
                    camera_options=options,
                    mode="photo",
                    stream=stream,
                    keep_stream=True,
                )
            )
            elapsed = time.perf_counter() - started
            photo = next(event for event in events if event.kind == "photo")
            print(
                f"photo {index + 1}: {elapsed:.3f}s, "
                f"inference={photo.inference_ms:.0f}ms, saved={photo.saved_path}"
            )
    finally:
        stream.release()


if __name__ == "__main__":
    main()
