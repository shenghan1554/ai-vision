"""Manual real-hardware UVC probe (does not run during unit discovery)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo_app.camera import CameraOptions, enumerate_usb_cameras, open_usb_camera
from yolo_app.processor import write_image


def main() -> None:
    parser = argparse.ArgumentParser(description="USB 免驱摄像头实测诊断")
    parser.add_argument("--source", default="0")
    parser.add_argument("--backend", choices=["auto", "dshow", "msmf", "any"], default="auto")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--output", default="camera-diagnostics/usb_frame.jpg")
    parser.add_argument("--enumerate", action="store_true")
    args = parser.parse_args()

    if args.enumerate:
        devices = enumerate_usb_cameras(max_index=4)
        print("Detected USB devices:", devices)
        return

    stream = open_usb_camera(
        args.source,
        CameraOptions(backend=args.backend, timeout_ms=4000),
    )
    frames = 0
    started = time.perf_counter()
    last_frame = None
    try:
        for _ in range(max(1, args.frames)):
            ok, frame = stream.read()
            if ok and frame is not None:
                frames += 1
                last_frame = frame
        elapsed = time.perf_counter() - started
        if last_frame is None:
            raise RuntimeError("Camera opened but delivered no frames")
        output = write_image(Path(args.output), last_frame)
        print(f"Stream: {stream.description}")
        print(f"Frame size: {last_frame.shape[1]}x{last_frame.shape[0]}")
        print(f"Frames: {frames}, average FPS: {frames / max(elapsed, 0.001):.2f}")
        print(f"Saved: {output}")
    finally:
        stream.release()


if __name__ == "__main__":
    main()
