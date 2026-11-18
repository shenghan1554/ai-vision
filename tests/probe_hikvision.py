"""Manual real-Hikvision MVS stream probe (not unit-discovered)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo_app.camera import CameraOptions
from yolo_app.hikvision import open_hikvision_camera
from yolo_app.output import create_run_output_directory, timestamped_result_path
from yolo_app.processor import write_image


def main() -> None:
    parser = argparse.ArgumentParser(description="海康 MVS 真实相机取帧诊断")
    parser.add_argument("--source", required=True, help="hik://索引或序列号")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    stream = open_hikvision_camera(
        args.source, CameraOptions(timeout_ms=4000, warmup_frames=1)
    )
    last = None
    received = 0
    try:
        for _ in range(max(1, args.frames * 3)):
            ok, frame = stream.read()
            if ok and frame is not None:
                last = frame
                received += 1
                if received >= args.frames:
                    break
        if last is None:
            raise RuntimeError("MVS camera opened but delivered no frames")
        output = create_run_output_directory(args.output)
        saved = write_image(timestamped_result_path(output, ".jpg"), last)
        print(f"Stream: {stream.description}")
        print(f"Frame: {last.shape[1]}x{last.shape[0]}, channels={last.shape[2]}")
        print(f"FPS: {stream.get_fps():.2f}, frames={received}")
        print(f"Saved: {saved}")
    finally:
        stream.release()


if __name__ == "__main__":
    main()
