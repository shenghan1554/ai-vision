"""Manual rapid open/close stress test for intermittent UVC driver failures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo_app.camera import CameraOptions, open_usb_camera


def main() -> None:
    parser = argparse.ArgumentParser(description="USB 快速连续打开/关闭压力测试")
    parser.add_argument("--source", default="0")
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=2500)
    args = parser.parse_args()
    for backend in ("dshow", "msmf", "auto"):
        print(f"BACKEND {backend}", flush=True)
        for cycle in range(args.cycles):
            try:
                stream = open_usb_camera(
                    args.source,
                    CameraOptions(
                        backend=backend,
                        warmup_frames=0,
                        timeout_ms=args.timeout,
                    ),
                )
                print(f"  {cycle}: OK {stream.description}", flush=True)
                stream.release()
            except Exception as exc:
                print(f"  {cycle}: FAIL {exc}", flush=True)


if __name__ == "__main__":
    main()
