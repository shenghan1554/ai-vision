#!/usr/bin/env python3
"""CLI for YOLOv8 images and live USB/Hikvision/GenICam cameras."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from yolo_app import __version__
from yolo_app.camera import CameraOptions, discover_cameras
from yolo_app.engine import YOLOv8Engine
from yolo_app.output import create_run_output_directory
from yolo_app.paths import app_dir, resource_path
from yolo_app.processor import RunEvent, process_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YOLOv8 图片/相机单次拍照检测（PT / ONNX，无视频功能）"
    )
    parser.add_argument(
        "--source",
        "-s",
        default=str(resource_path("samples/bus.jpg")),
        help="图片、图片目录、usb://0 或 hik://序列号",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=str(resource_path("models/yolov8n.pt")),
        help="YOLOv8 .pt 或 .onnx 模型路径",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸，默认 640")
    parser.add_argument("--device", default="auto", help="auto、cpu 或 CUDA 编号")
    parser.add_argument(
        "--output",
        "-o",
        default=str(app_dir() / "output"),
        help="结果目录（默认：程序目录下 output）",
    )
    parser.add_argument("--save-json", action="store_true", help="保存图片JSON/相机JSONL")
    parser.add_argument("--no-save", action="store_true", help="不保存任何结果文件")
    parser.add_argument("--run-subdir", action="store_true", help="创建独立 run_* 结果目录")
    parser.add_argument("--list-cameras", action="store_true", help="枚举相机后退出，不加载模型")
    parser.add_argument("--max-usb-index", type=int, default=4, help="USB 枚举最大索引数")
    parser.add_argument(
        "--camera-backend",
        choices=["auto", "dshow", "msmf", "any"],
        default="auto",
        help="USB 摄像头后端",
    )
    parser.add_argument("--camera-width", type=int, default=0, help="相机宽度，0 为默认")
    parser.add_argument("--camera-height", type=int, default=0, help="相机高度，0 为默认")
    parser.add_argument("--camera-fps", type=float, default=0.0, help="相机帧率，0 为默认")
    parser.add_argument("--camera-timeout", type=int, default=3000, help="打开/读取超时毫秒")
    parser.add_argument("--no-mjpg", action="store_true", help="USB 不请求 MJPG 像素格式")
    parser.add_argument(
        "--mode",
        choices=["photo", "live"],
        default="photo",
        help="相机模式：photo=拍一张识别（默认），live=实时识别直到 Ctrl+C",
    )
    parser.add_argument("--exposure", type=float, default=0.0, help="曝光时间(微秒)，0为不修改")
    parser.add_argument("--gain", type=float, default=0.0, help="模拟增益，0为不修改")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def _print_detections(event: RunEvent) -> None:
    print(f"\n输入：{event.source}")
    if not event.detections:
        print("未检测到目标。")
    for item in event.detections:
        box = ", ".join(f"{value:.1f}" for value in item.box)
        print(f"  {item.name:<20} 置信度={item.confidence:.3f} bbox_xyxy=({box})")
    print(f"共 {len(event.detections)} 个目标，单帧耗时 {event.inference_ms:.1f} ms")
    if event.saved_path:
        print(f"已保存：{event.saved_path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    camera_options = CameraOptions(
        backend=args.camera_backend,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        timeout_ms=args.camera_timeout,
        mjpg=not args.no_mjpg,
        exposure_us=args.exposure,
        gain=args.gain,
    )
    try:
        if args.list_cameras:
            devices, warnings = discover_cameras(
                probe_usb=True,
                max_usb_index=args.max_usb_index,
                options=camera_options,
            )
            if not devices:
                print("未发现可用相机。")
            for device in devices:
                print(f"{device.name}\t{device.detail}\t{device.source}")
            for warning in warnings:
                print(f"提示：{warning}", file=sys.stderr)
            return 0

        engine = YOLOv8Engine(
            model_path=args.model,
            names_path=None,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
        )
        print(f"模型：{engine.model_description}")
        output = Path(args.output)
        if args.run_subdir and not args.no_save:
            output = create_run_output_directory(output)
            print(f"本次输出：{output}")
        for event in process_source(
            engine,
            source=args.source,
            output_dir=output,
            save_json=args.save_json,
            save=not args.no_save,
            camera_options=camera_options,
            mode=args.mode,
        ):
            if event.kind in {"image", "photo"}:
                _print_detections(event)
            elif event.kind == "camera_started":
                print(f"\n{event.message}")
            elif event.kind == "frame":
                if event.frame_index == 1 or event.frame_index % 30 == 0:
                    print(
                        f"\r实时识别 第 {event.frame_index} 帧，"
                        f"{event.inference_ms:.0f} ms",
                        end="",
                        flush=True,
                    )
            elif event.kind == "camera_finished":
                print(f"\n{event.message}")
            elif event.kind == "completed":
                print(f"\n{event.message}")
                if event.saved_path:
                    print(f"输出目录：{event.saved_path}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n用户中断处理。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
