"""Image, image-directory, and single-shot camera processing."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from .camera import (
    CameraError,
    CameraOptions,
    is_camera_source,
    open_camera,
)
from .engine import Detection, DetectionResult, YOLOv8Engine
from .output import (
    atomic_write_bytes,
    encode_image,
    ensure_writable_directory,
    timestamped_result_path,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class RunEvent:
    """A processing update consumed by either the GUI or the CLI."""

    kind: str
    source: str = ""
    current: int = 0
    total: int = 0
    frame_index: int = 0
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    image: np.ndarray | None = None
    message: str = ""
    saved_path: str = ""


def read_image(path: str | Path) -> np.ndarray:
    """Read an image, including from Unicode paths on Windows."""
    source = Path(path)
    try:
        data = np.fromfile(str(source), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, cv2.error) as exc:
        raise ValueError(f"无法读取图片：{source}\n{exc}") from exc
    if image is None:
        raise ValueError(f"无法读取图片或图片格式不受支持：{source}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> Path:
    """Encode and atomically save an image with actionable errors."""
    destination = Path(path)
    try:
        return atomic_write_bytes(destination, encode_image(destination, image))
    except (PermissionError, ValueError):
        raise
    except (OSError, cv2.error) as exc:
        raise ValueError(f"无法保存图片：{destination}\n{exc}") from exc


def _image_record(
    engine: YOLOv8Engine,
    source: str,
    result: DetectionResult,
    original: np.ndarray,
) -> dict:
    return {
        "model": str(engine.model_path),
        "source": str(source),
        "image_width": int(original.shape[1]),
        "image_height": int(original.shape[0]),
        "inference_ms": round(result.inference_ms, 3),
        "detections": [item.to_dict() for item in result.detections],
    }


def _write_json(path: Path, record: dict) -> None:
    payload = json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")
    atomic_write_bytes(path, payload)


def _save_result(
    engine: YOLOv8Engine,
    output_dir: Path,
    original: np.ndarray,
    result: DetectionResult,
    source_label: str,
    extension: str,
    save: bool,
    save_json: bool,
) -> str:
    if not save:
        return ""
    result_path = timestamped_result_path(output_dir, extension or ".jpg")
    saved_path = str(write_image(result_path, result.annotated))
    if save_json:
        _write_json(
            result_path.with_suffix(".json"),
            _image_record(engine, source_label, result, original),
        )
    return saved_path


def _process_image(
    engine: YOLOv8Engine,
    source: Path,
    output_dir: Path,
    save: bool,
    save_json: bool,
    current: int,
    total: int,
) -> RunEvent:
    original = read_image(source)
    result = engine.predict(original)
    saved_path = _save_result(
        engine,
        output_dir,
        original,
        result,
        str(source),
        source.suffix,
        save,
        save_json,
    )
    return RunEvent(
        kind="image",
        source=str(source),
        current=current,
        total=total,
        frame_index=1,
        detections=result.detections,
        inference_ms=result.inference_ms,
        image=result.annotated,
        saved_path=saved_path,
        message=f"检测到 {len(result.detections)} 个目标",
    )


def _read_latest(stream: Any) -> tuple[bool, np.ndarray | None]:
    method = getattr(stream, "read_latest", None)
    return method() if callable(method) else stream.read()


def _read_next(stream: Any) -> tuple[bool, np.ndarray | None]:
    method = getattr(stream, "read_next", None)
    return method() if callable(method) else stream.read()


def _capture_camera_photo(
    engine: YOLOv8Engine,
    source: str | int,
    output_dir: Path,
    save: bool,
    save_json: bool,
    cancel_event: threading.Event,
    camera_options: CameraOptions,
    stream: Any = None,
    keep_stream: bool = False,
) -> Iterator[RunEvent]:
    """Grab one still image, detect and save.

    ``stream`` may be an already-open persistent stream (used by the GUI), in
    which case no device open/close cost is paid. If ``keep_stream`` is false,
    the stream is released exactly once when this function exits.
    """
    if stream is None:
        stream = open_camera(source, camera_options)
    description = stream.description
    control_notes = list(getattr(stream, "control_notes", []) or [])
    message = f"正在拍照：{description}"
    if control_notes:
        message += "；" + "；".join(control_notes)

    yield RunEvent(
        kind="camera_started",
        source=description,
        total=0,
        message=message,
    )

    try:
        frame = None
        # Optional extra settle frames (0 by default => use the open-frame).
        for _ in range(max(0, int(camera_options.warmup_frames))):
            if cancel_event.is_set():
                break
            ok, candidate = _read_next(stream)
            if ok and candidate is not None and candidate.size:
                frame = candidate
        if frame is None:
            deadline = time.monotonic() + max(2.0, float(camera_options.timeout_ms) / 1000.0)
            failures = 0
            while not cancel_event.is_set() and time.monotonic() < deadline:
                ok, candidate = _read_latest(stream)
                if ok and candidate is not None and candidate.size:
                    frame = candidate
                    break
                failures += 1
                time.sleep(0.02)
            if frame is None:
                raise CameraError(
                    f"未能从 {description} 获取图像（{failures} 次读取失败）。\n"
                    "请检查设备连接、网络/网卡、驱动，以及是否被其他软件占用。"
                )

        result = engine.predict(frame)
        saved_path = _save_result(
            engine,
            output_dir,
            frame,
            result,
            description,
            ".jpg",
            save,
            save_json,
        )
        yield RunEvent(
            kind="photo",
            source=description,
            total=1,
            current=1,
            frame_index=1,
            detections=result.detections,
            inference_ms=result.inference_ms,
            image=result.annotated,
            saved_path=saved_path,
            message=f"拍照检测完成：{len(result.detections)} 个目标",
        )
    finally:
        if not keep_stream:
            stream.release()


def _run_live_camera(
    engine: YOLOv8Engine,
    source: str | int,
    output_dir: Path,
    save: bool,
    save_json: bool,
    cancel_event: threading.Event,
    camera_options: CameraOptions,
    stream: Any = None,
    keep_stream: bool = False,
) -> Iterator[RunEvent]:
    """Continuous live detection until cancelled; saves last frame on stop.

    A persistent stream can be supplied to avoid re-opening the camera between
    live runs. It remains open when ``keep_stream`` is true.
    """
    if stream is None:
        stream = open_camera(source, camera_options)
    description = stream.description
    control_notes = list(getattr(stream, "control_notes", []) or [])
    message = f"开始实时识别：{description}"
    if control_notes:
        message += "；" + "；".join(control_notes)
    yield RunEvent(kind="camera_started", source=description, total=0, message=message)

    frame_index = 0
    last_annotated = None
    last_preview = 0.0
    failures = 0
    try:
        while not cancel_event.is_set():
            ok, frame = _read_next(stream)
            if not ok or frame is None or not frame.size:
                failures += 1
                if failures >= 10:
                    raise CameraError(
                        f"{description} 连续 {failures} 次读取失败。\n"
                        "请检查设备连接、网络/网卡、驱动和是否被其他软件占用。"
                    )
                time.sleep(0.02)
                continue
            failures = 0
            frame_index += 1
            result = engine.predict(frame)
            last_annotated = result.annotated
            now = time.monotonic()
            with_preview = frame_index == 1 or now - last_preview >= 0.05
            if with_preview:
                last_preview = now
            yield RunEvent(
                kind="frame",
                source=description,
                total=0,
                frame_index=frame_index,
                detections=result.detections if with_preview else [],
                inference_ms=result.inference_ms,
                image=result.annotated if with_preview else None,
                message=f"第 {frame_index} 帧：{len(result.detections)} 个目标",
            )
    finally:
        if not keep_stream:
            stream.release()

    snapshot_path = ""
    if save and last_annotated is not None:
        snapshot_path = str(
            write_image(timestamped_result_path(output_dir, ".jpg"), last_annotated)
        )
        if save_json:
            _write_json(
                Path(snapshot_path).with_suffix(".json"),
                {
                    "model": str(engine.model_path),
                    "source": description,
                    "frames": frame_index,
                    "detections": [],
                },
            )
    cancelled = cancel_event.is_set()
    yield RunEvent(
        kind="camera_finished",
        source=description,
        total=0,
        frame_index=frame_index,
        saved_path=snapshot_path,
        message=(
            f"已停止，共处理 {frame_index} 帧；已保存最后一帧"
            if cancelled and snapshot_path
            else f"实时识别结束，共处理 {frame_index} 帧"
        ),
    )


def process_source(
    engine: YOLOv8Engine,
    source: str | Path | int,
    output_dir: str | Path,
    save_json: bool = False,
    cancel_event: threading.Event | None = None,
    save: bool = True,
    camera_options: CameraOptions | None = None,
    mode: str = "photo",
    stream: Any = None,
    keep_stream: bool = False,
) -> Iterator[RunEvent]:
    """Process images or a camera; mode is "photo" (single shot) or "live"."""
    cancel_event = cancel_event or threading.Event()
    camera_options = camera_options or CameraOptions()
    normalised_mode = str(mode).strip().lower()
    if normalised_mode not in {"photo", "live"}:
        raise ValueError(f"不支持的相机模式：{mode}（可选 photo 或 live）")
    output_dir = Path(output_dir).expanduser()
    if save:
        ensure_writable_directory(output_dir)

    if is_camera_source(source):
        yield RunEvent(kind="started", source=str(source), total=0)
        worker = _run_live_camera if normalised_mode == "live" else _capture_camera_photo
        yield from worker(
            engine,
            source,
            output_dir,
            save,
            save_json,
            cancel_event,
            camera_options,
            stream=stream,
            keep_stream=keep_stream,
        )
        yield RunEvent(
            kind="completed",
            source=str(source),
            saved_path=str(output_dir) if save else "",
            message=(
                "实时识别已结束"
                if normalised_mode == "live"
                else "拍照检测已完成"
            ),
        )
        return

    source_path = Path(source).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"输入来源不存在：{source_path}")

    if source_path.is_dir():
        image_files = sorted(
            path
            for path in source_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
        if not image_files:
            raise ValueError(f"目录中没有支持的图片：{source_path}")
    else:
        if source_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError("视频功能已移除，请选择图片、图片目录或相机。")
        image_files = [source_path]

    total = len(image_files)
    yield RunEvent(kind="started", source=str(source_path), total=total)

    completed_files = 0
    for index, image_path in enumerate(image_files, start=1):
        if cancel_event.is_set():
            break
        yield _process_image(
            engine,
            image_path,
            output_dir,
            save,
            save_json,
            index,
            total,
        )
        completed_files += 1

    cancelled = cancel_event.is_set()
    yield RunEvent(
        kind="completed",
        source=str(source_path),
        current=completed_files,
        total=total,
        saved_path=str(output_dir) if save else "",
        message=(
            f"已取消；完成 {completed_files}/{total} 个文件"
            if cancelled
            else f"处理完成：{completed_files} 个文件"
        ),
    )