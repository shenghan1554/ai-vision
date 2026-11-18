"""Safe output-directory creation and atomic image/JSON saving."""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def ensure_writable_directory(path: str | Path) -> Path:
    """Create and actually probe a directory before model/camera startup."""
    directory = Path(path).expanduser()
    try:
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"输出路径不是目录：{directory}")
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / f".yolov8-write-test-{uuid.uuid4().hex}"
        try:
            with probe.open("xb") as handle:
                handle.write(b"write-test")
                handle.flush()
            probe.unlink()
        finally:
            if probe.exists():
                probe.unlink()
    except ValueError:
        raise
    except (PermissionError, OSError) as exc:
        raise PermissionError(
            f"输出目录不可写：{directory}\n"
            "请选择其他目录（例如程序目录下的 output），或检查 Windows "
            "“受控文件夹访问权限”、文件占用和目录权限。\n"
            f"系统错误：{exc}"
        ) from exc
    return directory


def timestamped_result_path(directory: str | Path, extension: str = ".jpg") -> Path:
    """Return a local-time result name, with a collision-safe numeric suffix."""
    suffix = extension if extension.startswith(".") else f".{extension}"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    directory = Path(directory)
    candidate = directory / f"{stamp}{suffix.lower()}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stamp}_{counter:03d}{suffix.lower()}"
        counter += 1
    return candidate


def create_run_output_directory(base: str | Path) -> Path:
    """Create a unique per-run folder to avoid overwriting/locking old files."""
    base_dir = ensure_writable_directory(base)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    candidate = base_dir / f"run_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = base_dir / f"run_{stamp}_{suffix}"
        suffix += 1
    return ensure_writable_directory(candidate)


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """Write bytes atomically and clean up partial files on failure."""
    destination = Path(path)
    ensure_writable_directory(destination.parent)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
        os.replace(temporary, destination)
    except (PermissionError, OSError) as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise PermissionError(
            f"无法保存文件：{destination}\n"
            "请检查输出目录权限、同名文件是否只读或被其他程序占用。\n"
            f"系统错误：{exc}"
        ) from exc
    return destination


def encode_image(path: str | Path, image: np.ndarray) -> bytes:
    """Encode an image for atomic Unicode-path-safe saving."""
    destination = Path(path)
    suffix = destination.suffix or ".jpg"
    try:
        ok, encoded = cv2.imencode(suffix, image)
    except cv2.error as exc:
        raise ValueError(f"图片编码失败（{suffix}）：{exc}") from exc
    if not ok:
        raise ValueError(f"图片编码失败：{suffix}")
    return encoded.tobytes()
