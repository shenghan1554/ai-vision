"""USB/UVC and Hikvision MVS camera discovery and single-shot capture."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from typing import Protocol
from urllib.parse import urlparse

import cv2
import numpy as np


CAMERA_SCHEMES = {"usb", "hik", "hikvision"}


class CameraError(ValueError):
    """A camera-specific error with an actionable user-facing message."""


@dataclass(slots=True)
class CameraDevice:
    source: str
    name: str
    kind: str
    serial: str = ""
    detail: str = ""
    available: bool = True


@dataclass(slots=True)
class CameraOptions:
    backend: str = "auto"
    width: int = 0
    height: int = 0
    fps: float = 0.0
    timeout_ms: int = 3000
    warmup_frames: int = 1
    mjpg: bool = False
    serial: str = ""
    exposure_us: float = 0.0
    gain: float = 0.0


class FrameStream(Protocol):
    description: str
    control_notes: list[str]

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def get_fps(self) -> float: ...

    def release(self) -> None: ...


class OpenCVStream:
    """Uniform stream wrapper for OpenCV USB cameras."""

    def __init__(
        self,
        capture: cv2.VideoCapture,
        description: str,
        pending_frame: np.ndarray | None = None,
        control_notes: list[str] | None = None,
    ) -> None:
        self._capture = capture
        self.description = description
        self._pending_frame = pending_frame
        self.control_notes = list(control_notes or [])

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._pending_frame is not None:
            frame = self._pending_frame
            self._pending_frame = None
            return True, frame
        ok, frame = self._capture.read()
        return bool(ok and frame is not None and frame.size), frame if ok else None

    def get_fps(self) -> float:
        value = float(self._capture.get(cv2.CAP_PROP_FPS))
        return value if np.isfinite(value) and value > 0 else 25.0

    def release(self) -> None:
        self._capture.release()


class LatestFrameStream:
    """Background acquisition with an immediately readable latest-frame buffer.

    UVC devices commonly produce frames only once per second. A synchronous
    ``capture.read()`` therefore adds a full frame interval to every photo.
    This wrapper keeps one reader thread alive and lets photo capture consume
    the newest completed frame immediately. Live mode uses ``read_next`` to
    wait for a new sequence and never repeatedly infers the same frame.
    """

    def __init__(self, base: FrameStream, timeout_ms: int = 3000) -> None:
        self._base = base
        self.description = base.description
        self.control_notes = list(getattr(base, "control_notes", []) or [])
        self._timeout_s = max(0.1, float(timeout_ms) / 1000.0)
        self._condition = threading.Condition()
        self._latest_frame: np.ndarray | None = None
        self._latest_seq = 0
        self._consumed_seq = 0
        self._reader_error: Exception | None = None
        self._stop_event = threading.Event()
        self._released = False
        # The first frame was already validated by open_camera and is normally
        # immediately available, so no additional device wait is introduced.
        try:
            ok, frame = self._base.read()
        except Exception:
            try:
                self._base.release()
            except Exception:
                pass
            raise
        if ok and frame is not None:
            self._latest_frame = frame
            self._latest_seq = 1
        self._thread = threading.Thread(
            target=self._acquire_loop, name="camera-frame-reader", daemon=True
        )
        self._thread.start()

    def _acquire_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                ok, frame = self._base.read()
                if not ok or frame is None or not frame.size:
                    time.sleep(0.01)
                    continue
                with self._condition:
                    self._latest_frame = frame
                    self._latest_seq += 1
                    self._condition.notify_all()
            except Exception as exc:
                with self._condition:
                    self._reader_error = exc
                    self._condition.notify_all()
                break

    def read_latest(self) -> tuple[bool, np.ndarray | None]:
        """Return the newest completed frame without waiting for the next one."""
        with self._condition:
            if self._released:
                return False, None
            if self._latest_frame is None and self._reader_error is None:
                self._condition.wait(self._timeout_s)
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame

    def read_next(self) -> tuple[bool, np.ndarray | None]:
        """Wait for and return a frame newer than the previous live result."""
        with self._condition:
            if self._released:
                return False, None
            if self._latest_seq <= self._consumed_seq and self._reader_error is None:
                self._condition.wait(self._timeout_s)
            if self._latest_seq <= self._consumed_seq:
                return False, None
            self._consumed_seq = self._latest_seq
            return True, self._latest_frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.read_next()

    def get_fps(self) -> float:
        return self._base.get_fps()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        # Unblock a driver read before joining the daemon. Most backends make
        # release idempotent and abort a blocked read promptly.
        try:
            self._base.release()
        except Exception:
            pass
        self._thread.join(timeout=1.0)
        try:
            self._base.release()
        except Exception:
            pass


def is_camera_source(source: object) -> bool:
    if isinstance(source, int):
        return True
    text = str(source).strip().lower()
    return text.isdigit() or any(
        text.startswith(f"{scheme}://") for scheme in CAMERA_SCHEMES
    )


def normalise_camera_source(source: str | int) -> str:
    text = str(source).strip()
    if text.isdigit():
        return f"usb://{int(text)}"
    parsed = urlparse(text)
    scheme = parsed.scheme.lower()
    if scheme == "hikvision":
        return "hik://" + text.split("://", 1)[1]
    if scheme not in {"usb", "hik"}:
        raise CameraError(f"不支持的摄像头来源：{source}")
    return text


def _source_id(source: str | int) -> str:
    normalised = normalise_camera_source(source)
    parsed = urlparse(normalised)
    return (parsed.netloc + parsed.path).strip("/")


def camera_output_stem(source: str | int) -> str:
    scheme = urlparse(normalise_camera_source(source)).scheme.lower()
    identity = _source_id(source) or "0"
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in identity)
    prefix = "camera" if scheme == "usb" else scheme
    return f"{prefix}_{safe}"


def _backend_values() -> dict[str, int]:
    return {
        "any": cv2.CAP_ANY,
        "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
        "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
    }


_LAST_USB_BACKEND: dict[int, str] = {}


def _open_capture(index: int, backend: int, timeout_ms: int) -> cv2.VideoCapture:
    # Do NOT pass CAP_PROP_OPEN_TIMEOUT_MSEC / CAP_PROP_READ_TIMEOUT_MSEC in
    # the constructor. Several UVC/MSMF drivers reject property 53 during
    # openFinalize and therefore fail before delivering any frame. Timeout
    # handling belongs above the driver-specific OpenCV open call instead.
    del timeout_ms  # retained in the signature for backwards compatibility
    return cv2.VideoCapture(index, backend)


def _apply_usb_controls(
    capture: cv2.VideoCapture, options: CameraOptions
) -> list[str]:
    """Apply optional exposure/gain to a UVC camera, returning user notes.

    UVC exposure is normally expressed in log2 seconds (DirectShow/MSMF), while
    the UI/CLI uses microseconds for both USB and Hikvision. Values are applied
    best-effort: unsupported controls produce a note instead of failing capture.
    """
    notes: list[str] = []
    if options.exposure_us > 0:
        seconds = max(1e-6, float(options.exposure_us) / 1_000_000.0)
        log_value = -math.log2(seconds)
        applied = False
        # Disable auto exposure first; drivers expose manual mode differently.
        for auto_value in (0.25, 1.0):
            try:
                capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_value)
            except (cv2.error, OSError):
                continue
            try:
                if capture.set(cv2.CAP_PROP_EXPOSURE, log_value):
                    applied = True
                    break
            except (cv2.error, OSError):
                continue
        if not applied:
            # A few UVC devices accept an absolute exposure value instead.
            try:
                applied = bool(
                    capture.set(cv2.CAP_PROP_EXPOSURE, float(options.exposure_us))
                )
            except (cv2.error, OSError):
                applied = False
        if applied:
            notes.append(f"曝光 {options.exposure_us:g} µs 已设置")
        else:
            notes.append(
                f"曝光 {options.exposure_us:g} µs 不受该 USB 驱动支持，已保持自动"
            )
    if options.gain > 0:
        try:
            applied = bool(capture.set(cv2.CAP_PROP_GAIN, float(options.gain)))
        except (cv2.error, OSError):
            applied = False
        if applied:
            notes.append(f"增益 {options.gain:g} 已设置")
        else:
            notes.append(f"增益 {options.gain:g} 不受该 USB 驱动支持")
    return notes


def open_usb_camera(source: str | int, options: CameraOptions | None = None) -> OpenCVStream:
    """Open a UVC camera and verify that it actually delivers a frame."""
    options = options or CameraOptions()
    identity = _source_id(source)
    if not identity.isdigit():
        raise CameraError(f"USB 摄像头编号无效：{identity}")
    index = int(identity)
    backends = _backend_values()
    requested = options.backend.strip().lower()
    if requested not in {"auto", *backends}:
        raise CameraError("USB 后端仅支持 auto、dshow、msmf 或 any。")
    if requested == "auto":
        default_order = ["dshow", "msmf", "any"] if sys.platform == "win32" else ["any"]
        cached = _LAST_USB_BACKEND.get(index)
        order = ([cached] + [name for name in default_order if name != cached]
                 if cached else default_order)
    else:
        order = [requested]

    attempts: list[str] = []
    profiles: list[tuple[str, CameraOptions]] = [("请求参数", options)]
    if options.mjpg or options.width > 0 or options.height > 0 or options.fps > 0:
        # Some UVC devices reject MJPG/custom modes even though they work with
        # their driver's defaults. Always retain a safe, automatic fallback.
        profiles.append(
            (
                "驱动默认格式",
                replace(
                    options,
                    width=0,
                    height=0,
                    fps=0.0,
                    mjpg=False,
                    warmup_frames=0,
                ),
            )
        )

    for backend_name in order:
        for profile_index, (profile_name, profile) in enumerate(profiles):
            for retry in range(2):
                capture = _open_capture(index, backends[backend_name], profile.timeout_ms)
                pending = None
                try:
                    if not capture.isOpened():
                        attempts.append(
                            f"{backend_name}/{profile_name}（第 {retry + 1} 次）：无法打开设备"
                        )
                    else:
                        # FourCC must be selected before width/height/FPS on
                        # several DirectShow drivers; never force it in fallback.
                        if profile.mjpg:
                            capture.set(
                                cv2.CAP_PROP_FOURCC,
                                cv2.VideoWriter_fourcc(*"MJPG"),
                            )
                        if profile.width > 0:
                            capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(profile.width))
                        if profile.height > 0:
                            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(profile.height))
                        if profile.fps > 0:
                            capture.set(cv2.CAP_PROP_FPS, float(profile.fps))
                        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        control_notes = _apply_usb_controls(capture, profile)

                        for frame_attempt in range(max(1, int(profile.warmup_frames) + 1)):
                            ok, frame = capture.read()
                            if ok and frame is not None and frame.size:
                                pending = frame
                            if frame_attempt < int(profile.warmup_frames):
                                time.sleep(0.03)
                        if pending is None:
                            attempts.append(
                                f"{backend_name}/{profile_name}：已打开但读取不到画面"
                            )
                        else:
                            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            mode = "，兼容模式" if profile_index else ""
                            _LAST_USB_BACKEND[index] = backend_name
                            return OpenCVStream(
                                capture,
                                f"USB 摄像头 {index}（{backend_name.upper()}，"
                                f"{width}x{height}{mode}）",
                                pending,
                                control_notes,
                            )
                except (cv2.error, OSError) as exc:
                    attempts.append(f"{backend_name}/{profile_name}：{exc}")
                finally:
                    if pending is None:
                        capture.release()
                if pending is None and retry == 0:
                    # Device discovery also opens/closes the camera. A short
                    # grace period handles drivers that release asynchronously.
                    time.sleep(0.25)
    raise CameraError(
        f"无法打开 USB 摄像头 {index}，或设备被其他程序占用。\n"
        "已尝试后端切换、短间隔重试和驱动默认像素格式。请关闭相机助手/"
        "视频会议软件，检查隐私权限；MSMF 失败时优先选择 dshow。\n"
        + "\n".join(attempts)
    )


def open_camera(source: str | int, options: CameraOptions | None = None) -> FrameStream:
    """Open any supported camera through the common FrameStream interface."""
    options = options or CameraOptions()
    normalised = normalise_camera_source(source)
    scheme = urlparse(normalised).scheme.lower()
    if scheme == "usb":
        return LatestFrameStream(
            open_usb_camera(normalised, options), options.timeout_ms
        )
    if scheme == "hik":
        from .hikvision import open_hikvision_camera

        return LatestFrameStream(
            open_hikvision_camera(normalised, options), options.timeout_ms
        )
    raise CameraError(f"不支持的摄像头类型：{scheme}")


def _windows_camera_names() -> list[str]:
    if sys.platform != "win32":
        return []
    command = (
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
        "Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -eq 'Camera' } | "
        "ForEach-Object { $_.Name } | ConvertTo-Json"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            capture_output=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout.decode("utf-8-sig", errors="replace") or "[]")
        if isinstance(data, str):
            return [data]
        return [str(item) for item in data] if isinstance(data, list) else []
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return []


def enumerate_usb_cameras(
    max_index: int = 4,
    options: CameraOptions | None = None,
) -> list[CameraDevice]:
    """Probe UVC indices on demand; never run this on the GUI thread."""
    options = options or CameraOptions(warmup_frames=0)
    names = _windows_camera_names()
    devices: list[CameraDevice] = []
    for index in range(max(1, int(max_index))):
        try:
            stream = open_usb_camera(index, options)
        except CameraError:
            continue
        friendly = names[index] if index < len(names) else "USB 免驱摄像头"
        detail = stream.description
        stream.release()
        devices.append(
            CameraDevice(
                source=f"usb://{index}",
                name=friendly,
                kind="usb",
                detail=detail,
            )
        )
    return devices


def discover_cameras(
    probe_usb: bool = True,
    max_usb_index: int = 4,
    options: CameraOptions | None = None,
) -> tuple[list[CameraDevice], list[str]]:
    """Return all available devices plus non-fatal SDK/discovery warnings."""
    options = options or CameraOptions(warmup_frames=0)
    devices: list[CameraDevice] = []
    warnings: list[str] = []
    if probe_usb:
        try:
            devices.extend(enumerate_usb_cameras(max_usb_index, options))
        except Exception as exc:
            warnings.append(f"USB 摄像头枚举失败：{exc}")
    try:
        from .hikvision import enumerate_hikvision_devices

        devices.extend(enumerate_hikvision_devices())
    except Exception as exc:
        warnings.append(str(exc))
    return devices, warnings
