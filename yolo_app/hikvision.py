"""Hikvision GigE/USB3 Vision cameras through the optional MVS Python SDK.

The SDK is imported only when discovery or opening is requested.  In particular,
importing :mod:`yolo_app.camera` does not require an MVS installation.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import sys
import threading
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np

from .camera import CameraDevice, CameraError, CameraOptions, FrameStream


_MVS_IMPORT_SUFFIX = Path("Development") / "Samples" / "Python" / "MvImport"
_DEFAULT_MVS_IMPORT = Path(
    r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"
)
_HIK_HINTS = (
    "请关闭 MVS 及其他采图程序，排除设备占用；GigE 相机请确认相机、网卡 IP "
    "同网段且无冲突；并检查 MVS 授权/运行环境、USB3 驱动及 DLL/PATH 配置。"
)
_SDK_LIFETIME_LOCK = threading.RLock()
_SDK_ENUM_LOCK = threading.RLock()
_SDK_IMPORT_LOCK = threading.Lock()
_SDK_REFERENCES = 0


def _unsigned32(value: object) -> int:
    return int(value) & 0xFFFFFFFF


def _error_text(action: str, result: object) -> CameraError:
    code = _unsigned32(result)
    return CameraError(f"{action}失败，MVS 错误码 0x{code:08X}。\n{_HIK_HINTS}")


def _check_result(action: str, result: object) -> None:
    if _unsigned32(result) != 0:
        raise _error_text(action, result)


def _zero(value: object) -> None:
    """Zero a ctypes structure, while remaining friendly to lightweight mocks."""
    try:
        ctypes.memset(ctypes.byref(value), 0, ctypes.sizeof(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        pass


def _add_candidate(candidates: list[Path], value: str | os.PathLike[str]) -> None:
    text = os.path.expandvars(os.path.expanduser(str(value).strip().strip('"')))
    if not text:
        return
    path = Path(text)
    if path.name.lower() == "mvimport":
        candidates.append(path)
        return
    if path.name.lower() == "development":
        # MVCAM_COMMON_RUNENV commonly points at MVS\Development, not the
        # MVS root. Avoid generating Development\Development\... duplicates.
        candidates.append(path / "Samples" / "Python" / "MvImport")
        return
    if path.name.lower() == "python" and path.parent.name.lower() == "samples":
        candidates.append(path / "MvImport")
        return
    candidates.extend(
        (
            path / _MVS_IMPORT_SUFFIX,
            path / "Samples" / "Python" / "MvImport",
            path / "MvImport",
        )
    )


def _candidate_sdk_paths() -> list[Path]:
    """Return SDK import roots in priority order, without importing the SDK."""
    candidates: list[Path] = []
    for name in ("MVS_SDK_PYTHON_PATH", "MVS_SDK_PYTHON", "MVS_SDK_PATH"):
        for item in os.environ.get(name, "").split(os.pathsep):
            if item.strip():
                _add_candidate(candidates, item)
    for item in os.environ.get("MVCAM_COMMON_RUNENV", "").split(os.pathsep):
        if item.strip():
            _add_candidate(candidates, item)

    for variable, fallback in (
        ("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        ("ProgramFiles", r"C:\Program Files"),
    ):
        _add_candidate(candidates, Path(os.environ.get(variable, fallback)) / "MVS")
    candidates.append(_DEFAULT_MVS_IMPORT)

    package_dir = Path(__file__).resolve().parent
    for base in (package_dir, package_dir.parent, package_dir.parent.parent, Path.cwd()):
        for relative in (
            Path("MvImport"),
            _MVS_IMPORT_SUFFIX,
            Path("MVS") / _MVS_IMPORT_SUFFIX,
            Path("third_party") / "MVS" / _MVS_IMPORT_SUFFIX,
        ):
            candidates.append(base / relative)

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


_RUNTIME_HANDLES: list[object] = []
_PREPARED_RUNTIME: Path | None = None
_RUNTIME_LOAD_ERRORS: list[str] = []


def _candidate_runtime_directories() -> list[Path]:
    """Find architecture-matched MVS Runtime directories containing the DLL."""
    architecture = "Win64_x64" if sys.maxsize > 2**32 else "Win32_i86"
    values: list[str] = []
    for name in ("MVS_RUNTIME_PATH", "GENICAM_GENTL64_PATH", "MVCAM_COMMON_RUNENV"):
        values.extend(os.environ.get(name, "").split(os.pathsep))
    for variable, fallback in (
        ("ProgramFiles(x86)", r"C:\\Program Files (x86)"),
        ("ProgramFiles", r"C:\\Program Files"),
    ):
        values.append(
            str(
                Path(os.environ.get(variable, fallback))
                / "Common Files"
                / "MVS"
                / "Runtime"
                / architecture
            )
        )

    candidates: list[Path] = []
    for value in values:
        if not value.strip():
            continue
        base = Path(os.path.expandvars(os.path.expanduser(value.strip().strip('"'))))
        for directory in (base, base / architecture):
            if (directory / "MvCameraControl.dll").is_file():
                candidates.append(directory)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _prepend_windows_path(directory: Path) -> None:
    entries = os.environ.get("PATH", "").split(os.pathsep)
    if os.path.normcase(str(directory)) not in {
        os.path.normcase(entry) for entry in entries if entry
    }:
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")


def _prepare_windows_sdk_runtime() -> list[Path]:
    """Register and preload MVS DLLs before the official name-only WinDLL call.

    PyInstaller's frozen loader does not reliably use PATH for ctypes.WinDLL.
    Preloading the exact absolute DLL both resolves that difference and ensures
    the later WinDLL("MvCameraControl.dll") reuses the loaded module.
    """
    global _PREPARED_RUNTIME, _RUNTIME_LOAD_ERRORS
    if os.name != "nt":
        return []
    if _PREPARED_RUNTIME is not None:
        return [_PREPARED_RUNTIME]

    errors: list[str] = []
    architecture = "Win64_x64" if sys.maxsize > 2**32 else "Win32_i86"
    for runtime in _candidate_runtime_directories():
        dependency_dirs = [runtime, runtime / "ThirdParty"]
        cl_protocol = os.environ.get("MVCAM_GENICAM_CLPROTOCOL", "")
        if cl_protocol:
            dependency_dirs.append(Path(cl_protocol) / architecture)
        for directory in dependency_dirs:
            if not directory.is_dir():
                continue
            _prepend_windows_path(directory)
            if hasattr(os, "add_dll_directory"):
                try:
                    # Handles must stay alive for the entire SDK lifetime.
                    _RUNTIME_HANDLES.append(os.add_dll_directory(str(directory)))
                except OSError as exc:
                    errors.append(f"注册 DLL 目录 {directory}：{exc}")
        library = runtime / "MvCameraControl.dll"
        try:
            try:
                handle = ctypes.WinDLL(str(library), winmode=0)
            except TypeError:
                handle = ctypes.WinDLL(str(library))
            _RUNTIME_HANDLES.append(handle)
            _PREPARED_RUNTIME = runtime
            _RUNTIME_LOAD_ERRORS = []
            return [runtime]
        except OSError as exc:
            errors.append(f"预加载 {library}：{exc}")
    _RUNTIME_LOAD_ERRORS = errors
    return []


def _load_sdk():
    """Import MvCameraControl_class without permanently changing sys.path."""
    with _SDK_IMPORT_LOCK:
        candidates = _candidate_sdk_paths()
        runtime_directories = _prepare_windows_sdk_runtime()
        paths = [
            path
            for path in candidates
            if (path / "MvCameraControl_class.py").is_file()
        ]
        # An empty list still permits an installation already present on sys.path.
        attempts = paths or [None]
        last_error: Exception | None = None
        for path in attempts:
            inserted = str(path) if path is not None else None
            try:
                if inserted is not None:
                    sys.path.insert(0, inserted)
                return importlib.import_module("MvCameraControl_class")
            except Exception as exc:  # DLL load errors are not always ImportError.
                last_error = exc
            finally:
                if inserted is not None:
                    try:
                        sys.path.remove(inserted)
                    except ValueError:
                        pass
        existing = [path for path in candidates if path.exists()]
        shown = paths or existing or candidates[:4]
        searched = "\n".join(str(path) for path in shown[:6])
        if len(shown) > 6:
            searched += "\n……"
        runtime_info = ""
        if runtime_directories:
            runtime_info = f"\n已注册 MVS Runtime：{runtime_directories[0]}"
        elif _RUNTIME_LOAD_ERRORS:
            runtime_info = "\nRuntime 加载明细：\n" + "\n".join(_RUNTIME_LOAD_ERRORS)
        raise CameraError(
            "无法加载海康 MVS Python SDK（MvCameraControl_class）。请安装与 Python "
            "位数一致的 MVS Runtime/SDK，或设置 MVS_SDK_PYTHON_PATH 和 "
            f"MVS_RUNTIME_PATH。原始错误：{last_error}\n"
            f"已检查 Python SDK 路径：\n{searched}{runtime_info}\n{_HIK_HINTS}"
        ) from last_error


class _SdkSession:
    """Reference-counted MV_CC_Initialize/Finalize pair."""

    def __init__(self, sdk) -> None:
        self.sdk = sdk
        self.active = False

    def open(self) -> None:
        global _SDK_REFERENCES
        if self.active:
            return
        with _SDK_LIFETIME_LOCK:
            if _SDK_REFERENCES == 0:
                _check_result("初始化 MVS SDK", self.sdk.MvCamera.MV_CC_Initialize())
            _SDK_REFERENCES += 1
            self.active = True

    def close(self) -> None:
        global _SDK_REFERENCES
        if not self.active:
            return
        with _SDK_LIFETIME_LOCK:
            self.active = False
            _SDK_REFERENCES = max(0, _SDK_REFERENCES - 1)
            if _SDK_REFERENCES == 0:
                try:
                    self.sdk.MvCamera.MV_CC_Finalize()
                except Exception:
                    # Cleanup must remain best-effort during interpreter shutdown.
                    pass

    def __enter__(self) -> _SdkSession:
        self.open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _decode_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.split("\x00", 1)[0].strip()
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    elif isinstance(value, ctypes.c_char_p):
        raw = value.value or b""
    else:
        try:
            raw = memoryview(value).tobytes()  # type: ignore[arg-type]
        except TypeError:
            try:
                raw = bytes(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return str(value).split("\x00", 1)[0].strip()
    raw = raw.split(b"\x00", 1)[0]
    for encoding in ("gbk", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace").strip()


def _ip_text(value: object) -> str:
    number = _unsigned32(value)
    return ".".join(str((number >> shift) & 0xFF) for shift in (24, 16, 8, 0))


def _transport(sdk, info) -> tuple[str, object] | None:
    layer = _unsigned32(info.nTLayerType)
    if layer in {
        _unsigned32(sdk.MV_GIGE_DEVICE),
        _unsigned32(getattr(sdk, "MV_GENTL_GIGE_DEVICE", 0)),
    }:
        return "GigE", info.SpecialInfo.stGigEInfo
    if layer == _unsigned32(sdk.MV_USB_DEVICE):
        return "USB3 Vision", info.SpecialInfo.stUsb3VInfo
    return None


def _describe_device(sdk, info, index: int) -> CameraDevice | None:
    transport = _transport(sdk, info)
    if transport is None:
        return None
    transport_name, special = transport
    model = _decode_text(getattr(special, "chModelName", b""))
    serial = _decode_text(getattr(special, "chSerialNumber", b""))
    user_name = _decode_text(getattr(special, "chUserDefinedName", b""))
    display = user_name or model or "海康工业相机"
    details = [transport_name]
    if transport_name == "GigE":
        ip = _ip_text(getattr(special, "nCurrentIp", 0))
        details.append(f"IP {ip}")
    if serial:
        details.append(f"S/N {serial}")
    # Prefer a stable serial-based identity; index is only a last-resort
    # fallback for devices that do not report one.
    identity_source = f"hik://{serial}" if serial else f"hik://{index}"
    return CameraDevice(
        source=identity_source,
        name=display,
        kind="hikvision",
        serial=serial,
        detail=" | ".join(details),
    )


def _device_info(sdk, pointer):
    return ctypes.cast(pointer, ctypes.POINTER(sdk.MV_CC_DEVICE_INFO)).contents


def _enumerate_raw(sdk) -> list[tuple[int, object, CameraDevice]]:
    device_list = sdk.MV_CC_DEVICE_INFO_LIST()
    _zero(device_list)
    layers = _unsigned32(sdk.MV_GIGE_DEVICE) | _unsigned32(sdk.MV_USB_DEVICE)
    result = sdk.MvCamera.MV_CC_EnumDevices(layers, device_list)
    _check_result("枚举海康相机", result)
    found: list[tuple[int, object, CameraDevice]] = []
    count = min(int(device_list.nDeviceNum), len(device_list.pDeviceInfo))
    for index in range(count):
        pointer = device_list.pDeviceInfo[index]
        if not pointer:
            continue
        info = _device_info(sdk, pointer)
        descriptor = _describe_device(sdk, info, index)
        if descriptor is not None:
            found.append((index, info, descriptor))
    return found


def enumerate_hikvision_devices() -> list[CameraDevice]:
    """Enumerate MVS GigE and USB3 Vision devices without opening them."""
    sdk = _load_sdk()
    with _SdkSession(sdk), _SDK_ENUM_LOCK:
        return [descriptor for _index, _info, descriptor in _enumerate_raw(sdk)]


def _source_identity(source: str) -> str:
    parsed = urlsplit(str(source).strip())
    if parsed.scheme.lower() not in {"hik", "hikvision"}:
        raise CameraError(f"海康相机来源必须使用 hik:// 或 hikvision://：{source}")
    identity = unquote((parsed.netloc + parsed.path).strip("/")).strip()
    if not identity:
        raise CameraError("海康相机来源为空，请使用 hik://0 或 hik://SERIAL。")
    return identity


def _select_device(
    devices: list[tuple[int, object, CameraDevice]],
    identity: str,
    *,
    serial_only: bool = False,
) -> tuple[int, object, CameraDevice]:
    if not serial_only and identity.isdecimal():
        index = int(identity)
        for device in devices:
            if device[0] == index:
                return device
    # Fall back to a serial-number match. This also covers all-digit serials,
    # which would otherwise be mistaken for an enumeration index.
    folded = identity.casefold()
    for device in devices:
        if device[2].serial and device[2].serial.casefold() == folded:
            return device
    available = "、".join(
        f"{item[2].source}({item[2].serial or '无序列号'})" for item in devices
    ) or "无"
    raise CameraError(
        f"未找到海康设备 {identity!r}；当前枚举设备：{available}。"
        "请刷新设备列表，检查相机连接后再试。\n" + _HIK_HINTS
    )


def _convert_frame_to_bgr(camera, sdk, frame) -> np.ndarray:
    """Convert one acquired MV_FRAME_OUT to a packed, contiguous BGR ndarray."""
    info = frame.stFrameInfo
    width = int(getattr(info, "nWidth", 0)) or int(getattr(info, "nExtendWidth", 0))
    height = int(getattr(info, "nHeight", 0)) or int(getattr(info, "nExtendHeight", 0))
    source_length = int(getattr(info, "nFrameLen", 0)) or int(
        getattr(info, "nFrameLenEx", 0)
    )
    if width <= 0 or height <= 0 or source_length <= 0:
        raise CameraError(
            f"海康帧信息无效：{width}x{height}，数据长度 {source_length}。"
        )
    required = width * height * 3
    if required > 0xFFFFFFFF or source_length > 0xFFFFFFFF:
        raise CameraError("海康图像超过 MVS 像素转换接口的 32 位长度限制。")

    use_ex = hasattr(sdk, "MV_CC_PIXEL_CONVERT_PARAM_EX") and hasattr(
        camera, "MV_CC_ConvertPixelTypeEx"
    )
    if not use_ex and (width > 0xFFFF or height > 0xFFFF):
        raise CameraError("当前 MVS 旧版转换接口不支持此扩展图像尺寸。")
    params = (
        sdk.MV_CC_PIXEL_CONVERT_PARAM_EX()
        if use_ex
        else sdk.MV_CC_PIXEL_CONVERT_PARAM()
    )
    _zero(params)
    output = (ctypes.c_ubyte * required)()
    params.nWidth = width
    params.nHeight = height
    params.enSrcPixelType = info.enPixelType
    params.pSrcData = frame.pBufAddr
    params.nSrcDataLen = source_length
    params.enDstPixelType = sdk.PixelType_Gvsp_BGR8_Packed
    params.pDstBuffer = ctypes.cast(output, ctypes.POINTER(ctypes.c_ubyte))
    params.nDstBufferSize = required
    params.nDstLen = 0
    result = (
        camera.MV_CC_ConvertPixelTypeEx(params)
        if use_ex
        else camera.MV_CC_ConvertPixelType(params)
    )
    if _unsigned32(result) != 0:
        code = _unsigned32(result)
        pixel_type = _unsigned32(info.enPixelType)
        raise CameraError(
            f"海康像素转换失败，MVS 错误码 0x{code:08X}，源格式 0x{pixel_type:08X}。"
            "请检查 MVS 运行环境及相机 PixelFormat 设置。"
        )
    output_length = int(params.nDstLen)
    if output_length < required or output_length > required:
        raise CameraError(
            f"海康 BGR 输出长度异常：期望 {required} 字节，实际 {output_length} 字节。"
        )
    # BGR8_Packed is tightly packed.  Explicit strides prevent accidental use of
    # source padding or a view whose lifetime depends on the SDK image buffer.
    image = np.ndarray(
        shape=(height, width, 3),
        dtype=np.uint8,
        buffer=output,
        strides=(width * 3, 3, 1),
    )
    return image.copy(order="C")


class HikvisionStream:
    """FrameStream owning an MvCamera and one SDK initialization reference."""

    def __init__(
        self,
        camera,
        sdk,
        session: _SdkSession,
        description: str,
        options: CameraOptions,
    ) -> None:
        self._camera = camera
        self._sdk = sdk
        self._session = session
        self._options = options
        self.description = description
        self._timeout_ms = max(1, int(options.timeout_ms))
        self._handle_created = False
        self._opened = False
        self._grabbing = False
        self._released = False
        self._pending_frame: np.ndarray | None = None
        self.control_notes: list[str] = []
        self._lock = threading.RLock()

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._released:
                return False, None
            if self._pending_frame is not None:
                frame = self._pending_frame
                self._pending_frame = None
                return True, frame
            frame_out = self._sdk.MV_FRAME_OUT()
            _zero(frame_out)
            result = self._camera.MV_CC_GetImageBuffer(frame_out, self._timeout_ms)
            code = _unsigned32(result)
            if code != 0:
                timeout_codes = {
                    _unsigned32(getattr(self._sdk, "MV_E_NODATA", 0x80000007)),
                    _unsigned32(getattr(self._sdk, "MV_E_GC_TIMEOUT", 0x80000107)),
                }
                if code in timeout_codes:
                    return False, None
                raise _error_text("海康相机取帧", code)
            if not frame_out.pBufAddr:
                return False, None
            try:
                image = _convert_frame_to_bgr(self._camera, self._sdk, frame_out)
            finally:
                # GetImageBuffer and FreeImageBuffer must always be paired, even
                # when validation or pixel conversion raises an exception.
                self._camera.MV_CC_FreeImageBuffer(frame_out)
            return True, image

    def get_fps(self) -> float:
        with self._lock:
            if self._released:
                return float(self._options.fps or 25.0)
            for key in ("ResultingFrameRate", "AcquisitionFrameRate"):
                value = self._sdk.MVCC_FLOATVALUE()
                _zero(value)
                result = self._camera.MV_CC_GetFloatValue(key, value)
                current = float(value.fCurValue)
                if _unsigned32(result) == 0 and np.isfinite(current) and current > 0:
                    return current
            configured = float(self._options.fps)
            return configured if np.isfinite(configured) and configured > 0 else 25.0

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
            self._pending_frame = None
            try:
                if self._grabbing:
                    try:
                        self._camera.MV_CC_StopGrabbing()
                    finally:
                        self._grabbing = False
            finally:
                try:
                    if self._opened:
                        try:
                            self._camera.MV_CC_CloseDevice()
                        finally:
                            self._opened = False
                finally:
                    try:
                        if self._handle_created:
                            try:
                                self._camera.MV_CC_DestroyHandle()
                            finally:
                                self._handle_created = False
                    finally:
                        self._session.close()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


def _apply_exposure_and_gain(camera, options: CameraOptions) -> list[str]:
    """Apply optional exposure/gain settings (best effort across camera models).

    Values are requested only when positive; cameras lacking a given node or
    using a different naming convention keep their own defaults rather than
    failing the whole open sequence. Returns notes for the UI log.
    """
    notes: list[str] = []
    if options.exposure_us > 0:
        target = float(options.exposure_us)
        applied = False
        for node in ("ExposureTime", "ExposureTimeAbs"):
            try:
                if _unsigned32(camera.MV_CC_SetFloatValue(node, target)) == 0:
                    applied = True
                    break
            except Exception:
                continue
        notes.append(
            f"曝光 {target:g} µs 已设置"
            if applied
            else f"曝光 {target:g} µs 设置失败，已保持相机当前值"
        )
    if options.gain > 0:
        target = float(options.gain)
        applied = False
        for node in ("Gain", "AnalogGain"):
            try:
                if _unsigned32(camera.MV_CC_SetFloatValue(node, target)) == 0:
                    applied = True
                    break
            except Exception:
                continue
        notes.append(
            f"增益 {target:g} 已设置"
            if applied
            else f"增益 {target:g} 设置失败，已保持相机当前值"
        )
    return notes


def open_hikvision_camera(
    source: str,
    options: CameraOptions | None = None,
) -> FrameStream:
    """Open ``hik://INDEX`` or ``hik://SERIAL`` and start grabbing BGR frames."""
    options = options or CameraOptions()
    requested_serial = options.serial.strip()
    identity = requested_serial or _source_identity(source)
    sdk = _load_sdk()
    session = _SdkSession(sdk)
    session.open()
    stream: HikvisionStream | None = None
    try:
        # SDK enumeration owns an internal reusable device list.  Keep its lock
        # until CreateHandle has copied the selected device information.
        with _SDK_ENUM_LOCK:
            devices = _enumerate_raw(sdk)
            _index, info, descriptor = _select_device(
                devices, identity, serial_only=bool(requested_serial)
            )
            camera = sdk.MvCamera()
            stream = HikvisionStream(camera, sdk, session, descriptor.name, options)
            result = camera.MV_CC_CreateHandle(info)
            _check_result("创建海康相机句柄", result)
            stream._handle_created = True

        _check_result(
            "打开海康相机（可能已被占用）",
            camera.MV_CC_OpenDevice(sdk.MV_ACCESS_Exclusive, 0),
        )
        stream._opened = True

        layer = _unsigned32(info.nTLayerType)
        if layer in {
            _unsigned32(sdk.MV_GIGE_DEVICE),
            _unsigned32(getattr(sdk, "MV_GENTL_GIGE_DEVICE", 0)),
        }:
            # As in GrabImage.py, packet-size detection requires an open device
            # and must occur before StartGrabbing.  Failure is non-fatal.
            packet_size = int(camera.MV_CC_GetOptimalPacketSize())
            if 0 < packet_size < 0x80000000:
                camera.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)

        _check_result(
            "关闭海康相机触发模式",
            camera.MV_CC_SetEnumValue(
                "TriggerMode", getattr(sdk, "MV_TRIGGER_MODE_OFF", 0)
            ),
        )
        if options.width > 0:
            _check_result(
                "设置海康相机宽度",
                camera.MV_CC_SetIntValue("Width", int(options.width)),
            )
        if options.height > 0:
            _check_result(
                "设置海康相机高度",
                camera.MV_CC_SetIntValue("Height", int(options.height)),
            )
        if options.fps > 0:
            _check_result(
                "设置海康相机帧率",
                camera.MV_CC_SetFloatValue("AcquisitionFrameRate", float(options.fps)),
            )
        stream.control_notes = _apply_exposure_and_gain(camera, options)
        _check_result("启动海康相机取流", camera.MV_CC_StartGrabbing())
        stream._grabbing = True

        # Warm up only as requested; retain the newest frame for the first read.
        for _ in range(max(0, int(options.warmup_frames))):
            ok, image = stream.read()
            if ok and image is not None:
                stream._pending_frame = image
        if options.warmup_frames > 0 and stream._pending_frame is None:
            raise CameraError(
                f"海康相机已打开，但在 {options.timeout_ms} ms 内未收到画面。\n{_HIK_HINTS}"
            )
        return stream
    except CameraError:
        if stream is not None:
            stream.release()
        else:
            session.close()
        raise
    except Exception as exc:
        if stream is not None:
            stream.release()
        else:
            session.close()
        raise CameraError(f"打开海康相机失败：{exc}\n{_HIK_HINTS}") from exc
