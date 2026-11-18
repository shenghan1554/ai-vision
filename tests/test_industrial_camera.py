"""Mock-only tests for industrial cameras; no SDK device access is performed."""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from yolo_app import hikvision
from yolo_app.camera import CameraError, CameraOptions


class _GigEInfo(ctypes.Structure):
    _fields_ = [
        ("chModelName", ctypes.c_ubyte * 32),
        ("chSerialNumber", ctypes.c_ubyte * 16),
        ("chUserDefinedName", ctypes.c_ubyte * 16),
        ("nCurrentIp", ctypes.c_uint),
    ]


class _UsbInfo(ctypes.Structure):
    _fields_ = [
        ("chModelName", ctypes.c_ubyte * 64),
        ("chSerialNumber", ctypes.c_ubyte * 64),
        ("chUserDefinedName", ctypes.c_ubyte * 64),
    ]


class _SpecialInfo(ctypes.Union):
    _fields_ = [("stGigEInfo", _GigEInfo), ("stUsb3VInfo", _UsbInfo)]


class _DeviceInfo(ctypes.Structure):
    _fields_ = [("nTLayerType", ctypes.c_uint), ("SpecialInfo", _SpecialInfo)]


class _DeviceInfoList(ctypes.Structure):
    _fields_ = [
        ("nDeviceNum", ctypes.c_uint),
        ("pDeviceInfo", ctypes.POINTER(_DeviceInfo) * 4),
    ]


class _FrameInfo(ctypes.Structure):
    _fields_ = [
        ("nWidth", ctypes.c_ushort),
        ("nHeight", ctypes.c_ushort),
        ("enPixelType", ctypes.c_int),
        ("nFrameLen", ctypes.c_uint),
        ("nExtendWidth", ctypes.c_uint),
        ("nExtendHeight", ctypes.c_uint),
        ("nFrameLenEx", ctypes.c_uint64),
    ]


class _FrameOut(ctypes.Structure):
    _fields_ = [
        ("pBufAddr", ctypes.POINTER(ctypes.c_ubyte)),
        ("stFrameInfo", _FrameInfo),
    ]


class _ConvertParam(ctypes.Structure):
    _fields_ = [
        ("nWidth", ctypes.c_uint),
        ("nHeight", ctypes.c_uint),
        ("enSrcPixelType", ctypes.c_int),
        ("pSrcData", ctypes.POINTER(ctypes.c_ubyte)),
        ("nSrcDataLen", ctypes.c_uint),
        ("enDstPixelType", ctypes.c_int),
        ("pDstBuffer", ctypes.POINTER(ctypes.c_ubyte)),
        ("nDstLen", ctypes.c_uint),
        ("nDstBufferSize", ctypes.c_uint),
    ]


class _FloatValue(ctypes.Structure):
    _fields_ = [("fCurValue", ctypes.c_float), ("fMax", ctypes.c_float), ("fMin", ctypes.c_float)]


def _set_text(target: ctypes.Array, text: str) -> None:
    encoded = text.encode("ascii")
    for index, value in enumerate(encoded):
        target[index] = value


def _make_device(serial: str, layer: int = 1, model: str = "MV-TEST") -> _DeviceInfo:
    info = _DeviceInfo()
    info.nTLayerType = layer
    special = info.SpecialInfo.stGigEInfo if layer == 1 else info.SpecialInfo.stUsb3VInfo
    _set_text(special.chModelName, model)
    _set_text(special.chSerialNumber, serial)
    if layer == 1:
        # Deliberately use the high bit to exercise unsigned IP formatting.
        info.SpecialInfo.stGigEInfo.nCurrentIp = 0xC0A80164
    return info


def _fake_mvs(
    serials: list[str] | None = None,
    *,
    pixel_type: int = 0x02180014,
    source_bytes: bytes = b"\x01\x02\x03\x04\x05\x06",
    width: int = 2,
    height: int = 1,
):
    serials = ["GIGE-A"] if serials is None else serials
    infos = [_make_device(serial, 1 if index == 0 else 4) for index, serial in enumerate(serials)]
    pointers = [ctypes.pointer(info) for info in infos]
    state = {
        "pixel_type": pixel_type,
        "fps": 30.0,
        "get_result": 0,
        "open_result": 0,
        "events": [],
    }
    source = (ctypes.c_ubyte * len(source_bytes))(*source_bytes)

    camera = MagicMock(name="MvCameraInstance")
    camera.MV_CC_CreateHandle.return_value = 0
    camera.MV_CC_OpenDevice.side_effect = lambda *_args: state["open_result"]
    camera.MV_CC_GetOptimalPacketSize.return_value = 1500
    for name, event in (
        ("MV_CC_SetIntValue", None),
        ("MV_CC_SetEnumValue", None),
        ("MV_CC_SetFloatValue", None),
        ("MV_CC_StartGrabbing", None),
    ):
        getattr(camera, name).return_value = 0
    camera.MV_CC_StopGrabbing.side_effect = lambda: state["events"].append("stop") or 0
    camera.MV_CC_CloseDevice.side_effect = lambda: state["events"].append("close") or 0
    camera.MV_CC_DestroyHandle.side_effect = lambda: state["events"].append("destroy") or 0
    camera.MV_CC_FreeImageBuffer.side_effect = lambda _frame: state["events"].append("free") or 0

    def create_handle(info):
        transport = hikvision._transport(sdk, info)
        assert transport is not None
        state["created_serial"] = hikvision._decode_text(transport[1].chSerialNumber)
        return 0

    camera.MV_CC_CreateHandle.side_effect = create_handle

    def enum_devices(_layers, output):
        output.nDeviceNum = len(infos)
        for index, pointer in enumerate(pointers):
            output.pDeviceInfo[index] = pointer
        return 0

    def get_image(frame, _timeout):
        frame.pBufAddr = ctypes.cast(source, ctypes.POINTER(ctypes.c_ubyte))
        frame.stFrameInfo.nWidth = width
        frame.stFrameInfo.nHeight = height
        frame.stFrameInfo.enPixelType = state["pixel_type"]
        frame.stFrameInfo.nFrameLen = len(source_bytes)
        return state["get_result"]

    camera.MV_CC_GetImageBuffer.side_effect = get_image

    def convert(params):
        raw = np.ctypeslib.as_array(params.pSrcData, shape=(params.nSrcDataLen,)).copy()
        if params.enSrcPixelType == 0x01080001:  # Mono8
            converted = np.repeat(raw, 3)
        elif params.enSrcPixelType == 0x02180014:  # RGB8
            converted = raw.reshape(-1, 3)[:, ::-1].copy().reshape(-1)
        elif params.enSrcPixelType == 0x02180015:  # BGR8
            converted = raw
        else:  # Bayer and other SDK-supported formats: deterministic fake output.
            converted = np.arange(params.nWidth * params.nHeight * 3, dtype=np.uint8)
        converted = np.ascontiguousarray(converted, dtype=np.uint8)
        ctypes.memmove(
            params.pDstBuffer,
            converted.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            converted.size,
        )
        params.nDstLen = converted.size
        return 0

    camera.MV_CC_ConvertPixelTypeEx.side_effect = convert
    camera.MV_CC_ConvertPixelType.side_effect = convert

    def get_float(_key, output):
        output.fCurValue = state["fps"]
        return 0

    camera.MV_CC_GetFloatValue.side_effect = get_float

    camera_type = MagicMock(name="MvCameraType", return_value=camera)
    camera_type.MV_CC_Initialize.return_value = 0
    camera_type.MV_CC_Finalize.side_effect = lambda: state["events"].append("finalize") or 0
    camera_type.MV_CC_EnumDevices.side_effect = enum_devices
    sdk = SimpleNamespace(
        MvCamera=camera_type,
        MV_CC_DEVICE_INFO_LIST=_DeviceInfoList,
        MV_CC_DEVICE_INFO=_DeviceInfo,
        MV_FRAME_OUT=_FrameOut,
        MV_CC_PIXEL_CONVERT_PARAM_EX=_ConvertParam,
        MV_CC_PIXEL_CONVERT_PARAM=_ConvertParam,
        MVCC_FLOATVALUE=_FloatValue,
        MV_GIGE_DEVICE=1,
        MV_USB_DEVICE=4,
        MV_GENTL_GIGE_DEVICE=0x40,
        MV_ACCESS_Exclusive=1,
        MV_TRIGGER_MODE_OFF=0,
        MV_E_NODATA=0x80000007,
        MV_E_GC_TIMEOUT=0x80000107,
        PixelType_Gvsp_BGR8_Packed=0x02180015,
    )
    return sdk, camera, state


class HikvisionTests(unittest.TestCase):
    def test_sdk_path_is_scoped_to_dynamic_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sdk_dir = Path(directory)
            (sdk_dir / "MvCameraControl_class.py").write_text("# mock", encoding="utf-8")
            expected = SimpleNamespace(MvCamera=object())
            original_path = list(sys.path)

            def fake_import(name):
                self.assertEqual(name, "MvCameraControl_class")
                self.assertIn(str(sdk_dir), sys.path)
                return expected

            with patch.object(hikvision, "_candidate_sdk_paths", return_value=[sdk_dir]), patch.object(
                hikvision.importlib, "import_module", side_effect=fake_import
            ), patch.object(hikvision, "_prepare_windows_sdk_runtime", return_value=[]):
                self.assertIs(hikvision._load_sdk(), expected)
            self.assertEqual(sys.path, original_path)
            self.assertIn(hikvision._DEFAULT_MVS_IMPORT, hikvision._candidate_sdk_paths())

    def test_missing_sdk_has_actionable_error(self) -> None:
        with patch.object(hikvision, "_candidate_sdk_paths", return_value=[]), patch.object(
            hikvision.importlib, "import_module", side_effect=ModuleNotFoundError("missing")
        ), patch.object(hikvision, "_prepare_windows_sdk_runtime", return_value=[]):
            with self.assertRaisesRegex(CameraError, "MVS_SDK_PYTHON_PATH"):
                hikvision._load_sdk()

    def test_development_candidate_is_not_duplicated(self) -> None:
        with patch.dict(
            os.environ,
            {"MVCAM_COMMON_RUNENV": r"C:\Program Files (x86)\MVS\Development"},
        ):
            paths = hikvision._candidate_sdk_paths()
        self.assertFalse(
            any("Development\\\\Development" in str(path) for path in paths)
        )
        self.assertIn(
            Path(r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"),
            paths,
        )

    def test_candidate_posix_path_has_no_duplicate_development(self) -> None:
        with patch.dict(
            os.environ,
            {"MVCAM_COMMON_RUNENV": r"C:\Program Files (x86)\MVS\Development"},
        ):
            paths = hikvision._candidate_sdk_paths()
        self.assertFalse(
            any("development/development" in path.as_posix().lower() for path in paths)
        )

    def test_runtime_preload_registers_and_retains_handles(self) -> None:
        old_runtime = hikvision._PREPARED_RUNTIME
        old_handles = list(hikvision._RUNTIME_HANDLES)
        old_errors = list(hikvision._RUNTIME_LOAD_ERRORS)
        old_path = os.environ.get("PATH", "")
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "Win64_x64"
            runtime.mkdir()
            library = runtime / "MvCameraControl.dll"
            library.write_bytes(b"mock")
            try:
                hikvision._PREPARED_RUNTIME = None
                hikvision._RUNTIME_HANDLES = []
                hikvision._RUNTIME_LOAD_ERRORS = []
                with patch.object(
                    hikvision, "_candidate_runtime_directories", return_value=[runtime]
                ), patch.object(
                    hikvision.os, "add_dll_directory", return_value=object()
                ) as add_directory, patch.object(
                    hikvision.ctypes, "WinDLL", return_value=object()
                ) as load_library:
                    result = hikvision._prepare_windows_sdk_runtime()
                self.assertEqual(result, [runtime])
                load_library.assert_called_once_with(str(library), winmode=0)
                self.assertIn(
                    unittest.mock.call(str(runtime)), add_directory.call_args_list
                )
                self.assertGreaterEqual(len(hikvision._RUNTIME_HANDLES), 2)
                self.assertIn(str(runtime), os.environ.get("PATH", ""))
            finally:
                hikvision._PREPARED_RUNTIME = old_runtime
                hikvision._RUNTIME_HANDLES = old_handles
                hikvision._RUNTIME_LOAD_ERRORS = old_errors
                os.environ["PATH"] = old_path

    def test_no_devices_returns_empty_and_finalizes(self) -> None:
        sdk, _camera, state = _fake_mvs([])
        with patch.object(hikvision, "_load_sdk", return_value=sdk):
            self.assertEqual(hikvision.enumerate_hikvision_devices(), [])
        sdk.MvCamera.MV_CC_Initialize.assert_called_once_with()
        sdk.MvCamera.MV_CC_Finalize.assert_called_once_with()
        self.assertEqual(state["events"], ["finalize"])

    def test_serial_selection_bgr_read_fps_and_release(self) -> None:
        sdk, camera, state = _fake_mvs(["GIGE-A", "USB-B"])
        with patch.object(hikvision, "_load_sdk", return_value=sdk):
            stream = hikvision.open_hikvision_camera(
                "hik://USB-B", CameraOptions(warmup_frames=0, timeout_ms=1234)
            )
            self.assertEqual(state["created_serial"], "USB-B")
            ok, image = stream.read()
            self.assertTrue(ok)
            np.testing.assert_array_equal(image.reshape(-1), [3, 2, 1, 6, 5, 4])
            self.assertEqual(image.shape, (1, 2, 3))
            self.assertTrue(image.flags.c_contiguous)
            camera.MV_CC_GetImageBuffer.assert_called_with(unittest.mock.ANY, 1234)
            camera.MV_CC_FreeImageBuffer.assert_called_once()
            self.assertEqual(stream.get_fps(), 30.0)
            stream.release()
            stream.release()
        self.assertEqual(
            [event for event in state["events"] if event != "free"],
            ["stop", "close", "destroy", "finalize"],
        )

    def test_gige_optimal_packet_and_unsigned_ip(self) -> None:
        sdk, camera, _state = _fake_mvs(["GIGE-A"])
        with patch.object(hikvision, "_load_sdk", return_value=sdk):
            devices = hikvision.enumerate_hikvision_devices()
            self.assertIn("192.168.1.100", devices[0].detail)
            stream = hikvision.open_hikvision_camera("hik://0", CameraOptions(warmup_frames=0))
            stream.release()
        camera.MV_CC_SetIntValue.assert_any_call("GevSCPSPacketSize", 1500)

    def test_common_pixel_formats_use_sdk_conversion(self) -> None:
        cases = (
            (0x01080001, b"\x00\x7f", [0, 0, 0, 127, 127, 127]),
            (0x02180014, b"\x01\x02\x03\x04\x05\x06", [3, 2, 1, 6, 5, 4]),
            (0x02180015, b"\x01\x02\x03\x04\x05\x06", [1, 2, 3, 4, 5, 6]),
            (0x01080009, b"\x01\x02", [0, 1, 2, 3, 4, 5]),
        )
        for pixel_type, raw, expected in cases:
            with self.subTest(pixel_type=hex(pixel_type)):
                sdk, camera, _state = _fake_mvs(pixel_type=pixel_type, source_bytes=raw)
                session = hikvision._SdkSession(sdk)
                session.open()
                stream = hikvision.HikvisionStream(
                    camera, sdk, session, "mock", CameraOptions(warmup_frames=0)
                )
                ok, image = stream.read()
                self.assertTrue(ok)
                np.testing.assert_array_equal(image.reshape(-1), expected)
                camera.MV_CC_ConvertPixelTypeEx.assert_called_once()
                camera.MV_CC_FreeImageBuffer.assert_called_once()
                stream.release()

    def test_timeout_and_open_error_cleanup(self) -> None:
        sdk, camera, state = _fake_mvs()
        state["get_result"] = -2147483641  # signed representation of 0x80000007
        session = hikvision._SdkSession(sdk)
        session.open()
        stream = hikvision.HikvisionStream(camera, sdk, session, "mock", CameraOptions())
        self.assertEqual(stream.read(), (False, None))
        camera.MV_CC_FreeImageBuffer.assert_not_called()
        stream.release()

        sdk, camera, state = _fake_mvs()
        state["open_result"] = 0x80000203
        with patch.object(hikvision, "_load_sdk", return_value=sdk):
            with self.assertRaises(CameraError) as caught:
                hikvision.open_hikvision_camera("hik://0", CameraOptions(warmup_frames=0))
        message = str(caught.exception)
        for text in ("0x80000203", "占用", "同网段", "授权"):
            self.assertIn(text, message)
        camera.MV_CC_DestroyHandle.assert_called_once_with()
        sdk.MvCamera.MV_CC_Finalize.assert_called_once_with()

    def test_exposure_and_gain_are_applied(self) -> None:
        sdk, camera, _state = _fake_mvs(["GIGE-A"])
        with patch.object(hikvision, "_load_sdk", return_value=sdk):
            stream = hikvision.open_hikvision_camera(
                "hik://0",
                CameraOptions(warmup_frames=0, exposure_us=2000.0, gain=2.0),
            )
            stream.release()
        camera.MV_CC_SetFloatValue.assert_any_call("ExposureTime", 2000.0)
        camera.MV_CC_SetFloatValue.assert_any_call("Gain", 2.0)

    def test_missing_serial_is_clear(self) -> None:
        sdk, _camera, _state = _fake_mvs(["KNOWN"])
        with patch.object(hikvision, "_load_sdk", return_value=sdk):
            with self.assertRaisesRegex(CameraError, "当前枚举设备.*KNOWN"):
                hikvision.open_hikvision_camera("hik://UNKNOWN", CameraOptions(warmup_frames=0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
