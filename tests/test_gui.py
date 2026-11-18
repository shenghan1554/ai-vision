"""Modern GUI integration tests with real Tk, image inference and a fake camera."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from yolo_app.camera import CameraOptions
from yolo_app.gui import (
    YOLOv8DesktopApp,
    _expand_saved_path,
    _make_path_portable,
)
from yolo_app.paths import app_dir, resource_path


ROOT = Path(__file__).resolve().parent.parent


class PortableSettingsTests(unittest.TestCase):
    def test_app_path_round_trip(self) -> None:
        original = app_dir() / "models" / "yolov8n.pt"
        portable = _make_path_portable(original)
        self.assertTrue(portable.startswith("{app}/"))
        self.assertEqual(Path(_expand_saved_path(portable)), original)

    def test_external_path_stays_absolute(self) -> None:
        external = Path(tempfile.gettempdir()).resolve() / "external.pt"
        if app_dir() not in external.parents:
            self.assertEqual(_make_path_portable(external), str(external))


class CameraConnectionTests(unittest.TestCase):
    def test_reuse_switch_and_close_release(self) -> None:
        class Stream:
            def __init__(self, name):
                self.description = name
                self.control_notes = []
                self.released = False

            def read(self):
                return True, np.zeros((8, 8, 3), dtype=np.uint8)

            def get_fps(self):
                return 30.0

            def release(self):
                self.released = True

        first = Stream("usb")
        second = Stream("hik")
        app = YOLOv8DesktopApp(root=tk.Tk())
        app.root.withdraw()
        try:
            with patch("yolo_app.gui.open_camera", side_effect=[first, second]) as opener:
                options = CameraOptions(warmup_frames=0)
                self.assertIs(app._get_or_open_camera_stream("usb://0", options), first)
                self.assertIs(app._get_or_open_camera_stream("usb://0", options), first)
                self.assertEqual(opener.call_count, 1)
                self.assertIs(app._get_or_open_camera_stream("hik://055", options), second)
                self.assertTrue(first.released)
            self.assertFalse(second.released)
        finally:
            app._on_close()
        self.assertTrue(second.released)


class CameraSelectionTests(unittest.TestCase):
    """Regression: selecting a camera must leave exactly one button selected."""

    def test_capture_settings_removed_and_branding(self) -> None:
        app = YOLOv8DesktopApp(root=tk.Tk())
        app.root.withdraw()
        tmp = tempfile.TemporaryDirectory()
        app._settings_path = Path(tmp.name) / "settings.json"
        try:
            # 采集设置面板及其参数已删除
            for attr in (
                "_open_camera_settings",
                "camera_settings_window",
                "camera_exposure_var",
                "camera_gain_var",
                "camera_backend_var",
                "camera_mjpg_var",
            ):
                self.assertFalse(hasattr(app, attr), f"{attr} should be removed")
            # 产品名
            self.assertEqual(app.root.title(), "AI Vision")
        finally:
            app._on_close()
            tmp.cleanup()

    def test_only_one_camera_shows_selected(self) -> None:
        from yolo_app.camera import CameraDevice

        class Button:
            def __init__(self):
                self.text = "选择此设备"

            def winfo_exists(self):
                return True

            def configure(self, **kwargs):
                if "text" in kwargs:
                    self.text = kwargs["text"]

        app = YOLOv8DesktopApp(root=tk.Tk())
        app.root.withdraw()
        tmp = tempfile.TemporaryDirectory()
        app._settings_path = Path(tmp.name) / "settings.json"
        try:
            devices = [
                CameraDevice(source="usb://0", name="USB Cam", kind="usb"),
                CameraDevice(source="hik://A", name="HIK-A", kind="hikvision"),
                CameraDevice(source="hik://B", name="HIK-B", kind="hikvision"),
            ]
            app._camera_devices = devices
            app._camera_device_buttons = {d.source: Button() for d in devices}
            app._available_camera_sources = {d.source for d in devices}

            app._select_camera_device(devices[0])
            app._select_camera_device(devices[1])
            app._select_camera_device(devices[2])

            selected = [
                source
                for source, button in app._camera_device_buttons.items()
                if "已选择" in button.text
            ]
            self.assertEqual(
                selected,
                ["hik://B"],
                "exactly one camera must be marked selected",
            )
            self.assertEqual(app.source_var.get(), "hik://B")
        finally:
            app._on_close()
            tmp.cleanup()


class GuiIntegrationTests(unittest.TestCase):
    def test_gui_photo_run_uses_camera_and_updates_preview(self) -> None:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        class Stream:
            description = "ov9734_techfront_camera"
            control_notes = []

            def read(self):
                return True, frame

            def get_fps(self):
                return 30.0

            def release(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            errors: list[str] = []
            app = YOLOv8DesktopApp(root=tk.Tk())
            app.root.withdraw()
            app._settings_path = Path(directory) / "settings.json"
            app.model_var.set(str(ROOT / "models" / "yolov8n.onnx"))
            app.source_var.set("usb://0")
            app.output_var.set(str(output))
            app.device_var.set("cpu")
            app.save_json_var.set(True)
            app.unique_output_var.set(False)
            try:
                with patch(
                    "yolo_app.processor.open_camera", return_value=Stream()
                ), patch(
                    "yolo_app.gui.open_camera", return_value=Stream()
                ), patch(
                    "yolo_app.gui.messagebox.showerror",
                    side_effect=lambda _title, message, **_kwargs: errors.append(
                        str(message)
                    ),
                ):
                    app.start_processing()
                    deadline = time.monotonic() + 60
                    while app._running and time.monotonic() < deadline:
                        app.root.update()
                        time.sleep(0.01)
                    app.root.update()

                self.assertFalse(app._running, app.log_text.get("1.0", "end"))
                self.assertFalse(app._run_failed, errors)
                self.assertEqual(errors, [])
                self.assertIsNotNone(app._preview_source)
                images = list(output.glob("*.jpg"))
                self.assertEqual(len(images), 1)
                self.assertTrue(images[0].with_suffix(".json").is_file())
                log = app.log_text.get("1.0", "end")
                self.assertIn("拍照检测", log)
                settings = json.loads(app._settings_path.read_text("utf-8"))
                self.assertTrue(str(settings["model"]).startswith("{app}/"))
                self.assertEqual(settings["source"], "usb://0")
            finally:
                if app._cancel_event is not None:
                    app._cancel_event.set()
                app._on_close()

    def test_gui_camera_photo_saves_single_image_in_run_directory(self) -> None:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        class FakeEngine:
            model_path = "fake.pt"
            model_description = "fake camera engine"

            def __init__(self, **_kwargs):
                pass

            def predict(self, image):
                return SimpleNamespace(
                    annotated=image, detections=[], inference_ms=1.0
                )

        class FakeStream:
            description = "Fake USB camera"

            def __init__(self):
                self.released = False

            def read(self):
                time.sleep(0.02)
                return True, frame

            def get_fps(self):
                return 25.0

            def release(self):
                self.released = True

        stream = FakeStream()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "base-output"
            errors: list[str] = []
            app = YOLOv8DesktopApp(root=tk.Tk())
            app.root.withdraw()
            app._settings_path = Path(directory) / "settings.json"
            app.model_var.set(str(ROOT / "models" / "yolov8n.onnx"))
            app.source_var.set("usb://0")
            app.output_var.set(str(output))
            app.unique_output_var.set(True)
            app.save_json_var.set(True)
            try:
                with patch("yolo_app.gui.YOLOv8Engine", FakeEngine), patch(
                    "yolo_app.processor.open_camera", return_value=stream
                ), patch(
                    "yolo_app.gui.open_camera", return_value=stream
                ), patch(
                    "yolo_app.gui.messagebox.showerror",
                    side_effect=lambda _title, message, **_kwargs: errors.append(
                        str(message)
                    ),
                ):
                    app.start_processing()
                    app.root.after(600, app.stop_processing)
                    deadline = time.monotonic() + 15
                    while app._running and time.monotonic() < deadline:
                        app.root.update()
                        time.sleep(0.01)
                    app.root.update()

                self.assertFalse(app._running, app.log_text.get("1.0", "end"))
                self.assertEqual(errors, [])
                self.assertFalse(
                    stream.released,
                    "persistent stream should stay open between photo runs",
                )
                self.assertIsNotNone(app.last_output_dir)
                run_dir = app.last_output_dir
                self.assertNotEqual(run_dir, output)
                self.assertTrue(run_dir.name.startswith("run_"))
                photos = list(run_dir.glob("*.jpg"))
                self.assertEqual(len(photos), 1)
                self.assertTrue(photos[0].with_suffix(".json").is_file())
                log = app.log_text.get("1.0", "end")
                self.assertIn("照片", log)
                settings = json.loads(app._settings_path.read_text("utf-8"))
                self.assertEqual(Path(_expand_saved_path(settings["output"])), output)
                self.assertNotIn("run_", settings["output"])
                self.assertTrue(settings["unique_output"])
            finally:
                if app._cancel_event is not None:
                    app._cancel_event.set()
                app._on_close()
            self.assertTrue(stream.released, "closing the app must release the camera")


if __name__ == "__main__":
    unittest.main(verbosity=2)
