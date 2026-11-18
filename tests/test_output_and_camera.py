"""Regression tests for output safety and UVC camera fallback behavior."""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from yolo_app.camera import (
    CameraError,
    LatestFrameStream,
    CameraOptions,
    camera_output_stem,
    is_camera_source,
    normalise_camera_source,
    open_usb_camera,
    _open_capture,
)
from yolo_app.output import (
    atomic_write_bytes,
    create_run_output_directory,
    ensure_writable_directory,
    timestamped_result_path,
)
from yolo_app.processor import process_source


class FakeCapture:
    def __init__(self, opened: bool, frames: list | None = None) -> None:
        self.opened = opened
        self.frames = list(frames or [])
        self.released = False
        self.settings = []

    def isOpened(self) -> bool:
        return self.opened

    def set(self, key, value) -> bool:
        self.settings.append((key, value))
        return True

    def get(self, _key) -> float:
        return 30.0

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def release(self) -> None:
        self.released = True


class OutputSafetyTests(unittest.TestCase):
    def test_run_directories_are_unique_and_writable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = create_run_output_directory(directory)
            second = create_run_output_directory(directory)
            self.assertNotEqual(first, second)
            self.assertTrue(first.name.startswith("run_"))
            self.assertEqual(ensure_writable_directory(first), first)

    def test_timestamp_names_and_collision_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "yolo_app.output.datetime"
        ) as clock:
            clock.now.return_value.strftime.return_value = "20261116_131500_123456"
            first = timestamped_result_path(directory, ".jpg")
            first.touch()
            second = timestamped_result_path(directory, ".jpg")
        self.assertEqual(first.name, "20261116_131500_123.jpg")
        self.assertEqual(second.name, "20261116_131500_123_001.jpg")

    def test_permission_error_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("pathlib.Path.open", side_effect=PermissionError(13, "denied")):
                with self.assertRaisesRegex(PermissionError, "受控文件夹访问权限"):
                    ensure_writable_directory(directory)

    def test_atomic_failure_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.jpg"
            target.write_bytes(b"original")
            with patch("yolo_app.output.os.replace", side_effect=PermissionError(13, "denied")):
                with self.assertRaisesRegex(PermissionError, "同名文件"):
                    atomic_write_bytes(target, b"new")
            self.assertEqual(target.read_bytes(), b"original")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


class UsbCameraTests(unittest.TestCase):
    def test_source_normalisation(self) -> None:
        self.assertEqual(normalise_camera_source(0), "usb://0")
        self.assertEqual(normalise_camera_source("1"), "usb://1")
        self.assertEqual(normalise_camera_source("hikvision://ABC"), "hik://ABC")
        self.assertFalse(is_camera_source("genicam://0"))
        self.assertFalse(is_camera_source("video.mp4"))
        self.assertEqual(camera_output_stem("hik://AB C"), "hik_AB_C")

    def test_open_capture_does_not_pass_timeout_property_53(self) -> None:
        capture = FakeCapture(True)
        with patch("yolo_app.camera.cv2.VideoCapture", return_value=capture) as constructor:
            _open_capture(0, 2, 3000)
        constructor.assert_called_once_with(0, 2)

    def test_usb_backend_fallback_and_pending_frame(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        unavailable = [FakeCapture(False), FakeCapture(False)]
        available = FakeCapture(True, [frame])
        with patch(
            "yolo_app.camera._open_capture",
            side_effect=[*unavailable, available],
        ), patch("yolo_app.camera.time.sleep"):
            stream = open_usb_camera(
                0, CameraOptions(warmup_frames=0, mjpg=False)
            )
        self.assertIn("MSMF", stream.description)
        ok, received = stream.read()
        self.assertTrue(ok)
        self.assertIs(received, frame)
        stream.release()
        self.assertTrue(all(capture.released for capture in unavailable))
        self.assertTrue(available.released)

    def test_mjpg_failure_falls_back_to_driver_defaults(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rejected = [FakeCapture(True), FakeCapture(True)]
        compatible = FakeCapture(True, [frame])
        with patch(
            "yolo_app.camera._open_capture",
            side_effect=[*rejected, compatible],
        ), patch("yolo_app.camera.time.sleep"):
            stream = open_usb_camera(
                0,
                CameraOptions(
                    backend="dshow",
                    warmup_frames=0,
                    mjpg=True,
                    width=1920,
                    height=1080,
                ),
            )
        self.assertIn("兼容模式", stream.description)
        self.assertFalse(
            any(key == cv2.CAP_PROP_FOURCC for key, _value in compatible.settings),
            "fallback must not force MJPG FourCC",
        )
        self.assertFalse(
            any(key == cv2.CAP_PROP_FRAME_WIDTH for key, _value in compatible.settings),
            "fallback must not force a custom width",
        )
        stream.release()

    def test_opened_camera_without_frames_is_rejected(self) -> None:
        with patch("yolo_app.camera._open_capture", side_effect=lambda *_args: FakeCapture(True)), patch(
            "yolo_app.camera.time.sleep"
        ):
            with self.assertRaisesRegex(CameraError, "读取不到画面"):
                open_usb_camera(0, CameraOptions(warmup_frames=0))

    def test_output_preflight_happens_before_camera_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_file = Path(directory) / "not-a-directory"
            output_file.write_text("x", encoding="utf-8")
            with patch("yolo_app.processor.open_camera") as opener:
                with self.assertRaisesRegex(ValueError, "不是目录"):
                    list(process_source(object(), 0, output_file))
                opener.assert_not_called()

    def test_transient_camera_failure_is_tolerated(self) -> None:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        class Engine:
            model_path = "fake.pt"

            def predict(self, image):
                return type("Result", (), {"annotated": image, "detections": [], "inference_ms": 1.0})()

        class Stream:
            description = "fake USB"
            released = False

            def __init__(self):
                self.reads = 0

            def read(self):
                self.reads += 1
                return (False, None) if self.reads == 1 else (True, frame)

            def get_fps(self):
                return 30.0

            def release(self):
                self.released = True

        stream = Stream()
        with tempfile.TemporaryDirectory() as directory:
            with patch("yolo_app.processor.open_camera", return_value=stream):
                events = list(
                    process_source(
                        Engine(),
                        0,
                        directory,
                    )
                )
            photo = next(event for event in events if event.kind == "photo")
            self.assertTrue(Path(photo.saved_path).is_file())
        self.assertEqual(stream.reads, 2)
        self.assertTrue(stream.released)
        self.assertEqual(events[-1].kind, "completed")

    def test_camera_single_shot_saves_one_photo_and_json(self) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        class Engine:
            model_path = "fake.pt"

            def predict(self, image):
                return type("Result", (), {"annotated": image, "detections": [], "inference_ms": 1.0})()

        class Stream:
            description = "GE2000C"
            control_notes = ["曝光 5000 µs 已设置"]

            def read(self):
                return True, frame

            def get_fps(self):
                return 30.0

            def release(self):
                self.released = True

        stream = Stream()
        with tempfile.TemporaryDirectory() as directory, patch(
            "yolo_app.processor.open_camera", return_value=stream
        ):
            events = list(
                process_source(Engine(), 0, directory, save_json=True)
            )
            # Exactly one photo is taken: no continuous live frames.
            self.assertEqual(len([e for e in events if e.kind == "photo"]), 1)
            self.assertFalse(any(e.kind == "frame" for e in events))
            photos = list(Path(directory).glob("*.jpg"))
            self.assertEqual(len(photos), 1)
            self.assertRegex(photos[0].name, r"^\d{8}_\d{6}_\d{3}(?:_\d{3})?\.jpg$")
            self.assertTrue(photos[0].with_suffix(".json").is_file())
            self.assertEqual(list(Path(directory).glob("*.jsonl")), [])
        self.assertTrue(stream.released)
        self.assertEqual(events[-1].kind, "completed")

class LatestFrameStreamTests(unittest.TestCase):
    def test_photo_uses_cached_frame_without_waiting(self) -> None:
        first = np.full((8, 8, 3), 10, dtype=np.uint8)
        second = np.full((8, 8, 3), 20, dtype=np.uint8)

        class Base:
            description = "fake"
            control_notes = []

            def __init__(self):
                self.frames = [first, second]
                self.released = False

            def read(self):
                if self.frames:
                    time.sleep(0.03)
                    return True, self.frames.pop(0)
                return False, None

            def get_fps(self):
                return 30.0

            def release(self):
                self.released = True

        base = Base()
        stream = LatestFrameStream(base, timeout_ms=200)
        try:
            self.assertTrue(stream.read_latest()[0])
            started = time.perf_counter()
            ok, frame = stream.read_latest()
            elapsed = time.perf_counter() - started
            self.assertTrue(ok)
            self.assertIsNotNone(frame)
            self.assertLess(elapsed, 0.15, "photo must not wait for next camera frame")
            self.assertTrue(stream.read_next()[0])
            # A later live read consumes a newer sequence, not the same frame.
            ok, newer = stream.read_next()
            self.assertTrue(ok)
            self.assertIsNotNone(newer)
        finally:
            stream.release()
        self.assertTrue(base.released)


class CameraModeTests(unittest.TestCase):
    class Engine:
        model_path = "fake.pt"

        def predict(self, image):
            return type(
                "Result", (), {"annotated": image, "detections": [], "inference_ms": 1.0}
            )()

    class Stream:
        description = "GE2000C"
        control_notes = []

        def __init__(self, frames=None, endless=False):
            self.frames = list(frames or [])
            self.endless = endless
            self.released = False
            self.reads = 0

        def read(self):
            self.reads += 1
            if self.frames:
                return True, self.frames.pop(0)
            if self.endless:
                return True, np.zeros((32, 32, 3), dtype=np.uint8)
            return False, None

        def get_fps(self):
            return 30.0

        def release(self):
            self.released = True

    def test_live_mode_streams_until_cancelled_and_saves_last_frame(self) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        cancel = threading.Event()

        class AutoCancelEngine(self.Engine):
            def predict(self, image):
                result = super().predict(image)
                if self.calls >= 2:
                    cancel.set()
                self.calls += 1
                return result

        engine = AutoCancelEngine()
        engine.calls = 0
        stream = self.Stream([frame, frame, frame], endless=True)
        with tempfile.TemporaryDirectory() as directory, patch(
            "yolo_app.processor.open_camera", return_value=stream
        ):
            events = list(
                process_source(
                    engine, "hik://SERIAL", directory, save_json=True,
                    camera_options=CameraOptions(warmup_frames=0),
                    mode="live", cancel_event=cancel,
                )
            )
            frames = [e for e in events if e.kind == "frame"]
            self.assertGreaterEqual(len(frames), 2)
            finished = next(e for e in events if e.kind == "camera_finished")
            self.assertTrue(Path(finished.saved_path).is_file())
            self.assertEqual(len(list(Path(directory).glob("*.jpg"))), 1)
        self.assertTrue(stream.released)
        self.assertEqual(events[-1].kind, "completed")

    def test_photo_mode_reuses_open_frame_and_takes_one_shot(self) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        stream = self.Stream([frame])
        with tempfile.TemporaryDirectory() as directory, patch(
            "yolo_app.processor.open_camera", return_value=stream
        ):
            events = list(
                process_source(
                    self.Engine(), "usb://0", directory,
                    camera_options=CameraOptions(warmup_frames=0), mode="photo",
                )
            )
            self.assertEqual(len([e for e in events if e.kind == "photo"]), 1)
            self.assertFalse(any(e.kind == "frame" for e in events))
            # Only the frame supplied by open() is consumed: no extra warm-up reads.
            self.assertEqual(stream.reads, 1)
        self.assertTrue(stream.released)

    def test_invalid_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "不支持的相机模式"):
                list(process_source(self.Engine(), "usb://0", directory, mode="bogus"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
