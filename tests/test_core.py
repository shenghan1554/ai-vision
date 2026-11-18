"""Integration tests using the bundled YOLOv8n PT and ONNX models."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from yolo_app.engine import YOLOv8Engine, load_names
from yolo_app.processor import process_source, read_image


ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "samples" / "bus.jpg"
PT_MODEL = ROOT / "models" / "yolov8n.pt"
ONNX_MODEL = ROOT / "models" / "yolov8n.onnx"


class ModelValidationTests(unittest.TestCase):
    def test_missing_model_has_clear_error(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "模型文件不存在"):
            YOLOv8Engine(ROOT / "models" / "missing.pt")

    def test_unsupported_model_extension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.txt"
            path.write_text("not a model", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "仅支持 .pt 或 .onnx"):
                YOLOv8Engine(path)

    def test_load_utf8_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "names.txt"
            path.write_text("# comment\n甲类\n乙类\n", encoding="utf-8")
            self.assertEqual(load_names(path), {0: "甲类", 1: "乙类"})


class InferenceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.onnx_engine = YOLOv8Engine(ONNX_MODEL, device="cpu")

    def test_onnx_image_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            events = list(
                process_source(
                    self.onnx_engine,
                    SAMPLE,
                    directory,
                    save_json=True,
                )
            )
            image_events = [event for event in events if event.kind == "image"]
            self.assertEqual(len(image_events), 1)
            names = {item.name for item in image_events[0].detections}
            self.assertIn("bus", names)
            self.assertIn("person", names)
            result_image = Path(image_events[0].saved_path)
            self.assertTrue(result_image.is_file())
            self.assertRegex(result_image.name, r"^\d{8}_\d{6}_\d{3}(?:_\d{3})?\.jpg$")
            record = json.loads(result_image.with_suffix(".json").read_text("utf-8"))
            self.assertEqual(record["image_width"], 810)
            self.assertEqual(record["image_height"], 1080)
            self.assertGreaterEqual(len(record["detections"]), 2)

    def test_pt_model_is_switchable(self) -> None:
        engine = YOLOv8Engine(PT_MODEL, device="cpu")
        image = read_image(SAMPLE)
        result = engine.predict(image)
        names = {item.name for item in result.detections}
        self.assertIn("bus", names)
        self.assertIn("person", names)
        self.assertEqual(result.annotated.shape, image.shape)
        self.assertGreaterEqual(len(engine.names), 80)

    def test_custom_names_apply_to_detections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            names_path = Path(directory) / "custom.names"
            names = (ROOT / "data" / "coco.names").read_text("utf-8").splitlines()
            names[0] = "worker"
            names_path.write_text("\n".join(names), encoding="utf-8")
            engine = YOLOv8Engine(
                ONNX_MODEL, names_path=names_path, device="cpu"
            )
            result = engine.predict(read_image(SAMPLE))
            self.assertIn("worker", {item.name for item in result.detections})

    def test_unicode_path_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_dir = Path(directory) / "输入图片"
            output_dir = Path(directory) / "输出结果"
            source_dir.mkdir()
            shutil.copy2(SAMPLE, source_dir / "公交车 一.jpg")
            shutil.copy2(SAMPLE, source_dir / "公交车 二.png")
            events = list(
                process_source(self.onnx_engine, source_dir, output_dir)
            )
            self.assertEqual(
                len([event for event in events if event.kind == "image"]), 2
            )
            self.assertEqual(len(list(output_dir.glob("*.jpg"))), 1)
            self.assertEqual(len(list(output_dir.glob("*.png"))), 1)
            self.assertFalse((output_dir / "公交车 一.jpg").exists())
            self.assertFalse((output_dir / "公交车 二.png").exists())

    def test_video_input_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "clip.mp4"
            source.write_bytes(b"video support removed")
            with self.assertRaisesRegex(ValueError, "视频功能已移除"):
                list(process_source(self.onnx_engine, source, directory))

    def test_cancellation_before_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cancel_event = threading.Event()
            cancel_event.set()
            events = list(
                process_source(
                    self.onnx_engine,
                    SAMPLE,
                    directory,
                    cancel_event=cancel_event,
                )
            )
            self.assertFalse(any(event.kind == "image" for event in events))
            self.assertIn("已取消", events[-1].message)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
