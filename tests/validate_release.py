"""Validate frozen EXE images, modern GUI, and optionally a real USB camera."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent
DIST = PROJECT / "dist" / "AIVision"
EXE = DIST / "AIVision.exe"
REPORT_DIR = PROJECT / "release" / "validation-v2.6"


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    return env


def _run(command: list[str], timeout: int = 240) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=PROJECT,
        env=_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def run_cli(name: str, extra_args: list[str]) -> dict:
    output = REPORT_DIR / f"{name}_{int(time.time() * 1000)}"
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(EXE), "--cli", "--device", "cpu", "--output", str(output),
        "--save-json", *extra_args,
    ]
    started = time.perf_counter()
    completed = _run(command)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} EXE returned {completed.returncode}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    images = list(output.glob("*.jpg"))
    if len(images) != 1:
        raise AssertionError(f"{name}: expected one timestamped image, found {images}")
    image_path = images[0]
    json_path = image_path.with_suffix(".json")
    if image_path.stat().st_size == 0:
        raise AssertionError(f"{name}: annotated image missing")
    record = json.loads(json_path.read_text(encoding="utf-8"))
    names = {item["name"] for item in record["detections"]}
    if not {"bus", "person"}.issubset(names):
        raise AssertionError(f"{name}: unexpected detections: {names}")
    return {
        "status": "passed",
        "elapsed_seconds": round(elapsed, 3),
        "detections": len(record["detections"]),
        "classes": sorted(names),
        "model": record["model"],
    }


def run_usb_camera(backend: str) -> dict:
    output = REPORT_DIR / f"usb_{backend}_{int(time.time() * 1000)}"
    command = [
        str(EXE), "--cli",
        "--model", str(DIST / "_internal" / "models" / "yolov8n.onnx"),
        "--source", "usb://0",
        "--device", "cpu",
        "--camera-backend", backend,
        "--run-subdir",
        "--save-json",
        "--output", str(output),
    ]
    started = time.perf_counter()
    completed = _run(command)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Frozen USB camera failed: {completed.returncode}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    runs = list(output.glob("run_*"))
    if len(runs) != 1:
        raise AssertionError(f"Expected one run directory, found: {runs}")
    run_dir = runs[0]
    snapshots = list(run_dir.glob("*.jpg"))
    if len(snapshots) != 1 or snapshots[0].stat().st_size == 0:
        raise AssertionError(f"Frozen USB camera photo missing: {snapshots}")
    snapshot = snapshots[0]
    if not snapshot.with_suffix(".json").is_file():
        raise AssertionError("Frozen USB camera JSON result missing")
    if list(run_dir.glob("*.jsonl")):
        raise AssertionError("Continuous JSONL must not be produced in photo mode")
    forbidden = [path.name for path in run_dir.iterdir() if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
    if forbidden:
        raise AssertionError(f"Video files must not be generated: {forbidden}")
    return {
        "status": "passed",
        "backend": backend,
        "elapsed_seconds": round(elapsed, 3),
        "photos": 1,
        "snapshot": str(snapshot),
        "video_files": 0,
    }


def run_hikvision_camera(serial: str) -> dict:
    output = REPORT_DIR / f"hik_{int(time.time() * 1000)}"
    command = [
        str(EXE), "--cli",
        "--model", str(DIST / "_internal" / "models" / "yolov8n.onnx"),
        "--source", f"hik://{serial}",
        "--device", "cpu",
        "--run-subdir",
        "--save-json",
        "--output", str(output),
    ]
    started = time.perf_counter()
    completed = _run(command, timeout=300)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Frozen Hikvision failed: {completed.returncode}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    runs = list(output.glob("run_*"))
    if len(runs) != 1:
        raise AssertionError(f"Expected one Hikvision run directory: {runs}")
    run_dir = runs[0]
    snapshots = list(run_dir.glob("*.jpg"))
    if len(snapshots) != 1 or snapshots[0].stat().st_size == 0:
        raise AssertionError(f"Frozen Hikvision photo missing: {snapshots}")
    if not snapshots[0].with_suffix(".json").is_file():
        raise AssertionError("Frozen Hikvision JSON result missing")
    forbidden = [path.name for path in run_dir.iterdir() if path.suffix.lower() in {".mp4", ".avi"}]
    if forbidden:
        raise AssertionError(f"Video files must not be generated: {forbidden}")
    return {
        "status": "passed",
        "serial": serial,
        "elapsed_seconds": round(elapsed, 3),
        "photos": 1,
        "snapshot": str(snapshots[0]),
        "video_files": 0,
    }


def check_gui_startup() -> dict:
    last_error: str = ""
    for attempt in range(2):
        if attempt:
            time.sleep(3)
        process = subprocess.Popen([str(EXE)], cwd=PROJECT, env=_environment())
        try:
            time.sleep(15)
            return_code = process.poll()
            if return_code is not None:
                last_error = f"Frozen modern GUI exited early: {return_code}"
                continue
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
        return {
            "status": "passed",
            "startup_seconds_checked": 15,
            "attempts": attempt + 1,
        }
    raise RuntimeError(last_error)


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 v2.1 冻结 EXE")
    parser.add_argument("--skip-usb", action="store_true", help="无USB设备时跳过真实相机测试")
    parser.add_argument("--hik-serial", default="", help="可选：验证冻结EXE真实海康相机序列号")
    args = parser.parse_args()
    if not EXE.is_file():
        raise FileNotFoundError(f"EXE not found: {EXE}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"executable": str(EXE), "tests": {}}
    report["tests"]["bundled_pt"] = run_cli("bundled_pt", [])
    report["tests"]["external_onnx"] = run_cli(
        "external_onnx",
        [
            "--model", str(PROJECT / "models" / "yolov8n.onnx"),
            "--source", str(PROJECT / "samples" / "bus.jpg"),
        ],
    )
    if not args.skip_usb:
        for backend in ("dshow", "msmf"):
            report["tests"][f"real_usb_{backend}"] = run_usb_camera(backend)
    if args.hik_serial:
        report["tests"]["real_hikvision"] = run_hikvision_camera(args.hik_serial)
    report["tests"]["gui_startup"] = check_gui_startup()
    report_path = REPORT_DIR / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Frozen release validation passed: {report_path}")


if __name__ == "__main__":
    main()
