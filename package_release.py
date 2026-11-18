"""Create clean source and Windows portable ZIP deliverables."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
DIST = PROJECT / "dist" / "AIVision"
RELEASE = PROJECT / "release"
SOURCE_ZIP = RELEASE / "AI Vision-v2.6-源码.zip"
PORTABLE_ZIP = RELEASE / "AI Vision-v2.6-便携版.zip"
EXCLUDED_DIRS = {
    ".venv",
    "build",
    "dist",
    "__pycache__",
    ".git",
    "test-output",
    "smoke-output",
    "exe-test-output",
}
EXCLUDED_FILES = {"settings.json", ".DS_Store"}


def source_files():
    for path in sorted(PROJECT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(PROJECT)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if path.name in EXCLUDED_FILES or path.suffix in {".pyc", ".pyo"}:
            continue
        if relative.parts[0] in {"input", "output"} and path.name != ".gitkeep":
            continue
        yield path, Path("AI Vision 源码") / relative


def add_tree(archive: zipfile.ZipFile, root: Path, prefix: str) -> None:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "settings.json":
            continue
        relative = path.relative_to(root)
        if relative.parts[0] in {"input", "output"}:
            continue
        archive.write(path, Path(prefix) / relative)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    RELEASE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        SOURCE_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path, arcname in source_files():
            archive.write(path, arcname)

    executable = DIST / "AIVision.exe"
    if not executable.is_file():
        raise FileNotFoundError(f"EXE 尚未构建：{executable}")
    for source_name, target_name in (
        ("README.md", "使用说明.md"),
        ("快速使用.txt", "快速使用.txt"),
        ("THIRD_PARTY_NOTICES.md", "第三方许可说明.md"),
        ("settings.example.json", "settings.example.json"),
        ("docs/CAMERA_SETUP.md", "相机接入指南.md"),
    ):
        shutil.copy2(PROJECT / source_name, DIST / target_name)
    (DIST / "output").mkdir(exist_ok=True)

    with zipfile.ZipFile(
        PORTABLE_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=1
    ) as archive:
        add_tree(archive, DIST, "AI Vision")

    lines = []
    installer = RELEASE / "AI Vision-v2.6-Setup.exe"
    for path in (SOURCE_ZIP, PORTABLE_ZIP, installer):
        lines.append(f"{sha256(path)}  {path.name}")
        print(f"{path.name}: {path.stat().st_size / 1024**2:.1f} MiB")
    (RELEASE / "AI Vision-v2.6-SHA256.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Release directory: {RELEASE}")


if __name__ == "__main__":
    main()
