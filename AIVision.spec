# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir build for the PT + ONNX desktop application."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata


project_dir = Path(SPECPATH)
datas = [
    (str(project_dir / "models"), "models"),
    (str(project_dir / "samples"), "samples"),
    (str(project_dir / "data"), "data"),
    (str(project_dir / "assets"), "assets"),
]
binaries = []
hiddenimports = []

# PyInstaller's official hooks already collect Ultralytics YAML/source files
# and ONNX Runtime provider DLLs. Only package metadata is added here because
# Ultralytics checks it before selecting a backend; missing metadata can cause
# unwanted pip auto-update attempts on an offline machine. Avoid collect_all:
# it pulls ONNX test suites, transformer tools, trackers, and server features.
for package in ("ultralytics", "onnx", "onnxruntime", "lap"):
    datas += copy_metadata(package)

# CustomTkinter carries theme JSON assets required by the modern UI. Hikvision's proprietary MVS SDK is
# intentionally loaded from the user's official MVS installation instead.
for package in ("customtkinter",):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports


a = Analysis(
    [str(project_dir / "app.py")],
    pathex=[str(project_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "notebook",
        "jupyter",
        "pyarrow",
        "tensorboard",
        "tensorflow",
        "torch.utils.tensorboard",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AIVision",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_dir / "assets" / "app.ico"),
    version=str(project_dir / "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AIVision",
)
