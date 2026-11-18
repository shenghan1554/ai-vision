"""Modern CustomTkinter desktop interface for the YOLOv8 engine.

Inference and camera discovery never run on the Tk thread. Workers communicate
through queues only: ordinary processing events are lossless and bounded, while
preview images use a one-slot, newest-frame-wins queue. All widget access,
settings updates and output-directory creation happen on the main thread.
"""

from __future__ import annotations

import copy
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, font, messagebox, ttk
import tkinter as tk
from typing import Any, Dict, Optional, Tuple

import customtkinter as ctk
from PIL import Image, ImageTk

from .camera import (
    CameraDevice,
    CameraOptions,
    discover_cameras,
    is_camera_source,
    normalise_camera_source,
    open_camera,
)
from .engine import YOLOv8Engine
from .output import create_run_output_directory, ensure_writable_directory
from .paths import app_dir, resource_path
from .processor import process_source

__all__ = ["YOLOv8DesktopApp", "run_gui"]


@dataclass
class _WorkerNotice:
    """Internal worker-to-GUI message (not a processing RunEvent)."""

    kind: str
    message: str = ""
    detail: str = ""


@dataclass
class _PreviewNotice:
    """Latest-frame preview, separate from lossless processing events."""

    image: Any


@dataclass
class _CameraDiscoveryNotice:
    """Camera results are data only; Tk widgets are updated by the poller."""

    generation: int
    devices: list[CameraDevice]
    warnings: list[str]
    error: str = ""
    detail: str = ""


_LIGHT = {
    "bg": "#F3F5F8",
    "card": "#FFFFFF",
    "card_soft": "#F8FAFC",
    "field": "#FFFFFF",
    "text": "#17202B",
    "muted": "#667085",
    "border": "#E1E7EF",
    "accent": "#0067C0",
    "accent_hover": "#005A9E",
    "accent_soft": "#EAF3FB",
    "warning": "#9A6700",
    "log": "#F8FAFC",
}

_DARK = {
    "bg": "#17191D",
    "card": "#23262B",
    "card_soft": "#292D33",
    "field": "#1D2025",
    "text": "#F2F4F7",
    "muted": "#AAB2BF",
    "border": "#383D45",
    "accent": "#4CA6E8",
    "accent_hover": "#75BCEF",
    "accent_soft": "#20384B",
    "warning": "#E7B75F",
    "log": "#1D2025",
}

_APPEARANCE_LABELS = {
    "light": "浅色",
    "dark": "深色",
    "system": "跟随系统",
}
_APPEARANCE_MODES = {label: mode for mode, label in _APPEARANCE_LABELS.items()}
_KIND_NAMES = {"usb": "USB / UVC", "hik": "Hikvision MVS"}


def _expand_saved_path(value: Any) -> str:
    """Expand portable settings after the application folder has moved."""
    if not value:
        return ""
    return (
        str(value)
        .replace("{app}", str(app_dir()))
        .replace("{resource}", str(resource_path(".")))
    )


def _make_path_portable(value: Any) -> str:
    """Store paths below the app/resource directory as portable tokens."""
    text = str(value or "").strip()
    if not text or text.isdigit() or is_camera_source(text):
        return text
    path = Path(text).expanduser()
    if not path.is_absolute():
        return text
    for base, token in (
        (app_dir(), "{app}"),
        (resource_path("."), "{resource}"),
    ):
        try:
            relative = path.relative_to(base)
        except (ValueError, OSError):
            continue
        return token if not relative.parts else f"{token}/{relative.as_posix()}"
    return text


def _camera_saved_value(saved: Dict[str, Any], key: str, default: Any) -> Any:
    camera = saved.get("camera")
    if isinstance(camera, dict) and key in camera:
        return camera[key]
    # Accept early development builds that stored flat camera_* keys.
    return saved.get(f"camera_{key}", default)


def _concise_error(message: Any, limit: int = 620) -> str:
    """Keep dialogs actionable; complete tracebacks belong in the log only."""
    text = str(message or "发生未知错误。").strip()
    if "Traceback (most recent call last)" in text:
        text = text.split("Traceback (most recent call last)", 1)[0].strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text + "\n\n详细错误信息已写入运行日志。"


class YOLOv8DesktopApp:
    """Resizable Win11-inspired YOLOv8 desktop application.

    ``root`` may be either :class:`tkinter.Tk` or :class:`customtkinter.CTk`.
    The public variables and worker hooks are intentionally kept compatible
    with the original desktop implementation and its integration tests.
    """

    POLL_INTERVAL_MS = 50
    CONTROL_QUEUE_SIZE = 256
    MAX_LOG_LINES = 1600

    def __init__(self, root: Optional[tk.Tk] = None) -> None:
        ctk.set_default_color_theme("blue")
        self.root = root if root is not None else ctk.CTk()
        self._closing = False
        self._running = False
        self._run_failed = False
        self._run_id = 0
        self._sequence = itertools.count()
        self._engine: Optional[YOLOv8Engine] = None
        self._engine_key: tuple = ()
        # Persistent camera connection: opening a UVC/MVS device is much slower
        # than taking a frame, so keep it alive between photo/live runs.
        self._camera_stream: Any = None
        self._camera_stream_source: Optional[str] = None
        self._camera_stream_lock = threading.RLock()
        self._camera_preconnect_thread: Optional[threading.Thread] = None
        self._camera_preconnect_error: Optional[Exception] = None
        self._worker: Optional[threading.Thread] = None
        self._cancel_event: Optional[threading.Event] = None
        self._poll_after_id: Optional[str] = None
        self._resize_after_id: Optional[str] = None

        self._control_queue: queue.Queue = queue.Queue(
            maxsize=self.CONTROL_QUEUE_SIZE
        )
        self._preview_queue: queue.Queue = queue.Queue(maxsize=1)
        self._preview_source: Optional[Image.Image] = None
        self._preview_photo: Optional[ImageTk.PhotoImage] = None
        self._preview_message = "等待检测画面"
        self._progress_mode = "determinate"

        self._camera_queue: queue.Queue = queue.Queue()
        self._camera_thread: Optional[threading.Thread] = None
        self._camera_generation = 0
        self._camera_discovering = False
        self._camera_devices: list[CameraDevice] = []
        self._camera_warnings: list[str] = []
        self._camera_device_buttons: dict[str, ctk.CTkButton] = {}
        self._available_camera_sources: set[str] = set()
        self.camera_window: Optional[ctk.CTkToplevel] = None
        self.camera_refresh_button: Optional[ctk.CTkButton] = None
        self.camera_list_frame: Optional[ctk.CTkScrollableFrame] = None
        self.last_output_dir: Optional[Path] = None

        self._settings_path = app_dir() / "settings.json"
        saved = self._read_settings()

        self.model_var = tk.StringVar(
            value=(
                _expand_saved_path(saved.get("model"))
                or str(resource_path("models/yolov8n.pt"))
            )
        )
        saved_source = _expand_saved_path(saved.get("source"))
        if saved_source and not is_camera_source(saved_source):
            saved_source = ""  # file/directory inputs are no longer supported
        self.source_var = tk.StringVar(value=saved_source)
        # output_var always contains the base directory, never a temporary run.
        self.output_var = tk.StringVar(
            value=(
                _expand_saved_path(saved.get("output"))
                or str(app_dir() / "output")
            )
        )
        self.conf_var = tk.StringVar(value=str(saved.get("conf", 0.25)))
        self.iou_var = tk.StringVar(value=str(saved.get("iou", 0.45)))
        self.imgsz_var = tk.StringVar(value=str(saved.get("imgsz", 640)))
        saved_device = str(saved.get("device", "auto")).strip().lower()
        if saved_device not in ("auto", "cpu", "0"):
            saved_device = "auto"
        self.device_var = tk.StringVar(value=saved_device)
        self.save_json_var = tk.BooleanVar(
            value=self._as_bool(saved.get("save_json", False))
        )
        saved_mode = str(saved.get("camera_mode", "photo")).strip().lower()
        if saved_mode not in {"photo", "live"}:
            saved_mode = "photo"
        self.camera_mode_var = tk.StringVar(
            value="拍照识别" if saved_mode == "photo" else "实时识别"
        )
        self.unique_output_var = tk.BooleanVar(
            value=self._as_bool(saved.get("unique_output", True))
        )

        appearance = str(saved.get("appearance", "light")).strip().lower()
        if appearance not in _APPEARANCE_LABELS:
            appearance = "light"
        self.appearance_var = tk.StringVar(value=appearance)
        self.appearance_display_var = tk.StringVar(
            value=_APPEARANCE_LABELS[appearance]
        )
        ctk.set_appearance_mode(appearance)

        # Camera acquisition is fully automatic now; the removed 采集设置 panel
        # no longer contributes runtime options. Only the serial is kept so a
        # Hikvision device stays addressable after a rescan.
        self.camera_serial_var = tk.StringVar(
            value=str(_camera_saved_value(saved, "serial", ""))
        )
        self.status_var = tk.StringVar(value="就绪 · 等待开始检测")
        self.model_description_var = tk.StringVar(value="模型尚未加载")
        self.camera_status_var = tk.StringVar(value="扫描 USB 与海康 MVS 相机")
        self.camera_warning_var = tk.StringVar(value="")
        self.camera_selected_name_var = tk.StringVar(value="尚未选择相机")
        self.camera_selected_source_var = tk.StringVar(value="—")
        self.camera_selected_detail_var = tk.StringVar(
            value="刷新设备列表后选择一个相机。"
        )

        self._palette_mode = ""
        self.palette = dict(_LIGHT)
        self._configure_window()
        self._build_ui()
        self._load_window_icon_if_available()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.model_var.trace_add("write", self._on_model_changed)
        self.source_var.trace_add("write", self._on_source_changed)
        self._on_source_changed()
        self._refresh_palette(force=True)
        self._append_log("工作台已就绪。选择模型与输入来源后即可开始检测。")
        self._poll_after_id = self.root.after(
            self.POLL_INTERVAL_MS, self._poll_worker_queues
        )

    # ------------------------------------------------------------------ UI
    def _font(self, size: int = 13, weight: str = "normal") -> ctk.CTkFont:
        return ctk.CTkFont(family=self.font_family, size=size, weight=weight)

    def _configure_window(self) -> None:
        self.root.title("AI Vision")
        self.root.geometry("1440x860")
        self.root.minsize(1180, 720)
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        try:
            families = set(font.families(self.root))
        except tk.TclError:
            families = set()
        self.font_family = next(
            (
                name
                for name in (
                    "Segoe UI Variable",
                    "Microsoft YaHei UI",
                    "Segoe UI",
                    "PingFang SC",
                    "Helvetica Neue",
                )
                if name in families
            ),
            "Segoe UI",
        )
        for font_name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            try:
                font.nametofont(font_name).configure(family=self.font_family, size=10)
            except tk.TclError:
                pass
        self.progress_style = ttk.Style(self.root)
        try:
            self.progress_style.theme_use("clam")
        except tk.TclError:
            pass

    def _load_window_icon_if_available(self) -> None:
        try:
            icon_path = resource_path("assets/app.ico")
            if icon_path.is_file() and sys.platform == "win32":
                self.root.iconbitmap(str(icon_path))
        except (OSError, tk.TclError):
            pass

    def _build_ui(self) -> None:
        shell = ctk.CTkFrame(self.root, fg_color="transparent", corner_radius=0)
        shell.grid(row=0, column=0, sticky="nsew")

        header = ctk.CTkFrame(shell, fg_color="transparent")
        header.pack(fill=tk.X, padx=26, pady=(20, 14))

        brand = ctk.CTkFrame(
            header,
            width=44,
            height=44,
            corner_radius=13,
            fg_color=(_LIGHT["accent"], _DARK["accent"]),
        )
        brand.pack(side=tk.LEFT)
        brand.pack_propagate(False)
        ctk.CTkLabel(
            brand,
            text="A",
            font=self._font(23, "bold"),
            text_color=("#FFFFFF", "#10202D"),
        ).pack(expand=True)

        title_block = ctk.CTkFrame(header, fg_color="transparent")
        title_block.pack(side=tk.LEFT, padx=(12, 0))
        ctk.CTkLabel(
            title_block,
            text="AI Vision",
            font=self._font(23, "bold"),
            text_color=(_LIGHT["text"], _DARK["text"]),
        ).pack(anchor=tk.W)
        self.appearance_selector = ctk.CTkSegmentedButton(
            header,
            values=list(_APPEARANCE_MODES),
            variable=self.appearance_display_var,
            command=self._on_appearance_changed,
            width=238,
            height=34,
            corner_radius=9,
            font=self._font(11),
            selected_color=(_LIGHT["accent"], _DARK["accent"]),
            selected_hover_color=(_LIGHT["accent_hover"], _DARK["accent_hover"]),
            unselected_color=(_LIGHT["card"], _DARK["card"]),
            unselected_hover_color=(_LIGHT["accent_soft"], _DARK["accent_soft"]),
            text_color=(_LIGHT["text"], _DARK["text"]),
        )
        self.appearance_selector.pack(side=tk.RIGHT)

        body = ctk.CTkFrame(shell, fg_color="transparent")
        body.pack(fill=tk.BOTH, expand=True, padx=24)
        # Three columns: settings | large preview | run log.
        body.columnconfigure(0, minsize=372, weight=0)
        body.columnconfigure(1, weight=1, minsize=520)
        body.columnconfigure(2, minsize=310, weight=0)
        body.rowconfigure(0, weight=1)

        self._build_settings_panel(body)
        self._build_preview_panel(body)
        self._build_log_panel(body)

        footer = ctk.CTkFrame(shell, fg_color="transparent", height=42)
        footer.pack(fill=tk.X, padx=27, pady=(11, 16))
        footer.pack_propagate(False)
        ctk.CTkLabel(
            footer,
            text="●",
            font=self._font(11),
            text_color=(_LIGHT["accent"], _DARK["accent"]),
        ).pack(side=tk.LEFT, padx=(0, 7))
        ctk.CTkLabel(
            footer,
            textvariable=self.status_var,
            font=self._font(11),
            text_color=(_LIGHT["muted"], _DARK["muted"]),
            anchor=tk.W,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ctk.CTkLabel(
            footer,
            text="任务进度",
            font=self._font(10),
            text_color=(_LIGHT["muted"], _DARK["muted"]),
        ).pack(side=tk.LEFT, padx=(14, 9))
        self.progress = ttk.Progressbar(
            footer,
            orient=tk.HORIZONTAL,
            mode="determinate",
            maximum=1,
            value=0,
            length=230,
            style="YoloModern.Horizontal.TProgressbar",
        )
        self.progress.pack(side=tk.RIGHT, pady=15)

    def _build_settings_panel(self, parent: ctk.CTkFrame) -> None:
        panel = ctk.CTkFrame(
            parent,
            corner_radius=17,
            fg_color=(_LIGHT["card"], _DARK["card"]),
            border_color=(_LIGHT["border"], _DARK["border"]),
            border_width=1,
        )
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)

        settings = ctk.CTkScrollableFrame(
            panel,
            fg_color="transparent",
            corner_radius=0,
            scrollbar_button_color=(_LIGHT["border"], _DARK["border"]),
            scrollbar_button_hover_color=(_LIGHT["muted"], _DARK["muted"]),
        )
        settings.grid(row=0, column=0, sticky="nsew", padx=13, pady=(17, 4))
        settings.columnconfigure(0, weight=1)

        self._section_header(settings, "01", "模型")
        self._path_group(
            settings,
            "模型文件",
            self.model_var,
            self._browse_model,
            "浏览",
        )
        self._divider(settings)
        self._section_header(settings, "02", "相机与输出")
        self.camera_manage_button = ctk.CTkButton(
            settings,
            text="相机管理",
            command=self._open_camera_manager,
            height=37,
            corner_radius=9,
            font=self._font(11, "bold"),
            fg_color=(_LIGHT["accent_soft"], _DARK["accent_soft"]),
            hover_color=("#DCECF9", "#29485F"),
            text_color=(_LIGHT["accent"], "#8ECCF4"),
            border_color=("#CFE3F4", "#36566C"),
            border_width=1,
        )
        self.camera_manage_button.pack(fill=tk.X, padx=4, pady=(8, 0))
        mode_row = ctk.CTkFrame(settings, fg_color="transparent")
        mode_row.pack(fill=tk.X, padx=4, pady=(8, 0))
        ctk.CTkLabel(
            mode_row,
            text="识别模式",
            font=self._font(10),
            text_color=(_LIGHT["muted"], _DARK["muted"]),
            width=58,
            anchor=tk.W,
        ).pack(side=tk.LEFT)
        self.mode_selector = ctk.CTkSegmentedButton(
            mode_row,
            values=["拍照识别", "实时识别"],
            variable=self.camera_mode_var,
            font=self._font(11),
            height=30,
            selected_color=(_LIGHT["accent"], _DARK["accent"]),
            selected_hover_color=(_LIGHT["accent_hover"], _DARK["accent_hover"]),
            unselected_color=(_LIGHT["card_soft"], _DARK["card_soft"]),
            unselected_hover_color=(_LIGHT["accent_soft"], _DARK["accent_soft"]),
            text_color=(_LIGHT["text"], _DARK["text"]),
        )
        self.mode_selector.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._path_group(
            settings,
            "输出目录",
            self.output_var,
            self._browse_output,
            "浏览",
            top_margin=13,
        )
        self.unique_output_check = ctk.CTkCheckBox(
            settings,
            text="每次任务创建独立文件夹",
            variable=self.unique_output_var,
            font=self._font(11),
            checkbox_width=18,
            checkbox_height=18,
            corner_radius=5,
            border_width=2,
            fg_color=(_LIGHT["accent"], _DARK["accent"]),
            hover_color=(_LIGHT["accent_hover"], _DARK["accent_hover"]),
            text_color=(_LIGHT["text"], _DARK["text"]),
        )
        self.unique_output_check.pack(anchor=tk.W, padx=5, pady=(9, 0))

        self._divider(settings)
        self._section_header(settings, "03", "推理参数")
        parameters = ctk.CTkFrame(settings, fg_color="transparent")
        parameters.pack(fill=tk.X, padx=3, pady=(10, 0))
        parameters.columnconfigure((0, 1), weight=1)
        self._parameter_field(parameters, 0, 0, "置信度", self.conf_var)
        self._parameter_field(parameters, 0, 1, "IoU 阈值", self.iou_var)
        self._parameter_field(parameters, 1, 0, "图像尺寸", self.imgsz_var)
        device_group = ctk.CTkFrame(parameters, fg_color="transparent")
        device_group.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=(0, 10))
        ctk.CTkLabel(
            device_group,
            text="运行设备",
            font=self._font(10),
            text_color=(_LIGHT["muted"], _DARK["muted"]),
            anchor=tk.W,
        ).pack(fill=tk.X)
        self.device_combo = ctk.CTkOptionMenu(
            device_group,
            variable=self.device_var,
            values=("auto", "cpu", "0"),
            height=34,
            corner_radius=8,
            font=self._font(11),
            fg_color=(_LIGHT["field"], _DARK["field"]),
            button_color=(_LIGHT["border"], _DARK["border"]),
            button_hover_color=(_LIGHT["accent"], _DARK["accent"]),
            text_color=(_LIGHT["text"], _DARK["text"]),
            dropdown_fg_color=(_LIGHT["card"], _DARK["card"]),
            dropdown_text_color=(_LIGHT["text"], _DARK["text"]),
        )
        self.device_combo.pack(fill=tk.X, pady=(5, 0))

        self.save_json_switch = ctk.CTkSwitch(
            settings,
            text="同时保存检测数据 JSON",
            variable=self.save_json_var,
            font=self._font(11),
            progress_color=(_LIGHT["accent"], _DARK["accent"]),
            button_color=("#FFFFFF", "#FFFFFF"),
            button_hover_color=("#F2F2F2", "#F2F2F2"),
            text_color=(_LIGHT["text"], _DARK["text"]),
        )
        self.save_json_switch.pack(anchor=tk.W, padx=5, pady=(5, 4))

        actions = ctk.CTkFrame(panel, fg_color="transparent")
        actions.grid(row=1, column=0, sticky="ew", padx=18, pady=(7, 17))
        actions.columnconfigure(0, weight=3)
        actions.columnconfigure(1, weight=1)
        self.start_button = ctk.CTkButton(
            actions,
            text="开始检测",
            command=self.start_processing,
            height=43,
            corner_radius=10,
            font=self._font(13, "bold"),
            fg_color=(_LIGHT["accent"], "#1679B9"),
            hover_color=(_LIGHT["accent_hover"], "#248DCC"),
            text_color="#FFFFFF",
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.stop_button = ctk.CTkButton(
            actions,
            text="取消",
            command=self.stop_processing,
            state=tk.DISABLED,
            height=43,
            corner_radius=10,
            font=self._font(12),
            fg_color=(_LIGHT["card_soft"], _DARK["card_soft"]),
            hover_color=("#E9EDF2", "#353A42"),
            text_color=(_LIGHT["text"], _DARK["text"]),
            border_color=(_LIGHT["border"], _DARK["border"]),
            border_width=1,
        )
        self.stop_button.grid(row=0, column=1, sticky="ew")
        self.open_output_button = ctk.CTkButton(
            actions,
            text="打开输出目录  ↗",
            command=self._open_output_directory,
            height=34,
            corner_radius=9,
            font=self._font(11),
            fg_color="transparent",
            hover_color=(_LIGHT["accent_soft"], _DARK["accent_soft"]),
            text_color=(_LIGHT["accent"], "#8ECCF4"),
        )
        self.open_output_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))

    def _build_preview_panel(self, parent: ctk.CTkFrame) -> None:
        """Center column: the preview takes the full available height."""
        preview_card = ctk.CTkFrame(
            parent,
            corner_radius=17,
            fg_color=(_LIGHT["card"], _DARK["card"]),
            border_color=(_LIGHT["border"], _DARK["border"]),
            border_width=1,
        )
        preview_card.grid(row=0, column=1, sticky="nsew")
        preview_card.columnconfigure(0, weight=1)
        preview_card.rowconfigure(1, weight=1)
        preview_header = ctk.CTkFrame(preview_card, fg_color="transparent")
        preview_header.grid(row=0, column=0, sticky="ew", padx=18, pady=(15, 11))
        ctk.CTkLabel(
            preview_header,
            text="拍照预览",
            font=self._font(15, "bold"),
            text_color=(_LIGHT["text"], _DARK["text"]),
        ).pack(side=tk.LEFT)
        ctk.CTkLabel(
            preview_header,
            text="●  最新画面",
            font=self._font(10),
            text_color=(_LIGHT["accent"], "#8ECCF4"),
            fg_color=(_LIGHT["accent_soft"], _DARK["accent_soft"]),
            corner_radius=8,
            padx=10,
            pady=4,
        ).pack(side=tk.RIGHT)

        canvas_host = ctk.CTkFrame(
            preview_card, fg_color="#11161C", corner_radius=12
        )
        canvas_host.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))
        self.preview_canvas = tk.Canvas(
            canvas_host,
            background="#11161C",
            highlightthickness=0,
            borderwidth=0,
            width=680,
            height=420,
        )
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.preview_canvas.bind("<Configure>", self._on_preview_resize)
        self._render_preview()

    def _build_log_panel(self, parent: ctk.CTkFrame) -> None:
        """Right column: run log occupies the full height."""
        log_card = ctk.CTkFrame(
            parent,
            corner_radius=17,
            fg_color=(_LIGHT["card"], _DARK["card"]),
            border_color=(_LIGHT["border"], _DARK["border"]),
            border_width=1,
        )
        log_card.grid(row=0, column=2, sticky="nsew")
        log_card.columnconfigure(0, weight=1)
        log_card.rowconfigure(1, weight=1)
        log_header = ctk.CTkFrame(log_card, fg_color="transparent")
        log_header.grid(row=0, column=0, sticky="ew", padx=18, pady=(12, 8))
        ctk.CTkLabel(
            log_header,
            text="运行日志",
            font=self._font(14, "bold"),
            text_color=(_LIGHT["text"], _DARK["text"]),
        ).pack(side=tk.LEFT)
        ctk.CTkLabel(
            log_header,
            text="详细异常堆栈仅显示在这里",
            font=self._font(10),
            text_color=(_LIGHT["muted"], _DARK["muted"]),
        ).pack(side=tk.LEFT, padx=12)
        ctk.CTkButton(
            log_header,
            text="清空",
            command=self._clear_log,
            width=54,
            height=26,
            corner_radius=7,
            font=self._font(10),
            fg_color="transparent",
            hover_color=(_LIGHT["accent_soft"], _DARK["accent_soft"]),
            text_color=(_LIGHT["muted"], _DARK["muted"]),
        ).pack(side=tk.RIGHT)

        log_host = ctk.CTkFrame(
            log_card,
            fg_color=(_LIGHT["log"], _DARK["log"]),
            corner_radius=10,
        )
        log_host.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        # Keep this a standard tk.Text for integrations and automated tests.
        self.log_text = tk.Text(
            log_host,
            width=38,
            height=9,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=(self.font_family, 10),
            background=_LIGHT["log"],
            foreground=_LIGHT["text"],
            insertbackground=_LIGHT["text"],
            selectbackground="#B9DDF5",
            selectforeground=_LIGHT["text"],
            highlightthickness=0,
            borderwidth=0,
            relief=tk.FLAT,
            padx=10,
            pady=8,
            spacing1=2,
            spacing3=2,
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll = ctk.CTkScrollbar(
            log_host,
            orientation=tk.VERTICAL,
            command=self.log_text.yview,
            width=10,
            corner_radius=6,
            fg_color="transparent",
            button_color=(_LIGHT["border"], _DARK["border"]),
            button_hover_color=(_LIGHT["muted"], _DARK["muted"]),
        )
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5, padx=(0, 4))
        self.log_text.configure(yscrollcommand=log_scroll.set)

    def _section_header(
        self, parent: ctk.CTkFrame, number: str, title: str
    ) -> None:
        group = ctk.CTkFrame(parent, fg_color="transparent")
        group.pack(fill=tk.X, padx=4, pady=(0, 5))
        ctk.CTkLabel(
            group,
            text=number,
            width=29,
            height=23,
            corner_radius=7,
            font=self._font(10, "bold"),
            text_color=(_LIGHT["accent"], "#8ECCF4"),
            fg_color=(_LIGHT["accent_soft"], _DARK["accent_soft"]),
        ).pack(side=tk.LEFT, padx=(0, 9))
        text_group = ctk.CTkFrame(group, fg_color="transparent")
        text_group.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ctk.CTkLabel(
            text_group,
            text=title,
            font=self._font(14, "bold"),
            text_color=(_LIGHT["text"], _DARK["text"]),
            anchor=tk.W,
        ).pack(fill=tk.X)

    def _path_group(
        self,
        parent: ctk.CTkFrame,
        label: str,
        variable: tk.StringVar,
        command: Any,
        button_text: str,
        extra_command: Any = None,
        extra_text: str = "",
        top_margin: int = 9,
    ) -> None:
        group = ctk.CTkFrame(parent, fg_color="transparent")
        group.pack(fill=tk.X, padx=4, pady=(top_margin, 0))
        ctk.CTkLabel(
            group,
            text=label,
            font=self._font(10),
            text_color=(_LIGHT["muted"], _DARK["muted"]),
            anchor=tk.W,
        ).pack(fill=tk.X)
        row = ctk.CTkFrame(group, fg_color="transparent")
        row.pack(fill=tk.X, pady=(5, 0))
        entry = ctk.CTkEntry(
            row,
            textvariable=variable,
            height=35,
            corner_radius=8,
            border_width=1,
            font=self._font(11),
            fg_color=(_LIGHT["field"], _DARK["field"]),
            border_color=(_LIGHT["border"], _DARK["border"]),
            text_color=(_LIGHT["text"], _DARK["text"]),
        )
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ctk.CTkButton(
            row,
            text=button_text,
            command=command,
            width=54,
            height=33,
            corner_radius=8,
            font=self._font(10),
            fg_color=(_LIGHT["card_soft"], _DARK["card_soft"]),
            hover_color=(_LIGHT["accent_soft"], _DARK["accent_soft"]),
            text_color=(_LIGHT["text"], _DARK["text"]),
            border_color=(_LIGHT["border"], _DARK["border"]),
            border_width=1,
        ).pack(side=tk.LEFT, padx=(6, 0))
        if extra_command is not None:
            ctk.CTkButton(
                row,
                text=extra_text,
                command=extra_command,
                width=54,
                height=33,
                corner_radius=8,
                font=self._font(10),
                fg_color=(_LIGHT["card_soft"], _DARK["card_soft"]),
                hover_color=(_LIGHT["accent_soft"], _DARK["accent_soft"]),
                text_color=(_LIGHT["text"], _DARK["text"]),
                border_color=(_LIGHT["border"], _DARK["border"]),
                border_width=1,
            ).pack(side=tk.LEFT, padx=(6, 0))

    def _parameter_field(
        self,
        parent: ctk.CTkFrame,
        row: int,
        column: int,
        label: str,
        variable: tk.StringVar,
    ) -> None:
        group = ctk.CTkFrame(parent, fg_color="transparent")
        group.grid(
            row=row,
            column=column,
            sticky="ew",
            padx=(0, 6) if column == 0 else (6, 0),
            pady=(0, 10),
        )
        ctk.CTkLabel(
            group,
            text=label,
            font=self._font(10),
            text_color=(_LIGHT["muted"], _DARK["muted"]),
            anchor=tk.W,
        ).pack(fill=tk.X)
        ctk.CTkEntry(
            group,
            textvariable=variable,
            height=34,
            corner_radius=8,
            border_width=1,
            font=self._font(11),
            fg_color=(_LIGHT["field"], _DARK["field"]),
            border_color=(_LIGHT["border"], _DARK["border"]),
            text_color=(_LIGHT["text"], _DARK["text"]),
        ).pack(fill=tk.X, pady=(5, 0))

    @staticmethod
    def _divider(parent: ctk.CTkFrame) -> None:
        ctk.CTkFrame(
            parent,
            height=1,
            fg_color=(_LIGHT["border"], _DARK["border"]),
        ).pack(fill=tk.X, padx=4, pady=16)

    # --------------------------------------------------------------- Appearance
    def _on_appearance_changed(self, choice: str) -> None:
        mode = _APPEARANCE_MODES.get(choice, "light")
        self.appearance_var.set(mode)
        ctk.set_appearance_mode(mode)
        self.root.after_idle(self._refresh_palette)
        self._save_settings()

    def _refresh_palette(self, force: bool = False) -> None:
        if self._closing:
            return
        try:
            mode = ctk.get_appearance_mode().lower()
        except Exception:
            mode = self.appearance_var.get()
        if mode not in ("light", "dark"):
            mode = "light"
        if not force and mode == self._palette_mode:
            return
        self._palette_mode = mode
        self.palette = dict(_LIGHT if mode == "light" else _DARK)
        colors = self.palette
        try:
            self.root.configure(bg=colors["bg"])
        except tk.TclError:
            pass
        if hasattr(self, "log_text"):
            self.log_text.configure(
                background=colors["log"],
                foreground=colors["text"],
                insertbackground=colors["text"],
                selectforeground=colors["text"],
            )
        if hasattr(self, "progress_style"):
            self.progress_style.configure(
                "YoloModern.Horizontal.TProgressbar",
                troughcolor=colors["border"],
                background=colors["accent"],
                bordercolor=colors["border"],
                lightcolor=colors["accent"],
                darkcolor=colors["accent"],
                borderwidth=0,
                thickness=6,
            )

    # ------------------------------------------------------------- File UI
    def _initial_directory(self, value: str) -> str:
        try:
            if is_camera_source(value):
                return str(app_dir())
            path = Path(value).expanduser()
            if path.is_dir():
                return str(path)
            if path.parent.is_dir():
                return str(path.parent)
        except (OSError, ValueError):
            pass
        return str(app_dir())

    def _browse_model(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="选择 YOLOv8 模型",
            initialdir=self._initial_directory(self.model_var.get()),
            filetypes=(
                ("YOLOv8 模型", "*.pt *.onnx"),
                ("PyTorch 模型", "*.pt"),
                ("ONNX 模型", "*.onnx"),
                ("所有文件", "*.*"),
            ),
        )
        if selected:
            self.model_var.set(selected)

    def _browse_output(self) -> None:
        selected = filedialog.askdirectory(
            parent=self.root,
            title="选择输出目录",
            initialdir=self._initial_directory(self.output_var.get()),
        )
        if selected:
            self.output_var.set(selected)

    def _open_output_directory(self) -> None:
        path = self.last_output_dir or Path(
            self.output_var.get().strip() or str(app_dir() / "output")
        ).expanduser()
        try:
            path.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except (OSError, ValueError) as exc:
            messagebox.showerror("无法打开目录", str(exc), parent=self.root)

    def _on_model_changed(self, *_args: Any) -> None:
        if self._closing or self._running:
            return
        text = self.model_var.get().strip()
        suffix = Path(text).suffix.upper().lstrip(".") or "模型"
        name = Path(text).name if text else "未选择文件"
        self.model_description_var.set(f"待加载 · {name} · {suffix}")

    def _on_source_changed(self, *_args: Any) -> None:
        # Only cameras are valid inputs now; nothing to render here.
        return

    # ---------------------------------------------------------- Camera options
    @staticmethod
    def _optional_int(
        variable: tk.StringVar, label: str, minimum: int = 0, default: int = 0
    ) -> int:
        text = variable.get().strip()
        if not text:
            return default
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"{label}必须是整数；自动模式可填 0 或留空。") from exc
        if value < minimum:
            raise ValueError(f"{label}不能小于 {minimum}。")
        return value

    def _camera_options(self) -> CameraOptions:
        """Return automatic acquisition options (no manual settings remain)."""
        return CameraOptions(
            backend="auto",
            width=0,
            height=0,
            fps=0.0,
            timeout_ms=3000,
            warmup_frames=0,
            mjpg=False,
            serial=self.camera_serial_var.get().strip(),
            exposure_us=0.0,
            gain=0.0,
        )

    def _open_camera_manager(self) -> None:
        if self._closing:
            return
        if self.camera_window is not None:
            try:
                if self.camera_window.winfo_exists():
                    self.camera_window.lift()
                    self.camera_window.focus_set()
                    return
            except tk.TclError:
                pass
        window = ctk.CTkToplevel(self.root)
        self.camera_window = window
        window.title("相机管理 · USB / 海康 MVS")
        window.geometry("920x680")
        window.minsize(780, 590)
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", self._close_camera_manager)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(1, weight=1)

        header = ctk.CTkFrame(window, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 13))
        title_group = ctk.CTkFrame(header, fg_color="transparent")
        title_group.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ctk.CTkLabel(
            title_group,
            text="相机管理",
            font=self._font(21, "bold"),
            text_color=(_LIGHT["text"], _DARK["text"]),
        ).pack(anchor=tk.W)
        ctk.CTkLabel(
            title_group,
            textvariable=self.camera_status_var,
            font=self._font(10),
            text_color=(_LIGHT["muted"], _DARK["muted"]),
        ).pack(anchor=tk.W, pady=(3, 0))
        self.camera_refresh_button = ctk.CTkButton(
            header,
            text="刷新设备",
            command=self._refresh_cameras,
            width=96,
            height=35,
            font=self._font(11, "bold"),
            fg_color=(_LIGHT["accent"], "#1679B9"),
            hover_color=(_LIGHT["accent_hover"], "#248DCC"),
        )
        self.camera_refresh_button.pack(side=tk.RIGHT)

        body = ctk.CTkFrame(window, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=22)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2, minsize=275)
        body.rowconfigure(0, weight=1)

        list_card = ctk.CTkFrame(
            body,
            corner_radius=15,
            fg_color=(_LIGHT["card"], _DARK["card"]),
            border_color=(_LIGHT["border"], _DARK["border"]),
            border_width=1,
        )
        list_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        list_card.columnconfigure(0, weight=1)
        list_card.rowconfigure(1, weight=1)
        ctk.CTkLabel(
            list_card,
            text="已发现设备",
            font=self._font(13, "bold"),
            text_color=(_LIGHT["text"], _DARK["text"]),
            anchor=tk.W,
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 7))
        self.camera_list_frame = ctk.CTkScrollableFrame(
            list_card,
            fg_color="transparent",
            scrollbar_button_color=(_LIGHT["border"], _DARK["border"]),
        )
        self.camera_list_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.camera_list_frame.columnconfigure(0, weight=1)
        self.camera_warning_label = ctk.CTkLabel(
            list_card,
            textvariable=self.camera_warning_var,
            font=self._font(10),
            text_color=(_LIGHT["warning"], _DARK["warning"]),
            wraplength=450,
            justify=tk.LEFT,
            anchor=tk.W,
        )
        self.camera_warning_label.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 10))

        right = ctk.CTkScrollableFrame(
            body,
            fg_color="transparent",
            scrollbar_button_color=(_LIGHT["border"], _DARK["border"]),
        )
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        selected = ctk.CTkFrame(
            right,
            corner_radius=15,
            fg_color=(_LIGHT["card"], _DARK["card"]),
            border_color=(_LIGHT["border"], _DARK["border"]),
            border_width=1,
        )
        selected.pack(fill=tk.X, pady=(0, 12))
        ctk.CTkLabel(
            selected,
            text="当前选择",
            font=self._font(13, "bold"),
            text_color=(_LIGHT["text"], _DARK["text"]),
            anchor=tk.W,
        ).pack(fill=tk.X, padx=16, pady=(15, 10))
        for variable, size, wrap in (
            (self.camera_selected_name_var, 12, 280),
            (self.camera_selected_source_var, 11, 280),
            (self.camera_selected_detail_var, 10, 280),
        ):
            ctk.CTkLabel(
                selected,
                textvariable=variable,
                font=self._font(size),
                text_color=(_LIGHT["muted"], _DARK["muted"]),
                wraplength=wrap,
                justify=tk.LEFT,
                anchor=tk.W,
            ).pack(fill=tk.X, padx=16, pady=(0, 8))

        manual = ctk.CTkFrame(
            right,
            corner_radius=15,
            fg_color=(_LIGHT["card"], _DARK["card"]),
            border_color=(_LIGHT["border"], _DARK["border"]),
            border_width=1,
        )
        manual.pack(fill=tk.X, pady=(0, 12))
        ctk.CTkButton(
            right,
            text="完成并返回工作台",
            command=self._close_camera_manager,
            height=38,
            corner_radius=9,
            font=self._font(11, "bold"),
            fg_color=(_LIGHT["accent"], "#1679B9"),
            hover_color=(_LIGHT["accent_hover"], "#248DCC"),
        ).pack(fill=tk.X, pady=(0, 14))

        footer = ctk.CTkFrame(window, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=25, pady=14)
        ctk.CTkLabel(
            footer,
            text="选中设备会立即写入工作台的输入来源；扫描在后台执行，不会冻结界面。",
            font=self._font(10),
            text_color=(_LIGHT["muted"], _DARK["muted"]),
        ).pack(side=tk.LEFT)
        self._rebuild_camera_list()
        self._set_running_state(self._running)

    def _close_camera_manager(self) -> None:
        window = self.camera_window
        self.camera_window = None
        self._camera_device_buttons.clear()
        if window is not None:
            try:
                window.destroy()
            except tk.TclError:
                pass

    @staticmethod
    def _window_exists(window: Any) -> bool:
        if window is None:
            return False
        try:
            return bool(window.winfo_exists())
        except tk.TclError:
            return False

    def _refresh_cameras(self) -> None:
        if self._closing or self._camera_discovering or self._running:
            return
        try:
            options = self._camera_options()
            max_usb_index = 4
        except ValueError as exc:
            messagebox.showerror("相机设置有误", str(exc), parent=self.root)
            return

        # Discovery opens and closes each device. Do not let it fight with our
        # persistent stream for exclusive access; the next scan reconnects.
        self._release_camera_stream()
        options = replace(options, warmup_frames=0)
        self._camera_generation += 1
        generation = self._camera_generation
        self._camera_discovering = True
        self.camera_status_var.set("正在后台扫描相机，请稍候…")
        self.camera_warning_var.set("")
        if self._window_exists(self.camera_refresh_button):
            self.camera_refresh_button.configure(state=tk.DISABLED, text="扫描中…")
        self._camera_thread = threading.Thread(
            target=self._camera_discovery_main,
            args=(generation, options, max_usb_index),
            name="camera-discovery",
            daemon=True,
        )
        self._camera_thread.start()

    def _camera_discovery_main(
        self, generation: int, options: CameraOptions, max_usb_index: int
    ) -> None:
        try:
            devices, warnings = discover_cameras(
                probe_usb=True, max_usb_index=max_usb_index, options=options
            )
            notice = _CameraDiscoveryNotice(
                generation=generation, devices=devices, warnings=warnings
            )
        except Exception as exc:
            notice = _CameraDiscoveryNotice(
                generation=generation,
                devices=[],
                warnings=[],
                error=str(exc) or repr(exc),
                detail=traceback.format_exc(),
            )
        self._camera_queue.put(notice)

    def _maybe_auto_select_camera(self) -> None:
        """Pick the first available camera when the current one is unusable."""
        current = self.source_var.get().strip()
        if current:
            available_sources = {
                device.source for device in self._camera_devices if device.available
            }
            if current in available_sources:
                return
            # Saved device is gone (unplugged/replaced): fall back to the first one.
        available = [device for device in self._camera_devices if device.available]
        if not available:
            return
        self._select_camera_device(available[0])

    def _handle_camera_notice(self, notice: _CameraDiscoveryNotice) -> None:
        if notice.generation != self._camera_generation or self._closing:
            return
        self._camera_discovering = False
        self._camera_thread = None
        if self._window_exists(self.camera_refresh_button):
            self.camera_refresh_button.configure(
                state=tk.DISABLED if self._running else tk.NORMAL,
                text="刷新设备",
            )
        if notice.error:
            self._camera_devices = []
            self._camera_warnings = []
            self.camera_status_var.set("相机扫描失败 · 可检查设置后重试")
            self.camera_warning_var.set("")
            self._append_log(f"相机扫描失败：{notice.error}")
            if notice.detail:
                self._append_log(notice.detail.rstrip())
            if self._window_exists(self.camera_window):
                messagebox.showerror(
                    "相机扫描失败", _concise_error(notice.error), parent=self.root
                )
        else:
            self._camera_devices = list(notice.devices)
            self._camera_warnings = [str(item) for item in notice.warnings if str(item)]
            count = len(self._camera_devices)
            warning_count = len(self._camera_warnings)
            self.camera_status_var.set(
                f"发现 {count} 台设备"
                + (f" · {warning_count} 条提示" if warning_count else " · 扫描完成")
            )
            self.camera_warning_var.set(
                "\n".join(f"• {item}" for item in self._camera_warnings)
            )
            self._append_log(
                f"相机扫描完成：{count} 台设备，{warning_count} 条提示。"
            )
            for warning in self._camera_warnings:
                self._append_log(f"相机提示：{warning}")
        self._rebuild_camera_list()
        if not notice.error:
            self._maybe_auto_select_camera()

    def _rebuild_camera_list(self) -> None:
        if not self._window_exists(getattr(self, "camera_list_frame", None)):
            return
        for child in self.camera_list_frame.winfo_children():
            child.destroy()
        self._camera_device_buttons.clear()
        if not self._camera_devices:
            ctk.CTkLabel(
                self.camera_list_frame,
                text=(
                    "正在扫描设备…"
                    if self._camera_discovering
                    else "尚未发现设备\n\n点击“刷新设备”扫描 USB 与海康 MVS 相机。"
                ),
                font=self._font(11),
                text_color=(_LIGHT["muted"], _DARK["muted"]),
                justify=tk.CENTER,
            ).pack(expand=True, fill=tk.BOTH, padx=20, pady=60)
            return

        self._available_camera_sources = {
            device.source for device in self._camera_devices if device.available
        }
        for device in self._camera_devices:
            card = ctk.CTkFrame(
                self.camera_list_frame,
                corner_radius=11,
                fg_color=(_LIGHT["card_soft"], _DARK["card_soft"]),
                border_color=(_LIGHT["border"], _DARK["border"]),
                border_width=1,
            )
            card.pack(fill=tk.X, padx=4, pady=4)
            top = ctk.CTkFrame(card, fg_color="transparent")
            top.pack(fill=tk.X, padx=12, pady=(11, 3))
            ctk.CTkLabel(
                top,
                text=device.name or "未命名相机",
                font=self._font(12, "bold"),
                text_color=(_LIGHT["text"], _DARK["text"]),
                anchor=tk.W,
                wraplength=340,
                justify=tk.LEFT,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)
            ctk.CTkLabel(
                top,
                text=_KIND_NAMES.get(device.kind, device.kind.upper()),
                font=self._font(9, "bold"),
                text_color=(_LIGHT["accent"], "#8ECCF4"),
                fg_color=(_LIGHT["accent_soft"], _DARK["accent_soft"]),
                corner_radius=6,
                padx=7,
                pady=3,
            ).pack(side=tk.RIGHT, padx=(6, 0))
            detail = device.detail or "无更多设备信息"
            ctk.CTkLabel(
                card,
                text=detail,
                font=self._font(10),
                text_color=(_LIGHT["muted"], _DARK["muted"]),
                anchor=tk.W,
                justify=tk.LEFT,
                wraplength=400,
            ).pack(fill=tk.X, padx=12, pady=(2, 7))
            button = ctk.CTkButton(
                card,
                text="选择此设备" if device.available else "设备不可用",
                command=lambda item=device: self._select_camera_device(item),
                state=tk.NORMAL if device.available else tk.DISABLED,
                height=28,
                corner_radius=7,
                font=self._font(10),
                fg_color="transparent",
                hover_color=(_LIGHT["accent_soft"], _DARK["accent_soft"]),
                text_color=(_LIGHT["accent"], "#8ECCF4"),
            )
            button.pack(anchor=tk.E, padx=10, pady=(0, 8))
            self._camera_device_buttons[device.source] = button

    def _select_camera_device(self, device: CameraDevice) -> None:
        source = str(device.source).strip()
        if source.isdigit():
            source = f"usb://{int(source)}"
        if not source:
            return
        new_key = normalise_camera_source(source)
        if self._camera_stream is not None and self._camera_stream_source != new_key:
            self._release_camera_stream()
        self.source_var.set(source)
        if device.serial:
            self.camera_serial_var.set(device.serial)
        self.camera_selected_name_var.set(device.name or source)
        self.camera_selected_source_var.set(source)
        self.camera_selected_detail_var.set(
            device.detail or _KIND_NAMES.get(device.kind, device.kind)
        )
        self.camera_status_var.set(f"已选择 · {device.name or source}")
        self.status_var.set(f"已选择相机：{device.name or source}")
        # Reset every button first: previously only the clicked one was updated,
        # so multiple cameras ended up showing "✓ 已选择" at the same time.
        for button_source, button in self._camera_device_buttons.items():
            if not self._window_exists(button):
                continue
            label = (
                "✓ 已选择"
                if button_source == device.source
                else ("选择此设备" if button_source in self._available_camera_sources else "设备不可用")
            )
            button.configure(text=label)
        self._start_camera_preconnect()

    def _use_manual_camera(self) -> None:
        # Manual camera entry (index / serial / GenICam) has been removed.
        # Cameras are selected exclusively from the enumerated device list.
        messagebox.showinfo(
            "请从设备列表选择",
            "请在相机管理中刷新设备，并直接点击列表里的相机型号进行选择。",
            parent=self.root,
        )

    def _mode_value(self) -> str:
        """Return "photo" or "live" for the current mode selector."""
        return "live" if "实时" in self.camera_mode_var.get() else "photo"

    # ------------------------------------------------------------ Validation
    def _validated_settings(self) -> Dict[str, Any]:
        model_text = self.model_var.get().strip()
        model_path = Path(model_text).expanduser() if model_text else None
        if model_path is None or not model_path.is_file():
            raise ValueError("请选择存在的模型文件。")
        if model_path.suffix.lower() not in (".pt", ".onnx"):
            raise ValueError("模型格式仅支持 .pt 或 .onnx。")

        source_text = self.source_var.get().strip()
        if not source_text:
            raise ValueError("请先在“相机管理”中选择一个相机。")
        if not is_camera_source(source_text):
            raise ValueError("当前仅支持摄像头采集，请通过“相机管理”选择相机。")
        source: Any = (
            int(source_text) if source_text.isdigit() else source_text
        )

        output_text = self.output_var.get().strip()
        if not output_text:
            raise ValueError("请选择输出目录。")
        # Probe writability before model loading or opening a camera.
        output_path = ensure_writable_directory(Path(output_text).expanduser())

        try:
            conf = float(self.conf_var.get().strip())
        except ValueError as exc:
            raise ValueError("置信度必须是数字。") from exc
        if not 0.0 <= conf <= 1.0:
            raise ValueError("置信度必须在 0 到 1 之间。")
        try:
            iou = float(self.iou_var.get().strip())
        except ValueError as exc:
            raise ValueError("IoU 必须是数字。") from exc
        if not 0.0 <= iou <= 1.0:
            raise ValueError("IoU 必须在 0 到 1 之间。")
        try:
            imgsz = int(self.imgsz_var.get().strip())
        except ValueError as exc:
            raise ValueError("图像尺寸必须是整数。") from exc
        if imgsz < 32:
            raise ValueError("图像尺寸必须大于或等于 32。")

        camera_options = (
            self._camera_options() if is_camera_source(source) else CameraOptions()
        )
        return {
            "model": str(model_path),
            "names": "",
            "source": source,
            "output": str(output_path),
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "device": self.device_var.get().strip() or "auto",
            "save_json": bool(self.save_json_var.get()),
            "mode": self._mode_value(),
            "camera_options": camera_options,
        }

    # --------------------------------------------------------- Worker control
    def start_processing(self) -> None:
        """Validate settings and start one worker; repeated starts are blocked."""
        if self._running or self._closing:
            return
        try:
            settings = self._validated_settings()
            base_output = ensure_writable_directory(settings["output"])
            # This must remain on the main thread and before model/camera startup.
            run_output = (
                create_run_output_directory(base_output)
                if bool(self.unique_output_var.get())
                else ensure_writable_directory(base_output)
            )
            settings["output"] = str(run_output)
        except (ValueError, OSError) as exc:
            messagebox.showerror("无法开始检测", str(exc), parent=self.root)
            return

        self.last_output_dir = run_output
        self._save_settings()
        self._drain_queue(self._control_queue)
        self._drain_queue(self._preview_queue)
        self._run_id += 1
        self._run_failed = False
        self._cancel_event = threading.Event()
        self._running = True
        self._set_running_state(True)
        self._reset_progress()
        self._clear_preview("正在准备检测…")
        self.model_description_var.set("正在加载模型…")
        self.status_var.set("正在加载模型，请稍候…")
        self._append_log(
            f"开始任务：来源={settings['source']}，输出={settings['output']}"
        )

        run_id = self._run_id
        cancel_event = self._cancel_event
        self._worker = threading.Thread(
            target=self._worker_main,
            args=(run_id, settings, cancel_event),
            name="yolov8-inference",
            daemon=True,
        )
        self._worker.start()

    def stop_processing(self) -> None:
        """Request cooperative cancellation; never forcibly kill the worker."""
        if not self._running or self._cancel_event is None:
            return
        if not self._cancel_event.is_set():
            self._cancel_event.set()
            self.stop_button.configure(state=tk.DISABLED)
            self.status_var.set("正在停止，等待当前推理结束…")
            self._append_log("已请求停止，正在安全结束当前任务。")

    def _release_camera_stream(self) -> None:
        """Release the persistent stream (never called on the UI hot path)."""
        with self._camera_stream_lock:
            stream = self._camera_stream
            self._camera_stream = None
            self._camera_stream_source = None
            if stream is not None:
                try:
                    stream.release()
                except Exception:
                    pass

    def _get_or_open_camera_stream(self, source: Any, options: CameraOptions) -> Any:
        """Return an existing stream or open it once in a background thread."""
        key = normalise_camera_source(source)
        preconnect = self._camera_preconnect_thread
        if (
            preconnect is not None
            and preconnect is not threading.current_thread()
            and preconnect.is_alive()
        ):
            # Let the selection-time preconnect finish instead of opening twice.
            preconnect.join(timeout=max(15, options.timeout_ms / 1000 + 10))
        with self._camera_stream_lock:
            if self._camera_stream is not None and self._camera_stream_source == key:
                return self._camera_stream
            old_stream = self._camera_stream
            self._camera_stream = None
            self._camera_stream_source = None
            if old_stream is not None:
                try:
                    old_stream.release()
                except Exception:
                    pass
            self._camera_stream = open_camera(source, options)
            self._camera_stream_source = key
            self._camera_preconnect_error = None
            if self._closing:
                # A close racing with preconnect must not leave the device open.
                self._release_camera_stream()
                raise RuntimeError("程序正在关闭，已取消相机连接。")
            return self._camera_stream

    def _start_camera_preconnect(self) -> None:
        """Pre-open the selected camera to hide initialization from the first photo."""
        if self._closing:
            return
        source = self.source_var.get().strip()
        if not source or not is_camera_source(source):
            return
        if self._camera_stream is not None and self._camera_stream_source == normalise_camera_source(source):
            return
        if self._camera_preconnect_thread is not None and self._camera_preconnect_thread.is_alive():
            return
        try:
            options = self._camera_options()
        except ValueError:
            return

        def work() -> None:
            try:
                self._get_or_open_camera_stream(source, options)
            except Exception as exc:
                self._camera_preconnect_error = exc

        self._camera_preconnect_thread = threading.Thread(
            target=work, name="camera-preconnect", daemon=True
        )
        self._camera_preconnect_thread.start()

    def _worker_main(
        self,
        run_id: int,
        settings: Dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        generator: Any = None
        try:
            # Reuse the loaded model: ONNX loading costs ~3 s and its first
            # inference ~2 s, so a repeated photo must not pay that again.
            engine_key = (
                settings["model"],
                settings["names"],
                settings["conf"],
                settings["iou"],
                settings["imgsz"],
                settings["device"],
            )
            engine = self._engine
            if engine is None or engine_key != self._engine_key:
                engine = YOLOv8Engine(
                    model_path=settings["model"],
                    names_path=settings["names"] or None,
                    conf=settings["conf"],
                    iou=settings["iou"],
                    imgsz=settings["imgsz"],
                    device=settings["device"],
                )
                self._engine = engine
                self._engine_key = engine_key
                # Warm the backend once so the first real photo is not slow.
                try:
                    engine.warmup()
                except Exception:
                    pass
            self._post_control(
                run_id, _WorkerNotice("model", str(engine.model_description))
            )
            if cancel_event.is_set():
                return

            camera_stream = None
            if is_camera_source(settings["source"]):
                camera_stream = self._get_or_open_camera_stream(
                    settings["source"], settings["camera_options"]
                )
            generator = process_source(
                engine,
                settings["source"],
                settings["output"],
                save_json=settings["save_json"],
                cancel_event=cancel_event,
                camera_options=settings["camera_options"],
                mode=settings["mode"],
                stream=camera_stream,
                keep_stream=camera_stream is not None,
            )
            for event in generator:
                self._post_run_event(run_id, event)
                if self._closing:
                    return
                # Cooperative cancellation is handled inside process_source.
                # Keep consuming events so it can release the camera, save the
                # photo event before completion.
        except Exception as exc:  # inference failures must return to the GUI
            # A disconnected device must not be reused on the next attempt.
            self._release_camera_stream()
            self._post_control(
                run_id,
                _WorkerNotice(
                    "error", str(exc) or repr(exc), traceback.format_exc()
                ),
            )
        finally:
            if generator is not None:
                try:
                    generator.close()
                except Exception:
                    pass
            self._post_control(run_id, _WorkerNotice("done"))

    def _post_run_event(self, run_id: int, event: Any) -> None:
        image = getattr(event, "image", None)
        if image is not None:
            self._post_preview(run_id, _PreviewNotice(image))
            event = copy.copy(event)
            event.image = None
        self._post_control(run_id, event)

    def _envelope(self, run_id: int, payload: Any) -> Tuple[int, int, Any]:
        return run_id, next(self._sequence), payload

    def _post_control(self, run_id: int, payload: Any) -> None:
        item = self._envelope(run_id, payload)
        while not self._closing:
            try:
                self._control_queue.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    def _post_preview(self, run_id: int, payload: Any) -> None:
        item = self._envelope(run_id, payload)
        while not self._closing:
            try:
                self._preview_queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._preview_queue.get_nowait()
                except queue.Empty:
                    pass

    # --------------------------------------------------------- Main-thread UI
    def _poll_worker_queues(self) -> None:
        """Drain worker/camera queues; only this Tk thread touches widgets."""
        if self._closing:
            return
        try:
            self._refresh_palette()
            controls = []
            while True:
                try:
                    controls.append(self._control_queue.get_nowait())
                except queue.Empty:
                    break
            latest_preview = None
            while True:
                try:
                    latest_preview = self._preview_queue.get_nowait()
                except queue.Empty:
                    break

            pending = controls + ([latest_preview] if latest_preview else [])
            pending.sort(key=lambda item: item[1])
            for run_id, _sequence, payload in pending:
                if run_id != self._run_id or self._closing:
                    continue
                try:
                    if isinstance(payload, _WorkerNotice):
                        self._handle_worker_notice(payload)
                    elif isinstance(payload, _PreviewNotice):
                        self._set_preview_image(payload.image)
                    else:
                        self._handle_run_event(payload)
                except Exception as exc:
                    self._append_log(f"界面事件更新失败：{exc}")

            while True:
                try:
                    notice = self._camera_queue.get_nowait()
                except queue.Empty:
                    break
                self._handle_camera_notice(notice)
        except Exception as exc:
            try:
                self._append_log(f"界面更新失败：{exc}")
            except tk.TclError:
                pass
        finally:
            if not self._closing:
                self._poll_after_id = self.root.after(
                    self.POLL_INTERVAL_MS, self._poll_worker_queues
                )

    def _handle_worker_notice(self, notice: _WorkerNotice) -> None:
        if notice.kind == "model":
            self.model_description_var.set(notice.message)
            self._append_log(f"模型加载完成：{notice.message}")
            self.status_var.set("模型已加载，开始处理…")
            return
        if notice.kind == "error":
            self._run_failed = True
            self._append_log(f"运行错误：{notice.message}")
            if notice.detail:
                self._append_log(notice.detail.rstrip())
            messagebox.showerror(
                "检测失败", _concise_error(notice.message), parent=self.root
            )
            return
        if notice.kind == "done":
            cancelled = bool(
                self._cancel_event is not None and self._cancel_event.is_set()
            )
            self._running = False
            self._worker = None
            self._set_running_state(False)
            self.progress.stop()
            self._progress_mode = "determinate"
            self.progress.configure(mode="determinate")
            if self._run_failed:
                self.status_var.set("运行失败 · 请查看日志并检查参数")
            elif cancelled:
                self.status_var.set("已停止 · 后台资源已释放")
                self._append_log("任务已停止，后台资源已释放。")
            else:
                self.status_var.set("处理完成 · 可以打开输出目录查看结果")
            self._save_settings()

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return 0

    def _handle_run_event(self, event: Any) -> None:
        kind = str(getattr(event, "kind", ""))
        total = self._safe_int(getattr(event, "total", 0))
        current = self._safe_int(getattr(event, "current", 0))
        frame_index = self._safe_int(getattr(event, "frame_index", 0))
        message = str(getattr(event, "message", "") or "")
        image = getattr(event, "image", None)
        if image is not None:
            self._set_preview_image(image)

        if kind == "started":
            if total > 0:
                self._set_determinate(total, 0)
                self.status_var.set(f"开始处理，共 {total} 张图片")
            else:
                self._start_indeterminate()
                self.status_var.set("正在连接相机拍照…")
            self._append_log(message or f"开始处理：{getattr(event, 'source', '')}")
        elif kind == "image":
            if total > 0:
                self._set_determinate(total, current)
            self.status_var.set(message or f"已完成 {current}/{total} 张图片")
            prefix = f"图片 {Path(str(getattr(event, 'source', ''))).name}"
            self._log_detections(event, prefix)
            self._log_saved_path(getattr(event, "saved_path", ""))
        elif kind == "camera_started":
            self._start_indeterminate()
            self.status_var.set(message or "相机已连接，正在拍照检测…")
            self._append_log(message or "开始相机拍照检测")
        elif kind == "frame":
            if total > 0:
                self._set_determinate(total, current)
            elif self._progress_mode != "indeterminate":
                self._start_indeterminate()
            inference_ms = float(getattr(event, "inference_ms", 0.0) or 0.0)
            self.status_var.set(
                f"实时识别中 · 第 {frame_index} 帧 · {inference_ms:.0f} ms"
            )
            detections = getattr(event, "detections", None) or []
            if detections and (frame_index == 1 or frame_index % 30 == 0):
                self._log_detections(event, f"第 {frame_index} 帧")
        elif kind == "camera_finished":
            self.progress.stop()
            self.status_var.set(message or f"实时识别结束，共 {frame_index} 帧")
            self._append_log(message or "实时识别结束")
            self._log_saved_path(getattr(event, "saved_path", ""), "最后一帧")
        elif kind == "photo":
            self.progress.stop()
            if total > 0:
                self._set_determinate(total, current)
            inference_ms = float(getattr(event, "inference_ms", 0.0) or 0.0)
            self.status_var.set(
                message or f"拍照检测完成，推理 {inference_ms:.1f} ms"
            )
            self._log_detections(event, "拍照检测")
            self._log_saved_path(getattr(event, "saved_path", ""), "照片")
        elif kind == "completed":
            self.progress.stop()
            if total > 0:
                self._set_determinate(total, current)
                if not (
                    self._cancel_event is not None and self._cancel_event.is_set()
                ):
                    self.progress.configure(value=total)
            self.status_var.set(message or "处理完成")
            self._append_log(message or "处理完成")
            self._log_saved_path(getattr(event, "saved_path", ""), "输出目录")
        else:
            if message:
                self._append_log(message)
            self._log_saved_path(getattr(event, "saved_path", ""))

    def _log_detections(self, event: Any, prefix: str) -> None:
        detections = list(getattr(event, "detections", None) or [])
        inference_ms = float(getattr(event, "inference_ms", 0.0) or 0.0)
        if not detections:
            self._append_log(f"{prefix}：未检测到目标，推理 {inference_ms:.1f} ms")
            return
        self._append_log(
            f"{prefix}：检测到 {len(detections)} 个目标，推理 {inference_ms:.1f} ms"
        )
        for detection in detections[:100]:
            class_id = getattr(detection, "class_id", -1)
            name = str(getattr(detection, "name", "未知类别"))
            confidence = float(getattr(detection, "confidence", 0.0))
            box = getattr(detection, "box", ())
            try:
                box_text = ", ".join(f"{float(value):.1f}" for value in box)
            except (TypeError, ValueError):
                box_text = str(box)
            self._append_log(
                f"  类别 {name}（ID {class_id}），置信度 {confidence:.2%}，"
                f"位置 [{box_text}]"
            )
        if len(detections) > 100:
            self._append_log(f"  ……其余 {len(detections) - 100} 个目标从略")

    def _log_saved_path(self, path: Any, label: str = "保存路径") -> None:
        if path:
            self._append_log(f"{label}：{path}")

    def _set_running_state(self, running: bool) -> None:
        self.start_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_button.configure(state=tk.NORMAL if running else tk.DISABLED)
        self.camera_manage_button.configure(
            state=tk.DISABLED if running else tk.NORMAL
        )
        if self._window_exists(getattr(self, "camera_refresh_button", None)):
            self.camera_refresh_button.configure(
                state=(
                    tk.DISABLED
                    if running or self._camera_discovering
                    else tk.NORMAL
                )
            )

    # -------------------------------------------------------------- Progress
    def _reset_progress(self) -> None:
        self.progress.stop()
        self._progress_mode = "determinate"
        self.progress.configure(mode="determinate", maximum=1, value=0)

    def _set_determinate(self, total: int, current: int) -> None:
        if total <= 0:
            return
        if self._progress_mode == "indeterminate":
            self.progress.stop()
        self._progress_mode = "determinate"
        self.progress.configure(
            mode="determinate",
            maximum=max(1, total),
            value=max(0, min(current, total)),
        )

    def _start_indeterminate(self) -> None:
        if self._progress_mode != "indeterminate":
            self._progress_mode = "indeterminate"
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)

    # --------------------------------------------------------------- Preview
    def _on_preview_resize(self, _event: Any = None) -> None:
        if self._closing:
            return
        if self._resize_after_id is not None:
            try:
                self.root.after_cancel(self._resize_after_id)
            except tk.TclError:
                pass
        self._resize_after_id = self.root.after(40, self._render_preview)

    def _clear_preview(self, text: str = "等待检测画面") -> None:
        self._preview_message = text
        self._preview_source = None
        self._preview_photo = None
        self._render_preview()

    def _set_preview_image(self, image_bgr: Any) -> None:
        try:
            if isinstance(image_bgr, Image.Image):
                image = image_bgr.convert("RGB")
            else:
                shape = getattr(image_bgr, "shape", ())
                if len(shape) == 2:
                    image = Image.fromarray(image_bgr).convert("RGB")
                elif len(shape) == 3 and int(shape[2]) >= 3:
                    image = Image.fromarray(
                        image_bgr[:, :, :3][:, :, ::-1].copy()
                    ).convert("RGB")
                else:
                    raise ValueError("预览图像形状无效")
        except Exception as exc:
            self._append_log(f"无法显示预览：{exc}")
            return
        self._preview_source = image
        self._render_preview()

    def _render_preview(self) -> None:
        self._resize_after_id = None
        if self._closing:
            return
        canvas = self.preview_canvas
        canvas.delete("all")
        if self._preview_source is None:
            self._preview_photo = None
            center_x = max(1, canvas.winfo_width() // 2)
            center_y = max(1, canvas.winfo_height() // 2)
            canvas.create_text(
                center_x,
                center_y - 26,
                text="◇",
                fill="#557086",
                font=(self.font_family, 30),
            )
            canvas.create_text(
                center_x,
                center_y + 18,
                text=self._preview_message,
                fill="#D4DEE7",
                font=(self.font_family, 14, "bold"),
            )
            canvas.create_text(
                center_x,
                center_y + 47,
                text="图片或相机拍照结果将显示在这里",
                fill="#7F919F",
                font=(self.font_family, 10),
            )
            return

        width = max(2, canvas.winfo_width() - 12)
        height = max(2, canvas.winfo_height() - 12)
        source_width, source_height = self._preview_source.size
        if source_width <= 0 or source_height <= 0:
            return
        scale = min(width / source_width, height / source_height)
        target_size = (
            max(1, int(source_width * scale)),
            max(1, int(source_height * scale)),
        )
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        resized = self._preview_source.resize(target_size, resampling)
        self._preview_photo = ImageTk.PhotoImage(resized, master=self.root)
        canvas.create_image(
            canvas.winfo_width() // 2,
            canvas.winfo_height() // 2,
            image=self._preview_photo,
            anchor=tk.CENTER,
        )

    # ------------------------------------------------------------------- Log
    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        lines = str(message).splitlines() or [""]
        text = "\n".join(
            f"[{timestamp}] {line}" if index == 0 else f"           {line}"
            for index, line in enumerate(lines)
        )
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + "\n")
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > self.MAX_LOG_LINES:
            self.log_text.delete("1.0", f"{line_count - self.MAX_LOG_LINES}.0")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # -------------------------------------------------------------- Settings
    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _read_settings(self) -> Dict[str, Any]:
        try:
            data = json.loads(self._settings_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _settings_for_json(self) -> Dict[str, Any]:
        def number(variable: tk.StringVar, converter: Any) -> Any:
            value = variable.get().strip()
            try:
                return converter(value)
            except (TypeError, ValueError):
                return value

        return {
            "appearance": self.appearance_var.get(),
            "model": _make_path_portable(self.model_var.get()),
            "source": _make_path_portable(self.source_var.get()),
            # Deliberately save the base, never last_output_dir / run_* path.
            "output": _make_path_portable(self.output_var.get()),
            "conf": number(self.conf_var, float),
            "iou": number(self.iou_var, float),
            "imgsz": number(self.imgsz_var, int),
            "device": self.device_var.get().strip(),
            "save_json": bool(self.save_json_var.get()),
            "camera_mode": self._mode_value(),
            "unique_output": bool(self.unique_output_var.get()),
            "camera": {"serial": self.camera_serial_var.get().strip()},
        }

    def _save_settings(self) -> None:
        try:
            self._settings_path.write_text(
                json.dumps(
                    self._settings_for_json(), ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )
        except (OSError, ValueError, TypeError) as exc:
            if hasattr(self, "log_text") and not self._closing:
                self._append_log(f"设置保存失败（不影响检测）：{exc}")

    # --------------------------------------------------------------- Cleanup
    @staticmethod
    def _drain_queue(target: queue.Queue) -> None:
        while True:
            try:
                target.get_nowait()
            except queue.Empty:
                return

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._camera_generation += 1
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._save_settings()
        for after_id in (self._poll_after_id, self._resize_after_id):
            if after_id is not None:
                try:
                    self.root.after_cancel(after_id)
                except tk.TclError:
                    pass
        for window in (self.camera_window,):
            if window is not None:
                try:
                    window.destroy()
                except tk.TclError:
                    pass
        self.camera_window = None
        self._release_camera_stream()
        self._preview_photo = None
        self._preview_source = None
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def run(self) -> None:
        """Enter the Tk event loop."""
        self.root.mainloop()


def run_gui() -> None:
    """Create and run the YOLOv8 desktop GUI (called by ``app.py``)."""
    app = YOLOv8DesktopApp()
    app.run()


if __name__ == "__main__":
    run_gui()
