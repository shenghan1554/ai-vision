# 摄像头与工业相机接入指南

本版本仅处理 **USB 免驱相机与海康 MVS 工业相机的拍照或实时检测**。图片文件/目录输入、视频、录像、类别名称及 GenICam 接入均已移除。

## 1. 来源标识

| 设备 | 来源示例 | 说明 |
| --- | --- | --- |
| Windows USB/UVC 免驱相机 | `usb://0` | 也兼容直接输入 `0` |
| 海康 MVS 相机 | `hik://序列号` | 界面按型号选择，内部按序列号定位 |

拍照模式每次保存一张 `YYYYMMDD_HHMMSS_mmm.jpg`；实时模式在停止时保存最后一帧。勾选结构化结果时保存同名 `.json`。不会生成视频或逐帧 JSONL。同一毫秒的结果自动追加 `_001` 等序号。

## 2. USB 免驱摄像头

Windows 通过 UVC/`usbvideo` 驱动接入，无需厂商专用驱动。

程序自动处理 USB 后端与格式：

- 自动在 DirectShow、Media Foundation 和 OpenCV 默认后端间回退，并缓存上次成功的后端；
- GUI 使用持久相机连接和后台持续取帧缓存，拍照直接读取最新帧；
- 不再强制 MJPG（该项在本机 DirectShow 上单次设置耗时约 1.2 秒）；
- 可选分辨率、帧率、曝光和增益；
- 不在打开构造函数中强制传入 `CAP_PROP_OPEN_TIMEOUT_MSEC`（属性 53），避免部分 MSMF 驱动报 `openFinalize ... can't set property 53`；
- 扫描后快速重开支持短间隔重试，MJPG/自定义参数失败时自动回退到驱动默认格式；
- 打开后实际读取一帧验证，避免“设备已打开但没有画面”；
- 偶尔读取超时自动容忍，连续失败才终止；
- 自动释放设备，供其他软件继续使用。

本机实测设备：`ov9734_techfront_camera`，Windows 服务为 `usbvideo`。DirectShow 下连续读取 30 帧成功；修复属性 53 后，Media Foundation 也连续读取 30 帧成功，实际输出均为 640×480。两种后端及 auto 的快速连续打开/关闭压力测试均通过。

### 打不开时

1. 关闭 MVS、相机助手、Zoom、Teams、浏览器相机页面等占用程序。
2. 检查 Windows“设置 → 隐私和安全性 → 相机”，允许桌面应用访问。
3. 在设备管理器确认设备为“照相机”且服务为 `usbvideo`。
4. 在界面将后端从 `auto` 改为 `dshow` 或 `msmf` 分别测试。
5. 降低分辨率或帧率，取消 MJPG 后再试。
6. 更换 USB 接口/线缆，避免供电不足的集线器。

手动诊断：

```bash
python tests/probe_usb.py --source 0 --backend auto --frames 30
python tests/probe_usb.py --enumerate
```

## 3. 海康 MVS / GigE Vision / USB3 Vision

### 安装要求

安装海康机器人官方 **MVS 客户端及工业相机 SDK**，需要包含：

- MVS Runtime；
- GigE Vision / USB3 Vision 驱动和 Transport Layer；
- Python Samples 中的 `MvCameraControl_class.py` 及配套模块。

本机已检测到：

- MVS 5.1.0；
- 工业相机 SDK 4.8.2.2；
- Python SDK：`C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport`；

程序动态加载官方 SDK，不复制或再分发海康专有 DLL。升级 MVS 后无需重新打包 EXE，但 Python/SDK 位数必须一致，当前交付为 64 位。

v2.1.1 会先通过 `os.add_dll_directory`、依赖目录 `PATH` 和 DLL 绝对路径预加载，再执行官方封装的 `WinDLL("MvCameraControl.dll")`，解决 PyInstaller 冻结版找不到 DLL 的问题。非默认安装位置可设置：

```text
MVS_RUNTIME_PATH=MVS Runtime\Win64_x64 的实际目录
MVS_SDK_PYTHON_PATH=MvCameraControl_class.py 所在的 MvImport 目录
```

`MVCAM_COMMON_RUNENV` 指向 `MVS\Development` 时会按该层级处理，不再生成 `Development\Development` 重复路径。

### GigE 网络检查

1. 在 MVS 客户端中先确认相机能枚举、取图。
2. 相机 IP 与采集网卡 IP 必须在同一网段，且不能与其他设备冲突。
3. 使用独立有线网卡/直连时，按现场网络规划设置静态 IP。
4. 检查 Windows 防火墙、网卡驱动和 GigE Vision 过滤驱动。
5. 高分辨率/高帧率下出现丢帧时，再在 MVS 中调整包长、Packet Delay、心跳和网卡巨帧；不要一开始就盲目开启巨帧。
6. 关闭 MVS 客户端的取图后再从本程序打开，同一个相机通常不能由两个进程独占采集。

界面中直接点击相机型号即可；命令行使用序列号：

```bash
python detect.py --source hik://055062320032 --model models/yolov8n.pt --save-json
```

未连接设备时，SDK 枚举返回空列表属于正常情况，不代表 SDK 安装失败。

### 采集参数

界面中的“采集设置”面板已删除，采集参数由程序自动处理。如需精细控制，可在命令行使用 `--exposure`、`--gain` 等参数。

## 4. 枚举与命令行诊断

```bash
python detect.py --list-cameras
```

仅枚举前两个 USB 索引，减少探测时间：

```bash
python detect.py --list-cameras --max-usb-index 2
```

USB 后端和参数：

```bash
python detect.py --source usb://0 --camera-backend dshow --camera-width 1280 --camera-height 720 --camera-fps 30
```

输出到独立目录：

```bash
python detect.py --source usb://0 --run-subdir --save-json
```

曝光与增益（每次只拍一张照片）：

```bash
python detect.py --source usb://0 --exposure 2000 --gain 2 --run-subdir --save-json
```

按 `Ctrl+C` 结束命令行实时识别。界面则使用“停止”，会等待当前帧推理结束并释放设备。

## 5. 输出权限

开始任务前程序会真实创建并删除一个测试文件，检查输出目录是否可写。每次任务默认创建 `run_日期_时间` 子目录，避免覆盖旧结果。

如果桌面目录受 Windows“受控文件夹访问权限”保护：

- 改选程序目录下的 `output` 或其他可写目录；
- 或在 Windows 安全中心中按组织安全策略授权；
- 不要通过以管理员身份长期运行识别程序来绕过权限设计。

图片和 JSON 使用临时文件 + 原子替换，写入失败会保留原文件并显示具体路径和系统错误。

## 6. 验证范围说明

单元测试会模拟设备枚举、像素转换、读取失败和资源释放。真实测试只证明当前电脑连接的设备可用：本机 USB 免驱相机可正常拍照；本机海康 GigE 相机可通过 MVS SDK 拍照取图。该结果不代表其他型号或网络环境都已验证。现场仍应记录相机型号、序列号、MVS/SDK 版本、IP 配置和拍照结果。
