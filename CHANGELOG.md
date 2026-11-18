# Changelog

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [2.6.0] — 2026-11-16

### 变更

- 产品统一更名为 **AI Vision**：程序名、窗口标题、EXE 文件名（`AIVision.exe`）、安装包名同步更新
- **删除「采集设置」面板**：原有参数已失效，采集参数全部自动化（后端、超时、MJPG、分辨率、帧率）
- **修复相机多选 Bug**：此前多次点击会让多台相机同时显示「已选择」，现在任何时刻只有一台处于选中状态
- 移除模型信息等说明性小字
- 新增安装版（Inno Setup）与便携版两种交付形态

### 保留

- v2.5 的相机持久连接与后台最新帧缓存（拍照直接取最新帧，USB 后续拍照约 0.12–0.16 s）
- PT / ONNX 双后端、自动扫描并选中首个相机、按型号切换

## [2.5.0]

- 相机持久连接、后台最新帧缓存，显著降低拍照延迟
- 拍照识别时 GUI 保持相机连接，便于连续拍摄

## [2.1.1]

- 冻结 EXE 中改为通过绝对路径预加载 `MvCameraControl.dll`，修复找不到海康 DLL 的问题
- 支持 `MVS_RUNTIME_PATH` / `MVS_SDK_PYTHON_PATH` 自定义路径

## [2.1.0]

- 不再向 UVC 驱动传入 `CAP_PROP_OPEN_TIMEOUT_MSEC`（属性 53），修复部分 MSMF 设备打开失败
- 支持短间隔重开重试；MJPG/自定义参数不被接受时回退驱动默认格式
- 相机列表改为只显示型号，点击型号即选择

## [2.0.0] 及更早

- YOLOv8 detect 目标检测，PT / ONNX 双后端
- USB 免驱相机（DirectShow / Media Foundation）与海康 MVS 工业相机
- 三栏界面、拍照/实时双模式、独立 `run_*` 输出目录、原子写入
