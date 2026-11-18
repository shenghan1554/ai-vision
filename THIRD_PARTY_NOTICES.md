# 第三方组件与模型许可提示

**本项目以 AGPL-3.0 发布（见 [LICENSE](LICENSE)）**，原因是使用 Ultralytics YOLOv8（AGPL-3.0）
并随仓库分发其预训练模型。分发或对外提供服务时须提供完整对应源码，或购买 Ultralytics 企业许可。

本项目未修改第三方库的名称或许可证。分发 EXE 便携目录时，请保留 `_internal` 中由依赖包附带的许可证文件，并自行核对实际安装版本对应的完整许可文本。

主要组件：

| 组件 | 用途 | 许可信息入口 |
| --- | --- | --- |
| Ultralytics | YOLOv8 模型加载、预处理、推理、NMS 与标注 | https://www.ultralytics.com/license |
| PyTorch | PT 模型推理 | https://github.com/pytorch/pytorch/blob/main/LICENSE |
| ONNX / ONNX Runtime | ONNX 模型解析与推理 | https://github.com/microsoft/onnxruntime/blob/main/LICENSE |
| OpenCV | 图片/视频读写 | https://github.com/opencv/opencv/blob/4.x/LICENSE |
| NumPy / Pillow | 数组及桌面预览 | 各依赖包内许可证 |
| CustomTkinter | 现代桌面控件与主题 | https://github.com/TomSchimansky/CustomTkinter/blob/master/LICENSE |

## Ultralytics 特别说明

Ultralytics 提供 AGPL-3.0 及企业许可等选项。若你的应用、模型或部署场景涉及商业、闭源或网络服务，请阅读其官方许可条款并选择适当授权；本提示本身不构成法律意见，也不替代完整许可证。

官方资源：

- https://github.com/ultralytics/ultralytics
- https://www.ultralytics.com/license

## 海康 MVS SDK

海康 MVS Runtime、SDK、驱动和 `.cti` 文件属于海康机器人官方组件。本程序仅从用户已安装的官方目录动态加载，不复制或再分发其专有 DLL。部署、升级和分发时请遵守对应 MVS/SDK 许可及现场授权要求。

## 内置模型

- `models/yolov8n.pt`
- `models/yolov8n.onnx`

均对应 Ultralytics YOLOv8n COCO 预训练模型，用于功能示例。PT 官方发布地址：

https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt

示例图 `samples/bus.jpg` 为参考包及 Ultralytics 常用示例。替换为自己的模型和图片时，请同时确认其数据来源与使用授权。
