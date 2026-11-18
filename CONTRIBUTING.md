# 贡献指南

感谢你愿意改进 **AI Vision**。这是一个完全本地运行的 Windows 桌面目标检测程序：
相机采集 + Ultralytics YOLOv8 推理，CPU 即可运行，不依赖云服务。

## 行为准则

参与本项目即表示你同意保持友善、就事论事的交流方式。

## 环境搭建

| 依赖 | 版本 | 说明 |
| --- | --- | --- |
| Windows | 10 / 11 x64 | 相机后端依赖 Win32（DirectShow / Media Foundation / MVS） |
| Python | 3.10–3.12（64 位） | |
| 磁盘 | 约 3 GB | PyTorch + Ultralytics + 虚拟环境 |

```bat
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python app.py
```

或者直接双击 `run_dev.bat`（优先用 `.venv`）。

> 只改代码不跑推理时，`python -m compileall -q yolo_app tests detect.py app.py` 就够做语法自检。

## 代码地图

```
app.py                入口：GUI 或 --cli 模式
detect.py             命令行检测入口
yolo_app/engine.py    YOLOv8 引擎封装（PT / ONNX 统一接口、模型复用与预热）
yolo_app/processor.py 单次检测流程（拍照 / 实时）
yolo_app/camera.py    USB 免驱相机（DirectShow / Media Foundation 回退、最新帧缓存）
yolo_app/hikvision.py 海康 MVS 动态接入（GigE Vision / USB3 Vision）
yolo_app/gui.py       CustomTkinter 三栏界面
yolo_app/output.py    结果保存（独立 run_* 目录、时间戳命名、原子写入）
yolo_app/paths.py     源码/冻结两套路径
data/coco.names       COCO 类别名（备用）
models/               演示模型 yolov8n.pt / yolov8n.onnx
tests/                单元、集成、GUI、工业相机探针与发布校验
installer/            Inno Setup 安装脚本
```

约定：

1. **GUI 只做展示与调度**：推理与相机逻辑放在 `engine/processor/camera/hikvision`，便于脱离界面测试。
2. **相机对象必须可释放**：任何异常路径都要走统一的 `release/close`，避免设备被占用。
3. **输出必须原子**：图片/JSON 先写临时文件再替换，沿用 `output.py` 的做法。
4. **新增依赖要谨慎**：这是 CPU、离线优先的项目，引入大体积运行时需在 README 与
   `THIRD_PARTY_NOTICES.md` 中说明；许可证必须与 AGPL-3.0 兼容。
5. **不要提交 `settings.json`、`output/`、`input/` 下的真实数据**。

## 测试

```bat
:: 全量测试（需要模型；部分用例需要真实相机，会自动跳过）
.venv\Scripts\python -m unittest discover -s tests -v

:: 相机诊断
.venv\Scripts\python detect.py --list-cameras
.venv\Scripts\python tests\probe_usb.py --source 0 --backend auto --frames 30

:: 冻结 EXE 校验（构建后）
.venv\Scripts\python tests\validate_release.py --skip-usb
```

提交前确认：

- [ ] 语法自检通过，能跑的测试通过
- [ ] 没有提交 `dist/`、`build/`、`.venv/`、`settings.json`、`release/`
- [ ] 没有提交个人路径、密钥或含敏感信息的图片
- [ ] 没有提交海康专有 DLL/SDK 或来源不明的模型权重

## 提交规范

采用 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/)：

```
feat(camera): USB 打开失败时自动重试一次
fix(gui): 修复切换相机后旧连接未释放
perf(engine): 模型预热与帧缓存，降低拍照延迟
docs: 补充海康 GigE 同网段排查
```

## Pull Request

1. 从 `main` 切分支：`feat/xxx`、`fix/xxx`、`docs/xxx`。
2. 一个 PR 只做一件事；描述里写清动机、做法、验证环境（相机型号/系统版本）。
3. 涉及界面的改动附截图；涉及识别的改动附修改前后对比。
4. 涉及新依赖或许可证的改动会在 review 中重点确认。

## 许可

本项目以 **AGPL-3.0** 发布（见 [LICENSE](LICENSE)），因为使用了 Ultralytics YOLOv8（AGPL-3.0）
与随仓库分发的 YOLOv8 预训练模型。提交贡献即表示你同意以同一许可授权你的代码。
