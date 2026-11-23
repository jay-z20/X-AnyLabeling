# RKNN Exporter (ONNX → RKNN via Docker)

将 X-AnyLabeling 导出的 ONNX 模型转换为 Rockchip NPU 使用的 RKNN 模型。

转换过程**全部在 Docker 容器内完成**（容器内需安装 rknn-toolkit2），本工具只负责调用 docker 命令，宿主机无需安装任何 RKNN 依赖。

## 使用方法

```bash
# 基本转换（fp16，不做量化），默认镜像 a-rknn、默认平台 rk3588
python tools/rknn_exporter/export_rknn.py \
    --onnx /path/to/model.onnx \
    --output /path/to/model.rknn

# 指定目标芯片与镜像
python tools/rknn_exporter/export_rknn.py \
    --onnx /path/to/model.onnx \
    --output /path/to/model.rknn \
    --platform rk3588 \
    --image a-rknn

# INT8 量化（提供标定图片目录）
python tools/rknn_exporter/export_rknn.py \
    --onnx /path/to/model.onnx \
    --output /path/to/model.rknn \
    --dataset /path/to/calib_images_dir
```

## 参数说明

| 参数 | 说明 |
| --- | --- |
| `--onnx` | 输入的 ONNX 模型路径（必需） |
| `--output` | 输出的 .rknn 模型路径（必需） |
| `--platform` | 目标平台：rk1808 / rv1103 / rv1106 / rk3562 / rk3566 / rk3568 / rk3576 / rk3588（默认 rk3588） |
| `--image` | Docker 镜像名，需内置 rknn-toolkit2（默认 `a-rknn`） |
| `--dataset` | 可选，INT8 量化标定图片目录（*.jpg/*.jpeg/*.png/*.bmp） |

## 镜像要求

镜像内需要满足：

- 有 `python3` 命令；
- 已安装 `rknn-toolkit2`（`from rknn.api import RKNN` 可用）。

容器内执行的转换逻辑（由本工具自动生成）：

```python
rknn = RKNN(verbose=True)
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=...)
rknn.load_onnx(model=...)
rknn.build(do_quantization=..., dataset=...)
rknn.export_rknn(output_path=...)
```

> 注意：`mean/std` 当前固定为 `[0,0,0]/[255,255,255]`（即 0~1 归一化）。如模型训练时使用其他归一化方式，请修改
> `tools/rknn_exporter/export_rknn.py` 中的 `CONVERT_SCRIPT_TEMPLATE`。

## 宿主机要求

- 已安装并运行 Docker（Windows 下为 Docker Desktop）；
- 本地存在转换镜像，例如 `docker build -t a-rknn .` 或 `docker pull <image>`。
