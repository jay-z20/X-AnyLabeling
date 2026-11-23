# ONNX Inference (images & videos)

独立推理工具：加载训练/导出得到的 ONNX 模型，对**图像**和**视频**做检测推理并保存可视化结果。不依赖 GUI，只要求 `onnxruntime + opencv + numpy`。

## 使用方法

```bash
# 单张图片
python tools/inference/infer_onnx.py \
    --onnx /path/to/best.onnx \
    --input /path/to/image.jpg \
    --output /path/to/out_dir

# 整个图片目录
python tools/inference/infer_onnx.py \
    --onnx best.onnx --input /path/to/images_dir --output out_dir

# 视频文件（自动保存标注后的视频 *_infer.mp4）
python tools/inference/infer_onnx.py \
    --onnx best.onnx --input /path/to/video.mp4 --output out_dir

# 摄像头（设备索引），按 q 退出
python tools/inference/infer_onnx.py --onnx best.onnx --input 0 --show

# 常用可选参数
--conf 0.25        # 置信度阈值
--iou 0.45         # NMS IoU 阈值
--show             # 实时预览窗口
--save-txt out/labels.txt   # 检测结果追加写入文本
--classes names.txt         # 手动指定类名（每行一个，按类别 id 排序）
```

## 类名与输入尺寸

- 训练管线（`anylabeling/services/auto_training/ultralytics`）导出 ONNX 时会把
  `names`（类别名）和 `imgsz`（输入尺寸）写入 ONNX metadata，本工具会自动读取，
  无需额外配置。
- 若模型没有 metadata（第三方 ONNX），请通过 `--classes` 提供类名文件，
  否则脚本会报错退出。

## 支持的输出布局（自动识别）

| 布局 | 形状 | 对应模型 |
| --- | --- | --- |
| Ultralytics 标准 | `(1, 4+num_classes, N)` | YOLOv5/v8/v9/v11/v12/YOLO26 等 |
| 端到端 | `(1, M, 6)` | YOLOv10（无需 NMS） |

> 说明：当前版本针对**检测**任务（bbox）。分割/姿态/旋转框等任务的
> 后处理与 GUI 端 `anylabeling/services/auto_labeling/__base__/yolo.py` 保持一致逻辑，
> 如需支持可在 `postprocess` 中扩展。

## 与训练管线的衔接

```
训练 (TrainingManager -> ultralytics worker)
  -> 导出 ONNX (auto_training/ultralytics/exporter.py，metadata 含 names/imgsz)
  -> 本工具离线推理 (tools/inference/infer_onnx.py)
  -> （可选）导出 RKNN (tools/rknn_exporter/export_rknn.py，docker)
```
