#!/usr/bin/env python3
"""Standalone ONNX inference tool for X-AnyLabeling trained models.

Loads an ONNX model exported from the built-in Ultralytics training
pipeline (anylabeling/services/auto_training) — or any Ultralytics-style
YOLO ONNX export — and runs detection inference on **images** and
**videos**.

Class names and input size are read automatically from the ONNX metadata
("names" / "imgsz" keys, written by the trainer exporter); a plain text
file (--classes, one class name per line) can override them.

Usage:
    # Single image
    python tools/inference/infer_onnx.py \
        --onnx /path/to/best.onnx \
        --input /path/to/image.jpg \
        --output /path/to/out_dir

    # Folder of images
    python tools/inference/infer_onnx.py \
        --onnx best.onnx --input /path/to/images_dir --output out_dir

    # Video file
    python tools/inference/infer_onnx.py \
        --onnx best.onnx --input /path/to/video.mp4 --output out_dir

    # Webcam (device index)
    python tools/inference/infer_onnx.py \
        --onnx best.onnx --input 0 --show

    # Optional flags
    --conf 0.25 --iou 0.45 --show --save-txt out_dir/labels.txt

Supported output layouts (auto-detected):
    - Ultralytics standard:  (1, 4 + num_classes, N)  -> yolo11/v8/v5...
    - End-to-end (v10):      (1, M, 6)                -> yolov10
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("[Error] onnxruntime is required:  pip install onnxruntime")
    sys.exit(1)

# Speed up repeated color generation for class visualization
_rng = np.random.default_rng(10086)

CLASS_COLORS = (
    _rng.integers(low=100, high=255, size=(1024, 3)).tolist()
)

BBOX_COLOR = (255, 0, 255)


def _print_error(msg):
    print(f"\033[91m[Error]\033[0m {msg}")


def _print_info(msg):
    print(f"\033[94m[Info]\033[0m {msg}")


def _print_success(msg):
    print(f"\033[92m[Success]\033[0m {msg}")


def load_metadata(model_path):
    """Read class names / input size from ONNX metadata (if present)."""
    names, imgsz = [], None
    try:
        import onnx

        model = onnx.load(model_path, load_external_data=False)
        meta = {item.key: item.value for item in model.metadata_props}
        raw_names = meta.get("names")
        if raw_names:
            import json

            data = json.loads(raw_names)
            if isinstance(data, list):
                names = [str(n) for n in data]
            elif isinstance(data, dict):
                names = [
                    str(data[k])
                    for k in sorted(data, key=lambda v: (0, int(v)) if str(v).isdigit() else (1, str(v)))
                ]
        if meta.get("imgsz"):
            imgsz = tuple(int(x) for x in json.loads(meta["imgsz"]))
    except Exception:
        pass
    return names, imgsz


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize image to fit new_shape keeping aspect ratio, pad borders."""
    height, width = image.shape[:2]
    r = min(new_shape[0] / height, new_shape[1] / width)
    new_unpad = (int(round(width * r)), int(round(height * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    if width != new_unpad[0] or height != new_unpad[1]:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return image, r, (left, top)


class OnnxDetector:
    """Minimal ONNX detection wrapper (image & video friendly)."""

    def __init__(self, model_path, classes=None, imgsz=None, providers=None):
        self.session = ort.InferenceSession(
            model_path,
            providers=providers
            or ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        names_from_meta, meta_imgsz = load_metadata(model_path)

        if classes:
            self.classes = classes
        elif names_from_meta:
            self.classes = names_from_meta
        else:
            self.classes = []
            _print_error(
                "No class names found in ONNX metadata. Provide --classes "
                "FILE (one class name per line, ordered by class id)."
            )
        self.input_shape = tuple(imgsz) if imgsz else tuple(
            self.session.get_inputs()[0].shape[-2:]
        )
        if any(not isinstance(v, int) for v in self.input_shape):
            self.input_shape = (640, 640)
            _print_info("Dynamic input shape detected, defaulting to 640x640.")

        _print_info(f"Input shape : {self.input_shape}")
        _print_info(f"Classes     : {self.classes or '(unknown ids)'}")

    def preprocess(self, image):
        img, r, (dx, dy) = letterbox(image, self.input_shape)
        blob = img.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
        return np.ascontiguousarray(blob), r, (dx, dy)

    def postprocess(self, output, conf_thres, iou_thres, r, pad):
        dx, dy = pad
        preds = np.squeeze(output)  # (4+nc, N) or (M, 6)

        # End-to-end models (YOLOv10): (M, 6) [x1,y1,x2,y2,conf,cls]
        if preds.ndim == 2 and preds.shape[-1] == 6 and preds.shape[0] < 1000:
            preds = preds[preds[:, 4] > conf_thres]
            boxes = preds[:, :4]
            scores = preds[:, 4]
            class_ids = preds[:, 5].astype(int)
        else:
            if preds.ndim == 1:
                preds = preds[:, None]
            if preds.shape[0] < preds.shape[1]:
                preds = preds.T  # (N, 4+nc)
            box_xywh = preds[:, :4]
            class_scores = preds[:, 4:]
            class_ids = np.argmax(class_scores, axis=1)
            scores = class_scores[np.arange(len(class_ids)), class_ids]
            keep = scores > conf_thres
            box_xywh, scores, class_ids = (
                box_xywh[keep],
                scores[keep],
                class_ids[keep],
            )
            # cxcywh -> xyxy
            boxes = np.empty_like(box_xywh)
            boxes[:, 0] = box_xywh[:, 0] - box_xywh[:, 2] / 2
            boxes[:, 1] = box_xywh[:, 1] - box_xywh[:, 3] / 2
            boxes[:, 2] = box_xywh[:, 0] + box_xywh[:, 2] / 2
            boxes[:, 3] = box_xywh[:, 1] + box_xywh[:, 3] / 2

        # Undo letterbox
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / r
        return self.nms(boxes, scores, class_ids, iou_thres)

    @staticmethod
    def nms(boxes, scores, class_ids, iou_thres):
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), 0.0, iou_thres
        )
        if len(indices) == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)
        indices = np.array(indices).flatten()
        return boxes[indices], scores[indices], class_ids[indices]

    def detect(self, image, conf_thres, iou_thres):
        blob, r, pad = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: blob})[0]
        return self.postprocess(outputs, conf_thres, iou_thres, r, pad)


def draw_results(image, boxes, scores, class_ids, classes):
    for box, score, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        color = tuple(
            int(c) for c in CLASS_COLORS[cid % len(CLASS_COLORS)]
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = (
            f"{classes[cid]}: {score:.2f}"
            if cid < len(classes)
            else f"id{cid}: {score:.2f}"
        )
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        y_top = max(y1 - 6, th + 2)
        cv2.rectangle(image, (x1, y_top - th - 4), (x1 + tw, y_top), color, -1)
        cv2.putText(
            image,
            label,
            (x1, y_top - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return image


def run_image(detector, input_path, args, classes):
    image = cv2.imread(str(input_path))
    if image is None:
        _print_error(f"Cannot read image: {input_path}")
        return []
    boxes, scores, class_ids = detector.detect(
        image, args.conf, args.iou
    )
    annotated = draw_results(image.copy(), boxes, scores, class_ids, classes)
    out_path = args.output_dir / input_path.name
    cv2.imwrite(str(out_path), annotated)
    _print_info(f"{input_path.name}: {len(boxes)} objects -> {out_path}")
    if args.save_txt:
        with open(args.save_txt, "a", encoding="utf-8") as f:
            for box, score, cid in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = box.astype(int)
                name = classes[cid] if cid < len(classes) else str(cid)
                f.write(
                    f"{input_path.name} {name} {score:.3f} "
                    f"{x1} {y1} {x2} {y2}\n"
                )
    if args.show:
        cv2.imshow("X-AnyLabeling ONNX inference", annotated)
        cv2.waitKey(500)
    return list(zip(boxes.tolist(), scores.tolist(), class_ids.tolist()))


def run_video(detector, input_path, args, classes):
    source = int(input_path) if str(input_path).isdigit() else str(input_path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        _print_error(f"Cannot open video source: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

    writer = None
    if not str(input_path).isdigit():
        out_path = args.output_dir / f"{Path(str(input_path)).stem}_infer.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        _print_info(f"Annotated video will be saved to: {out_path}")

    frame_idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes, scores, class_ids = detector.detect(frame, args.conf, args.iou)
        annotated = draw_results(frame.copy(), boxes, scores, class_ids, classes)
        if writer:
            writer.write(annotated)
        if args.show:
            cv2.imshow("X-AnyLabeling ONNX inference", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                _print_info("Stopped by user (q).")
                break
        frame_idx += 1
        if frame_idx % 50 == 0:
            elapsed = time.time() - t0
            _print_info(
                f"frame {frame_idx}/{total if total > 0 else '?'} "
                f"({frame_idx / elapsed:.1f} FPS), "
                f"last frame: {len(boxes)} objects"
            )

    cap.release()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    _print_success(f"Finished video inference: {frame_idx} frames processed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ONNX inference on images and videos "
        "(for models trained/exported by X-AnyLabeling's Ultralytics pipeline)."
    )
    parser.add_argument(
        "--onnx", required=True, help="Path to the exported ONNX model."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Image file / image directory / video file / camera index.",
    )
    parser.add_argument(
        "--output",
        default="infer_results",
        help="Output directory for annotated results (default: infer_results).",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold."
    )
    parser.add_argument(
        "--iou", type=float, default=0.45, help="NMS IoU threshold."
    )
    parser.add_argument(
        "--classes",
        default="",
        help="Optional txt file overriding class names (one per line).",
    )
    parser.add_argument(
        "--save-txt",
        default="",
        help="Optional txt file to append detection results "
        "(frame/image, class, score, x1 y1 x2 y2).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live preview window (press q in video mode to stop).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.onnx)
    if not model_path.is_file():
        _print_error(f"ONNX model not found: {model_path}")
        sys.exit(1)

    classes = []
    if args.classes:
        classes = Path(args.classes).read_text(encoding="utf-8").split()
        _print_info(f"Loaded {len(classes)} class names from {args.classes}")

    detector = OnnxDetector(str(model_path), classes=classes)
    if not classes and not detector.classes:
        sys.exit(1)
    classes = classes or detector.classes

    args.output_dir = Path(args.output)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_txt:
        args.save_txt = Path(args.save_txt)
        args.save_txt.parent.mkdir(parents=True, exist_ok=True)

    src = args.input
    if str(src).isdigit():  # camera index
        run_video(detector, src, args, classes)
    elif os.path.isdir(src):
        image_files = [
            p
            for p in sorted(Path(src).iterdir())
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        ]
        if not image_files:
            _print_error(f"No images found in directory: {src}")
            sys.exit(1)
        _print_info(f"Found {len(image_files)} images in {src}")
        for img in image_files:
            run_image(detector, img, args, classes)
        _print_success(f"Batch inference done, results in {args.output_dir}")
    elif Path(src).suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"):
        run_video(detector, src, args, classes)
    else:
        run_image(detector, Path(src), args, classes)
        _print_success(f"Result saved to {args.output_dir}")


if __name__ == "__main__":
    main()
