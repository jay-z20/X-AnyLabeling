#!/usr/bin/env python3
"""Export ONNX models to RKNN format via a Docker container.

This tool converts an ONNX model to a RKNN model (Rockchip NPU format,
e.g. for RK3588 / RK3576 / RK3568 boards) by delegating the actual
conversion to a Docker image that ships with rknn-toolkit2 (the image
is expected to provide a working ``python3`` with ``rknn.api``).

The host machine only invokes docker commands — all conversion steps
(load onnx, config, build, export) run inside the container.

Usage:
    # Basic conversion (no quantization):
    python tools/rknn_exporter/export_rknn.py \
        --onnx /path/to/model.onnx \
        --output /path/to/model.rknn

    # Specify target platform and docker image:
    python tools/rknn_exporter/export_rknn.py \
        --onnx /path/to/model.onnx \
        --output /path/to/model.rknn \
        --platform rk3588 \
        --image a-rknn

    # INT8 quantization with a calibration dataset:
    python tools/rknn_exporter/export_rknn.py \
        --onnx /path/to/model.onnx \
        --output /path/to/model.rknn \
        --dataset /path/to/calib_images_dir

Requirements on the host:
    - Docker must be installed and running.
    - The docker image (default: ``a-rknn``) must exist locally, e.g.:
      docker build -t a-rknn .   (any image with rknn-toolkit2 installed)

The conversion script executed inside the container:
    from rknn.api import RKNN
    rknn = RKNN(verbose=True)
    rknn.config(mean_values=..., std_values=..., target_platform=...)
    rknn.load_onnx(model=...)
    rknn.build(do_quantization=..., dataset=...)
    rknn.export_rknn(output_path=...)
"""

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

SUPPORTED_PLATFORMS = (
    "rk1808",
    "rv1103",
    "rv1106",
    "rk3562",
    "rk3566",
    "rk3568",
    "rk3576",
    "rk3588",
)

CONVERT_SCRIPT_NAME = "_xanylabeling_convert_rknn.py"

CONVERT_SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
"""Auto-generated ONNX -> RKNN conversion script (runs inside container)."""
import os
import sys

from rknn.api import RKNN

def main():
    onnx_path = sys.argv[1]
    output_path = sys.argv[2]
    platform = sys.argv[3]
    do_quant = sys.argv[4] == "1"
    dataset = sys.argv[5] if sys.argv[5] != "-" else None

    rknn = RKNN(verbose=True)

    # Model configuration
    mean_values = [[0, 0, 0]]
    std_values = [[255, 255, 255]]
    ret = rknn.config(
        mean_values=mean_values,
        std_values=std_values,
        target_platform=platform,
    )
    if ret != 0:
        print("Configure rknn model config failed!")
        sys.exit(1)

    # Load ONNX model
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print("Load ONNX model failed!")
        sys.exit(1)

    # Build model
    ret = rknn.build(do_quantization=do_quant, dataset=dataset)
    if ret != 0:
        print("Build RKNN model failed!")
        sys.exit(1)

    # Export RKNN model
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print("Export RKNN model failed!")
        sys.exit(1)

    print("Export RKNN model success ->", output_path)
    sys.exit(0)

if __name__ == "__main__":
    main()
'''


def _print_error(msg: str) -> None:
    print(f"\033[91m[Error]\033[0m {msg}")


def _print_info(msg: str) -> None:
    print(f"\033[94m[Info]\033[0m {msg}")


def _print_success(msg: str) -> None:
    print(f"\033[92m[Success]\033[0m {msg}")


def check_docker_available() -> bool:
    """Verify docker is installed and the daemon is reachable."""
    try:
        subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            check=True,
            capture_output=True,
        )
        return True
    except FileNotFoundError:
        _print_error(
            "Docker not found on PATH. Please install Docker first: "
            "https://docs.docker.com/get-docker/"
        )
        return False
    except subprocess.CalledProcessError as exc:
        _print_error(
            "Docker daemon is not running or not accessible: "
            f"{exc.stderr.decode(errors='ignore').strip()}"
        )
        return False


def check_image_exists(image: str) -> bool:
    """Verify the conversion image exists locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
    )
    if result.returncode != 0:
        _print_error(
            f"Docker image '{image}' not found locally. Build or pull it "
            f"first, e.g.:  docker pull {image}  or  docker build -t {image} ."
        )
        return False
    return True


def export_rknn(
    onnx_path: Path,
    output_path: Path,
    platform: str,
    image: str,
    dataset_path: str,
) -> bool:
    """Run the ONNX -> RKNN conversion inside a docker container.

    Mounts:
        - <onnx parent dir> -> /input  (read-only)
        - <output parent dir> -> /output (read-write)
        - <dataset dir> -> /dataset (read-write, only when quantizing)
        - generated conversion script -> /tmp (single file mount)
    """
    onnx_path = onnx_path.resolve()
    output_path = output_path.resolve()
    onnx_in_container = f"/input/{onnx_path.name}"
    output_in_container = f"/output/{output_path.name}"

    work_dir = tempfile.mkdtemp(prefix="rknn_export_")
    script_host_path = Path(work_dir) / CONVERT_SCRIPT_NAME
    script_host_path.write_text(CONVERT_SCRIPT_TEMPLATE, encoding="utf-8")

    do_quant = bool(dataset_path)
    dataset_arg = "-"
    dataset_mount = []
    if do_quant:
        dataset_dir = Path(dataset_path).resolve()
        if not dataset_dir.is_dir():
            _print_error(f"Dataset directory not found: {dataset_dir}")
            return False
        dataset_host_dir = Path(work_dir) / "dataset"
        dataset_host_dir.mkdir(exist_ok=True)
        # Write rknn dataset list file (one image path per line)
        image_files = [
            p.name
            for p in sorted(dataset_dir.iterdir())
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        ]
        if not image_files:
            _print_error(
                f"No calibration images (*.jpg/*.png/*.bmp) found in: "
                f"{dataset_dir}"
            )
            return False
        list_file = dataset_host_dir / "dataset.txt"
        list_file.write_text(
            "\n".join(f"/dataset/{name}" for name in image_files) + "\n",
            encoding="utf-8",
        )
        dataset_arg = "/dataset_work/dataset.txt"
        dataset_mount = [f"{dataset_dir}:/dataset", f"{dataset_host_dir}:/dataset_work"]

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{onnx_path.parent}:/input:ro",
        "-v",
        f"{output_path.parent}:/output",
        "-v",
        f"{script_host_path}:/tmp/{CONVERT_SCRIPT_NAME}:ro",
    ]
    if dataset_mount:
        cmd += ["-v", dataset_mount[0], "-v", dataset_mount[1]]
    cmd += [
        image,
        "python3",
        f"/tmp/{CONVERT_SCRIPT_NAME}",
        onnx_in_container,
        output_in_container,
        platform,
        "1" if do_quant else "0",
        dataset_arg,
    ]

    _print_info("Running container:")
    _print_info(" ".join(shlex.quote(part) for part in cmd))
    result = subprocess.run(cmd)
    return result.returncode == 0 and output_path.exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ONNX model to RKNN format via docker "
        "(conversion runs inside the container, e.g. image 'a-rknn')."
    )
    parser.add_argument(
        "--onnx",
        required=True,
        help="Path to the input ONNX model file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output .rknn model file.",
    )
    parser.add_argument(
        "--platform",
        default="rk3588",
        choices=SUPPORTED_PLATFORMS,
        help="Target RKNN platform (default: rk3588).",
    )
    parser.add_argument(
        "--image",
        default="a-rknn",
        help="Docker image with rknn-toolkit2 installed (default: a-rknn).",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Optional directory of calibration images (*.jpg/*.png) for "
        "INT8 quantization. If omitted, the model is exported in fp16.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    onnx_path = Path(args.onnx)
    output_path = Path(args.output)

    if not onnx_path.is_file():
        _print_error(f"ONNX model not found: {onnx_path}")
        sys.exit(1)
    if output_path.suffix.lower() != ".rknn":
        _print_error("Output path must end with '.rknn'.")
        sys.exit(1)

    _print_info(f"ONNX model   : {onnx_path}")
    _print_info(f"Output RKNN  : {output_path}")
    _print_info(f"Platform     : {args.platform}")
    _print_info(f"Docker image : {args.image}")
    _print_info(
        "Quantization : "
        + (f"INT8 (dataset={args.dataset})" if args.dataset else "fp16 (no quantization)")
    )

    if not check_docker_available():
        sys.exit(1)
    if not check_image_exists(args.image):
        sys.exit(1)

    ok = export_rknn(
        onnx_path=onnx_path,
        output_path=output_path,
        platform=args.platform,
        image=args.image,
        dataset_path=args.dataset,
    )
    if ok:
        _print_success(f"RKNN model saved to: {output_path}")
    else:
        _print_error("ONNX -> RKNN conversion failed, see container logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
