"""红外图像无参考质量评估（BRISQUE / NIQE / PIQE，均为越低越好）。

依赖：pip install torch pyiqa opencv-python numpy
示例（在项目根目录运行）：
    python utils/brisque_niqe_piqe_calculate.py --input results --recursive
    python utils/brisque_niqe_piqe_calculate.py --input test.png --device cpu
    python utils/brisque_niqe_piqe_calculate.py --input images --data-range 16383

默认使用 pyiqa 的 original BRISQUE、NIQE 和 PIQE；首次运行可能下载模型参数。
8 位/16 位整数图像分别除以 255/65535；浮点图像默认必须在 [0, 1]。
12/14 位数据存放在 uint16 中时，请指定 --data-range 4095/16383。
不做逐图 min-max 拉伸、增强或缩放；彩色输入转灰度，透明图像拒绝处理。
NIQE 内部按 96x96 整块裁剪，PIQE 使用后端默认的分块/边界处理。
BRISQUE/NIQE 的自然图像先验可能与红外图像存在域差异；分数宜用于相同
数据和预处理条件下的相对比较，不能直接作为红外图像质量的绝对等级。
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


METRICS = ("brisque", "niqe", "piqe")
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_image(path: Path, data_range: float | None = None,
               crop_border: int = 0) -> torch.Tensor:
    """保留原始位深读取，返回 [1, 1, H, W]、[0, 1] 灰度张量。"""
    if crop_border < 0:
        raise ValueError("crop_border 不能为负数")
    # imdecode 支持 Windows 中文路径，IMREAD_UNCHANGED 避免 16 位截断。
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("无法解码图像")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("只支持单通道或三通道图像，请先明确处理透明通道")
    if image.ndim not in (2, 3):
        raise ValueError(f"不支持的图像形状：{image.shape}")

    if data_range is None:
        if image.dtype == np.uint8:
            data_range = 255.0
        elif image.dtype == np.uint16:
            data_range = 65535.0
        elif np.issubdtype(image.dtype, np.floating):
            data_range = 1.0
        else:
            raise ValueError(f"位深 {image.dtype} 需要显式指定 --data-range")
    if not math.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range 必须是有限正数")
    if not np.isfinite(image).all():
        raise ValueError("输入含 NaN 或 Inf")
    if image.min() < 0 or image.max() > data_range:
        raise ValueError(f"像素超出 [0, {data_range:g}]，请检查 --data-range")

    image = image.astype(np.float32) / data_range
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if crop_border:
        if min(image.shape) <= 2 * crop_border:
            raise ValueError("裁边后图像为空")
        image = image[crop_border:-crop_border, crop_border:-crop_border]
    return torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).unsqueeze(0)


class InfraredQualityEvaluator:
    """模型仅初始化一次；每个指标独立记录计算失败原因。"""

    def __init__(self, device: str = "cpu"):
        import pyiqa

        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto" else device
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA 不可用，请使用 --device cpu")
        self.backend_version = getattr(pyiqa, "__version__", "unknown")
        self.models = {
            name: pyiqa.create_metric(name, device=self.device).eval()
            for name in METRICS
        }

    @torch.inference_mode()
    def evaluate(self, path: Path, data_range: float | None = None,
                 crop_border: int = 0) -> dict:
        """返回单图记录；失败指标留空，不使用 0 替代失败值。"""
        row = {"image": str(path), "height": "", "width": "",
               **{name: "" for name in METRICS}, "status": "error", "error": ""}
        try:
            tensor = load_image(Path(path), data_range, crop_border).to(self.device)
        except (OSError, ValueError, RuntimeError, cv2.error) as exc:
            row["error"] = str(exc)
            return row
        return self.evaluate_tensor(tensor, str(path))

    @torch.inference_mode()
    def evaluate_tensor(self, tensor, image="tensor"):
        """Evaluate floating grayscale pixels directly, before PNG quantization."""
        row = {"image": str(image), "height": "", "width": "",
               **{name: "" for name in METRICS}, "status": "error", "error": ""}
        if tensor.ndim != 4 or tensor.shape[:2] != (1, 1):
            raise ValueError("Expected [1,1,H,W]")
        tensor = tensor.detach().float().to(self.device)
        if not torch.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
            raise ValueError("Expected finite pixels in [0,1]")
        height, width = tensor.shape[-2:]
        row.update(height=height, width=width)
        # 常量图的统计特征退化，部分后端可能返回误导性的有限分数。
        if tensor.max().item() == tensor.min().item():
            row["error"] = "常量图像：统计特征退化，三个指标不计入汇总"
            return row
        errors = []
        for name, model in self.models.items():
            try:
                if min(height, width) < 16:
                    raise ValueError("本脚本要求图像宽高至少为 16")
                if name == "niqe" and (
                    min(height, width) < 96 or (height // 96) * (width // 96) < 2
                ):
                    raise ValueError("至少需要两个完整的 96x96 块估计协方差")
                score = float(model(tensor).item())
                if not math.isfinite(score):
                    raise ValueError("后端返回 NaN/Inf，图像统计特征可能退化")
                row[name] = score
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        row["status"] = "ok" if not errors else (
            "partial" if any(row[name] != "" for name in METRICS) else "error"
        )
        row["error"] = " | ".join(errors)
        return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="图片文件或目录")
    parser.add_argument("--output", type=Path, default=Path("nr_iqa_results.csv"),
                        help="逐图 CSV 保存路径")
    parser.add_argument("--recursive", action="store_true", help="递归搜索子目录")
    parser.add_argument("--device", default="cpu", help="cpu、cuda、cuda:0 或 auto")
    parser.add_argument("--data-range", type=float, help="统一像素上限，如 255/4095/16383/65535")
    parser.add_argument("--crop-border", type=int, default=0, help="从四边各裁去的像素数")
    args = parser.parse_args()
    if args.crop_border < 0:
        parser.error("--crop-border 不能为负数")
    if args.data_range is not None and (
        not math.isfinite(args.data_range) or args.data_range <= 0
    ):
        parser.error("--data-range 必须为有限正数")
    if args.input.is_file():
        paths = [args.input]
    elif args.input.is_dir():
        iterator = args.input.rglob("*") if args.recursive else args.input.iterdir()
        paths = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in EXTENSIONS)
    else:
        parser.error(f"输入路径不存在：{args.input}")
    if not paths:
        parser.error("没有找到支持的图像")
    if args.output.resolve() in {p.resolve() for p in paths}:
        parser.error("输出路径不能覆盖输入图像")

    try:
        evaluator = InfraredQualityEvaluator(args.device)
    except Exception as exc:
        print(f"指标初始化失败：{exc}\n请检查 pyiqa 依赖及模型参数下载/缓存。", file=sys.stderr)
        return 1
    print(f"pyiqa={evaluator.backend_version}, device={evaluator.device}, "
          f"data_range={args.data_range or '按位深'}, crop_border={args.crop_border}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    fields = ["image", "height", "width", *METRICS, "status", "error",
              "pyiqa_version", "data_range", "crop_border"]
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, path in enumerate(paths, 1):
            row = evaluator.evaluate(path, args.data_range, args.crop_border)
            row.update(pyiqa_version=evaluator.backend_version,
                       data_range=args.data_range or "dtype", crop_border=args.crop_border)
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            scores = ", ".join(
                f"{name.upper()}={row[name]:.4f}" if row[name] != ""
                else f"{name.upper()}=N/A" for name in METRICS
            )
            print(f"[{index}/{len(paths)}] {path.name}: {scores}")
            if row["error"]:
                print(f"  {row['error']}")

    print("\n逐指标均值（仅统计各自有效值，括号内为有效图像数）：")
    for name in METRICS:
        values = [row[name] for row in rows if row[name] != ""]
        mean = f"{np.mean(values):.4f}" if values else "N/A"
        print(f"  {name.upper()}: {mean} ({len(values)}/{len(rows)})")
    complete = [row for row in rows if row["status"] == "ok"]
    print(f"三个指标均有效：{len(complete)}/{len(rows)}")
    print(f"CSV 已保存：{args.output.resolve()}")
    # 有部分失败时也保留完整 CSV，但给批处理调用方非零退出码。
    return 0 if len(complete) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
