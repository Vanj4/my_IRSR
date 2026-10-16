"""红外读取、固定量程归一化、裁剪和成对增强；参数默认来自config.py。
退化调用scripts.prepare_data.bicubic_degradation，不在本文件复制插值算法。
训练默认128裁块，25%水平翻转，从±90/±180四项等概率旋转。
验证/测试通过build_dataset使用确定性中心块或整图，无随机增强。
"""
from __future__ import annotations
import argparse
import math
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import config
from scripts.prepare_data import bicubic_degradation, _validate_scale

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_UNSET = object()  # 区分“未传参”和显式None（整图或自动量程）。

def read_infrared_image(path: str | Path,
                        data_range: float | None = _UNSET) -> torch.Tensor:
    """保留位深读取图片，返回 [1, H, W] 灰度张量，支持中文路径。"""
    data_range = config.DATA_RANGE if data_range is _UNSET else data_range
    path = Path(path)
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法解码图像：{path}")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError(f"仅支持单通道或三通道图像，不支持透明通道：{path}")
    if data_range is None:
        if image.dtype == np.uint8:
            data_range = 255.0
        elif image.dtype == np.uint16:
            data_range = 65535.0
        elif np.issubdtype(image.dtype, np.floating):
            data_range = 1.0
        else:
            raise ValueError(f"请为 {image.dtype} 输入指定 data_range：{path}")
    if not math.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range 必须是有限正数")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > data_range:
        raise ValueError(f"图像含无效像素或超出 [0, {data_range}]：{path}")
    image = image.astype(np.float32) / data_range
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)

class InfraredSRDataset(Dataset):
    """每次 __getitem__ 从 HR 在线生成 LR，不缓存或写出退化图像。

    hr_patch_size=None：裁掉底部/右侧余数，使 HR 宽高为 scale 的倍数。
    hr_patch_size=整数：从原始 HR 随机裁取正方形训练 patch，再退化。
    random_crop=False 时使用中心 patch，适合确定性的验证。
    对于不同大小的整图，DataLoader 应使用 batch_size=1；训练时设置
    固定 hr_patch_size 可直接堆叠批次。过小图像报错，不进行额外放大。
    随机裁剪使用 PyTorch RNG，可通过 torch.manual_seed 控制随机种子。
    默认训练：裁128方块→下采样→25%水平翻转→从±90/±180等概率旋转。
    LR/HR共享增强参数，±180等价，因此实际180度概率50%，不含0度。
    random_crop=False或hr_patch_size=None时默认关闭增强。
    """

    def __init__(self, root: str | Path | None = None, scale: int | None = None,
                 hr_patch_size: int | None = _UNSET, recursive: bool | None = None,
                 random_crop: bool | None = None, data_range: float | None = _UNSET,
                 augment: bool | None = None, horizontal_flip_probability: float | None = None,
                 rotation_angles: tuple[int, ...] | None = None, seed: int | None = None):
        self.seed, self.epoch = seed, 0
        root = config.TRAIN_ROOT if root is None else root
        scale = config.SCALE if scale is None else scale
        hr_patch_size = config.HR_PATCH_SIZE if hr_patch_size is _UNSET else hr_patch_size
        recursive = config.RECURSIVE if recursive is None else recursive
        random_crop = config.RANDOM_CROP if random_crop is None else random_crop
        data_range = config.DATA_RANGE if data_range is _UNSET else data_range
        horizontal_flip_probability = (config.HORIZONTAL_FLIP_PROBABILITY
            if horizontal_flip_probability is None else horizontal_flip_probability)
        rotation_angles = config.ROTATION_ANGLES if rotation_angles is None else rotation_angles
        _validate_scale(scale)
        if hr_patch_size is not None and (
            isinstance(hr_patch_size, bool) or not isinstance(hr_patch_size, int)
            or hr_patch_size < scale or hr_patch_size % scale
        ):
            raise ValueError("hr_patch_size 必须是正整数且为 scale 的倍数")
        if data_range is not None and (not math.isfinite(data_range) or data_range <= 0):
            raise ValueError("data_range 必须是有限正数")
        root = Path(root).expanduser()
        if root.is_file():
            self.paths = [root]
        elif root.is_dir():
            candidates = root.rglob("*") if recursive else root.iterdir()
            self.paths = sorted(p for p in candidates
                                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        else:
            raise FileNotFoundError(f"输入路径不存在：{root}")
        if not self.paths:
            raise ValueError(f"没有找到支持的图像：{root}")
        if not math.isfinite(horizontal_flip_probability) or not 0 <= horizontal_flip_probability <= 1:
            raise ValueError("水平翻转概率必须在[0,1]")
        if not rotation_angles or any(a not in (90, -90, 180, -180) for a in rotation_angles):
            raise ValueError("旋转角度必须从+90/-90/+180/-180中指定")
        self.augment = (config.TRAIN_AUGMENT and random_crop and hr_patch_size is not None) if augment is None else bool(augment)
        if self.augment and (not random_crop or hr_patch_size is None):
            raise ValueError("验证/整图模式不启用随机增强；训练请设置随机方形裁块")
        self.horizontal_flip_probability = horizontal_flip_probability
        self.rotation_angles = tuple(rotation_angles)
        self.scale = scale
        self.hr_patch_size = hr_patch_size
        self.random_crop = random_crop
        self.data_range = data_range

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def rgb_path_for(path):
        path = Path(path)
        ir_root = next((parent for parent in path.parents if parent.name.lower() == "ir"), None)
        if ir_root is None:
            raise ValueError(f"红外路径必须位于IR目录中：{path}")
        return ir_root.parent / "RGB" / path.relative_to(ir_root)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        generator = None if self.seed is None else torch.Generator().manual_seed(
            (self.seed + self.epoch * 1000003 + index * 9176) % (2**63 - 1))
        path = self.paths[index]
        hr = read_infrared_image(path, self.data_range)
        rgb_path = self.rgb_path_for(path)
        if not rgb_path.is_file():
            raise FileNotFoundError(f"缺少同名 RGB 配对图像：{rgb_path}")
        rgb_image = cv2.imdecode(np.fromfile(rgb_path, dtype=np.uint8),
                                 cv2.IMREAD_UNCHANGED)
        if rgb_image is None:
            raise ValueError(f"无法解码 RGB 图像：{rgb_path}")
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(f"RGB 必须是三通道彩色图像：{rgb_path}")
        if rgb_image.dtype == np.uint8:
            rgb_range = 255.0
        elif rgb_image.dtype == np.uint16:
            rgb_range = 65535.0
        elif np.issubdtype(rgb_image.dtype, np.floating):
            rgb_range = 1.0
        else:
            raise ValueError(f"不支持的 RGB 数据类型：{rgb_image.dtype}")
        if (not np.isfinite(rgb_image).all() or rgb_image.min() < 0
                or rgb_image.max() > rgb_range):
            raise ValueError(f"RGB 像素超出 [0, {rgb_range}]：{rgb_path}")
        # OpenCV decodes BGR; return RGB, float32, CHW, normalized to [0,1].
        rgb_image = cv2.cvtColor(rgb_image.astype(np.float32) / rgb_range,
                                 cv2.COLOR_BGR2RGB)
        rgb = torch.from_numpy(np.ascontiguousarray(rgb_image.transpose(2, 0, 1)))
        height, width = hr.shape[-2:]
        if rgb.shape[-2:] != hr.shape[-2:]:
            raise ValueError(f"IR/RGB 尺寸不一致，不能直接同步裁剪："
                             f"{path} {tuple(hr.shape[-2:])}；"
                             f"{rgb_path} {tuple(rgb.shape[-2:])}")
        top = left = 0
        size = self.hr_patch_size
        if size is not None:
            if min(height, width) < size:
                raise ValueError(f"图像 {path} 的尺寸 {height}x{width} 小于 patch {size}")
            if self.random_crop:
                top = torch.randint(height - size + 1, (1,), generator=generator).item()
                left = torch.randint(width - size + 1, (1,), generator=generator).item()
            else:
                top, left = (height - size) // 2, (width - size) // 2
            hr = hr[:, top:top + size, left:left + size]
            rgb = rgb[:, top:top + size, left:left + size]
        else:
            height = height // self.scale * self.scale
            width = width // self.scale * self.scale
            if not height or not width:
                raise ValueError(f"图像尺寸小于退化倍数 {self.scale}：{path}")
            hr = hr[:, :height, :width]
            rgb = rgb[:, :height, :width]
        hr = hr.contiguous()
        lr = bicubic_degradation(hr, self.scale)
        # Geometry is sampled once and applied identically to LR and HR AFTER degradation.
        flipped, angle = False, 0
        if self.augment:
            flipped = bool(torch.rand((), generator=generator).item() < self.horizontal_flip_probability)
            if flipped:
                lr, hr, rgb = lr.flip(-1), hr.flip(-1), rgb.flip(-1)
            angle = self.rotation_angles[torch.randint(len(self.rotation_angles), (), generator=generator).item()]
            lr, hr, rgb = (torch.rot90(t, angle // 90, (-2, -1))
                           for t in (lr, hr, rgb))
        metadata = {"top": int(top), "left": int(left), "horizontal_flip": flipped,
                    "angle_degrees": angle, "scale": self.scale,
                    "degradation": "pytorch_bicubic_antialias", "quantize_lr": False}
        return {"lr": lr.contiguous(), "hr": hr.contiguous(),
                "rgb": rgb.contiguous(), "rgb_path": str(rgb_path), "path": str(path),
                "sample_meta": json.dumps(metadata, ensure_ascii=False)}


def build_dataset(split="train"):
    """按config构建对应数据集；验证/测试路径缺失时直接说明，不替用户划分。"""
    if split == "train":
        if config.TRAIN_ROOT is None: raise ValueError("请在config.py配置TRAIN_ROOT")
        return InfraredSRDataset(config.TRAIN_ROOT)
    if split not in ("validation", "test"):
        raise ValueError("split必须是train、validation或test")
    root = config.VALIDATION_ROOT if split == "validation" else config.TEST_ROOT
    size = config.VALIDATION_PATCH_SIZE if split == "validation" else config.TEST_PATCH_SIZE
    if root is None:
        raise ValueError(f"请先在config.py配置{split}的独立HR路径")
    return InfraredSRDataset(root,hr_patch_size=size,random_crop=False,augment=False)


def main():
    parser=argparse.ArgumentParser(description="检查config指定的红外读取与预处理，不保存LR")
    parser.add_argument("--split",choices=("train","validation","test"),default="train")
    parser.add_argument("--count",type=int,default=1,help="仅预览多少张；不限制正式数据集长度")
    args=parser.parse_args()
    if args.count<1: parser.error("count必须为正数")
    data=build_dataset(args.split)
    print(f"split={args.split}, images={len(data)}")
    for index in range(min(args.count,len(data))):
        sample=data[index]
        print(sample["path"],tuple(sample["hr"].shape),"->",tuple(sample["lr"].shape),sample["sample_meta"])


if __name__=="__main__":
    torch.manual_seed(config.SEED)
    main()
