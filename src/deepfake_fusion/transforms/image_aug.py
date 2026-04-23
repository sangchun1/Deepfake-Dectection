from __future__ import annotations

import io
import random
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """
    dict 또는 attribute 방식 모두 지원하는 안전한 config getter.
    예:
        _cfg_get(cfg, "image", "input_size", default=224)
        _cfg_get(cfg, "augmentation", "hflip_prob", default=0.5)
    """
    current = cfg

    for key in keys:
        if current is None:
            return default

        if isinstance(current, Mapping):
            current = current.get(key, None)
        else:
            current = getattr(current, key, None)

        if current is None:
            return default

    return current


def _to_2tuple(size: Union[int, Sequence[int]]) -> Tuple[int, int]:
    """
    입력 크기를 (H, W) 형태로 정규화.
    """
    if isinstance(size, int):
        return (size, size)

    size = tuple(size)
    if len(size) != 2:
        raise ValueError(f"input_size must be int or sequence of length 2, got: {size}")

    return int(size[0]), int(size[1])


def _to_float_tuple(values: Optional[Sequence[float]], default: Tuple[float, ...]) -> Tuple[float, ...]:
    if values is None:
        return default
    return tuple(float(v) for v in values)


class RandomJPEGCompression:
    """
    PIL 이미지를 확률적으로 JPEG 재인코딩했다가 다시 읽어오는 transform.
    """

    def __init__(self, quality_min: int = 40, quality_max: int = 95, p: float = 0.0) -> None:
        if not (1 <= quality_min <= 100 and 1 <= quality_max <= 100):
            raise ValueError("JPEG quality must be in [1, 100].")
        if quality_min > quality_max:
            raise ValueError("quality_min must be <= quality_max.")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1].")

        self.quality_min = int(quality_min)
        self.quality_max = int(quality_max)
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.p <= 0.0 or random.random() >= self.p:
            return img

        quality = random.randint(self.quality_min, self.quality_max)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        return out


class RandomResizeDownUp:
    """
    PIL 이미지를 먼저 축소한 뒤 다시 원래 크기로 복원.
    compression / resize 아티팩트에 대한 robustness 학습용.
    """

    def __init__(
        self,
        scale_min: float = 0.5,
        scale_max: float = 0.9,
        p: float = 0.0,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        if not (0.0 < scale_min <= 1.0 and 0.0 < scale_max <= 1.0):
            raise ValueError("scale_min/scale_max must be in (0, 1].")
        if scale_min > scale_max:
            raise ValueError("scale_min must be <= scale_max.")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1].")

        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.p = float(p)
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.p <= 0.0 or random.random() >= self.p:
            return img

        w, h = img.size
        scale = random.uniform(self.scale_min, self.scale_max)
        down_w = max(1, int(round(w * scale)))
        down_h = max(1, int(round(h * scale)))

        img = img.resize((down_w, down_h), self.interpolation)
        img = img.resize((w, h), self.interpolation)
        return img


class AddGaussianNoise:
    """
    tensor 이미지에 가우시안 노이즈 추가.
    ToTensor 이후, Normalize 이전에 적용.
    """

    def __init__(self, std: float = 0.02, p: float = 0.0) -> None:
        if std < 0.0:
            raise ValueError("std must be >= 0.")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1].")
        self.std = float(std)
        self.p = float(p)

   def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0 or random.random() >= self.p or self.std == 0.0:
            return x
        noise = torch.randn_like(x) * self.std
        return (x + noise).clamp(0.0, 1.0)


def build_train_transform(
    input_size: Union[int, Sequence[int]] = 224,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    hflip_prob: float = 0.5,
    rotation_degrees: float = 10.0,
    color_jitter_prob: float = 0.2,
    color_jitter_brightness: float = 0.2,
    color_jitter_contrast: float = 0.2,
    color_jitter_saturation: float = 0.2,
    color_jitter_hue: float = 0.02,
    gaussian_blur_prob: float = 0.0,
    gaussian_blur_kernel: int = 3,
    gaussian_noise_prob: float = 0.0,
    gaussian_noise_std: float = 0.02,
    jpeg_prob: float = 0.0,
    jpeg_quality_min: int = 40,
    jpeg_quality_max: int = 90,
    resize_downup_prob: float = 0.0,
    resize_scale_min: float = 0.5,
    resize_scale_max: float = 0.9,
) -> T.Compose:
    """
    spatial branch용 train transform.
    너무 강한 augmentation은 baseline 재현성을 해칠 수 있어서
    기본적으로 약한 augmentation만 적용한다.
    """
    size = _to_2tuple(input_size)
    mean = _to_float_tuple(mean, IMAGENET_MEAN)
    std = _to_float_tuple(std, IMAGENET_STD)

    color_jitter = T.ColorJitter(
        brightness=color_jitter_brightness,
        contrast=color_jitter_contrast,
        saturation=color_jitter_saturation,
        hue=color_jitter_hue,
    )
    gaussian_blur_kernel = int(gaussian_blur_kernel)
    if gaussian_blur_kernel % 2 == 0:
        gaussian_blur_kernel += 1

    transform = T.Compose(
        [
            T.Resize(size),
            T.RandomHorizontalFlip(p=hflip_prob),
            T.RandomRotation(degrees=rotation_degrees),
            T.RandomApply([color_jitter], p=color_jitter_prob),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=gaussian_blur_kernel)],
                p=gaussian_blur_prob,
            ),
            RandomJPEGCompression(
                quality_min=jpeg_quality_min,
                quality_max=jpeg_quality_max,
                p=jpeg_prob,
            ),
            RandomResizeDownUp(
                scale_min=resize_scale_min,
                scale_max=resize_scale_max,
                p=resize_downup_prob,
            ),
             T.ToTensor(),
            AddGaussianNoise(
                std=gaussian_noise_std,
                p=gaussian_noise_prob,
            ),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return transform


def build_eval_transform(
    input_size: Union[int, Sequence[int]] = 224,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> T.Compose:
    """
    val/test용 deterministic transform.
    """
    size = _to_2tuple(input_size)
    mean = _to_float_tuple(mean, IMAGENET_MEAN)
    std = _to_float_tuple(std, IMAGENET_STD)

    transform = T.Compose(
        [
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return transform


def build_image_transform(
    data_cfg: Any,
    split: str = "train",
    aug_cfg: Optional[Any] = None,
) -> T.Compose:
    """
    data config를 받아 split별 transform 생성.

    기대하는 data config 예시:
        cfg.data.image.input_size
        cfg.data.image.mean
        cfg.data.image.std

    aug_cfg가 있으면 아래 키를 선택적으로 override 가능:
        hflip_prob
        rotation_degrees
        color_jitter_prob
        color_jitter_brightness
        color_jitter_contrast
        color_jitter_saturation
        color_jitter_hue
        gaussian_blur_prob
        gaussian_blur_kernel
        gaussian_noise_prob
        gaussian_noise_std
        jpeg_prob
        jpeg_quality_min
        jpeg_quality_max
        resize_downup_prob
        resize_scale_min
        resize_scale_max
    """
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be one of ['train', 'val', 'test'], got: {split}")

    input_size = _cfg_get(data_cfg, "image", "input_size", default=224)
    mean = _cfg_get(data_cfg, "image", "mean", default=IMAGENET_MEAN)
    std = _cfg_get(data_cfg, "image", "std", default=IMAGENET_STD)

    if split == "train":
        return build_train_transform(
            input_size=input_size,
            mean=mean,
            std=std,
            hflip_prob=float(_cfg_get(aug_cfg, "hflip_prob", default=0.5)),
            rotation_degrees=float(_cfg_get(aug_cfg, "rotation_degrees", default=10.0)),
            color_jitter_prob=float(_cfg_get(aug_cfg, "color_jitter_prob", default=0.2)),
            color_jitter_brightness=float(
                _cfg_get(aug_cfg, "color_jitter_brightness", default=0.2)
            ),
            color_jitter_contrast=float(
                _cfg_get(aug_cfg, "color_jitter_contrast", default=0.2)
            ),
            color_jitter_saturation=float(
                _cfg_get(aug_cfg, "color_jitter_saturation", default=0.2)
            ),
            color_jitter_hue=float(_cfg_get(aug_cfg, "color_jitter_hue", default=0.02)),
            gaussian_blur_prob=float(_cfg_get(aug_cfg, "gaussian_blur_prob", default=0.0)),
            gaussian_blur_kernel=int(_cfg_get(aug_cfg, "gaussian_blur_kernel", default=3)),
            gaussian_noise_prob=float(_cfg_get(aug_cfg, "gaussian_noise_prob", default=0.0)),
            gaussian_noise_std=float(_cfg_get(aug_cfg, "gaussian_noise_std", default=0.02)),
            jpeg_prob=float(_cfg_get(aug_cfg, "jpeg_prob", default=0.0)),
            jpeg_quality_min=int(_cfg_get(aug_cfg, "jpeg_quality_min", default=40)),
            jpeg_quality_max=int(_cfg_get(aug_cfg, "jpeg_quality_max", default=90)),
            resize_downup_prob=float(_cfg_get(aug_cfg, "resize_downup_prob", default=0.0)),
            resize_scale_min=float(_cfg_get(aug_cfg, "resize_scale_min", default=0.5)),
            resize_scale_max=float(_cfg_get(aug_cfg, "resize_scale_max", default=0.9)),
        )

    return build_eval_transform(
        input_size=input_size,
        mean=mean,
        std=std,
    )


def build_transforms_from_config(cfg: Any) -> dict:
    """
    experiment config 전체를 받아 train/val/test transform 한 번에 생성.

    사용 예:
        transforms = build_transforms_from_config(cfg)
        train_tf = transforms["train"]
        val_tf = transforms["val"]

    우선순위:
        1) cfg.train.augmentation
        2) cfg.augmentation
        3) 기본값
    """
    data_cfg = _cfg_get(cfg, "data", default=cfg)
    aug_cfg = _cfg_get(cfg, "train", "augmentation", default=None)

    if aug_cfg is None:
        aug_cfg = _cfg_get(cfg, "augmentation", default=None)

    return {
        "train": build_image_transform(data_cfg=data_cfg, split="train", aug_cfg=aug_cfg),
        "val": build_image_transform(data_cfg=data_cfg, split="val", aug_cfg=aug_cfg),
        "test": build_image_transform(data_cfg=data_cfg, split="test", aug_cfg=aug_cfg),
    }