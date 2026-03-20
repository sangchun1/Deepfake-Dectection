from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from torchvision import transforms as T


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

    transform = T.Compose(
        [
            T.Resize(size),
            T.RandomHorizontalFlip(p=hflip_prob),
            T.RandomRotation(degrees=rotation_degrees),
            T.RandomApply([color_jitter], p=color_jitter_prob),
            T.ToTensor(),
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