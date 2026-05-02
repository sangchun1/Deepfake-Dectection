from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Type

from torch.utils.data import DataLoader

from deepfake_fusion.datasets.binary_image_dataset import BinaryImageDataset
from deepfake_fusion.engine.trainer import Trainer
from deepfake_fusion.models.build_model import build_model, get_model_summary
from deepfake_fusion.transforms.image_aug import build_transforms_from_config
from deepfake_fusion.utils.calibration import load_temperature_value
from deepfake_fusion.utils.config import (
    load_experiment_config,
    pretty_print_config,
    resolve_path,
)
from deepfake_fusion.utils.seed import (
    get_torch_generator,
    seed_everything,
    seed_worker,
)

DATASET_REGISTRY: Dict[str, Type] = {
    "binaryimagedataset": BinaryImageDataset,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained deepfake detector.")

    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data/default.yaml",
        help="Path to data config YAML.",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model/resnet18.yaml",
        help="Path to model config YAML.",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train/spatial_resnet.yaml",
        help="Path to train config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. If omitted, output_dir/best.pth is used.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from config. Example: cuda, cpu, mps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override dataloader batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override dataloader num_workers.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save evaluation result JSON.",
    )
    parser.add_argument(
        "--temperature_json",
        type=str,
        default=None,
        help=(
            "Optional path to temperature.json produced by "
            "scripts/calibrate_temperature.py. If provided, logits are "
            "scaled by 1/T before evaluation."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Optional scalar temperature override. Mutually exclusive with "
            "--temperature_json."
        ),
    )
    parser.add_argument(
        "--threshold_override",
        type=float,
        default=None,
        help=(
            "Optional threshold override for binary prediction. If omitted, "
            "cfg.train.evaluation.threshold is used."
        ),
    )
    parser.add_argument(
        "--split_csv_override",
        type=str,
        default=None,
        help=(
            "Optional CSV path to evaluate instead of cfg.data.paths.<split>_csv. "
            "Useful for per-generator test evaluation."
        ),
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=True)


def format_metrics(metrics: Dict[str, Any]) -> str:
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


def build_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    seed: int,
):
    use_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker if use_workers else None,
        generator=get_torch_generator(seed),
        persistent_workers=use_workers,
    )


def get_split_csv_path(cfg, split: str) -> str:
    if split == "train":
        return cfg.data.paths.train_csv
    if split == "val":
        return cfg.data.paths.val_csv
    if split == "test":
        return cfg.data.paths.test_csv
    raise ValueError(f"Unsupported split: {split}")


def get_effective_split_csv_path(cfg, split: str, split_csv_override: Optional[str]) -> str:
    if split_csv_override is not None:
        return split_csv_override
    return get_split_csv_path(cfg, split)


def infer_generator_name_from_csv(csv_path: Path) -> Optional[str]:
    stem = csv_path.stem.strip()
    if not stem:
        return None
    return stem


def get_split_shuffle(cfg, split: str) -> bool:
    if split == "train":
        return bool(cfg.data.train.shuffle)
    if split == "val":
        return bool(cfg.data.val.shuffle)
    if split == "test":
        return bool(cfg.data.test.shuffle)
    raise ValueError(f"Unsupported split: {split}")


def get_dataset_class(cfg):
    dataset_key = None

    if getattr(cfg.data, "dataset_class", None) is not None:
        dataset_key = str(cfg.data.dataset_class)
    elif getattr(cfg.data, "name", None) is not None:
        dataset_key = str(cfg.data.name)

    if dataset_key is None:
        raise ValueError(
            "Could not determine dataset class. Set cfg.data.dataset_class "
            "or cfg.data.name in the data config."
        )

    if dataset_key not in DATASET_REGISTRY:
        supported = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unsupported dataset '{dataset_key}'. Supported values: {supported}"
        )

    return DATASET_REGISTRY[dataset_key]


def build_single_dataset(dataset_cls, csv_path, root_dir, transform):
    return dataset_cls(
        csv_path=csv_path,
        root_dir=root_dir,
        transform=transform,
    )


def main() -> None:
    args = parse_args()

    cfg = load_experiment_config(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
    )

    if args.device is not None:
        cfg.train.experiment.device = args.device
    if args.batch_size is not None:
        cfg.data.dataloader.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.data.dataloader.num_workers = args.num_workers

    seed = int(cfg.train.experiment.seed)
    seed_everything(seed)

    print("=" * 80)
    print("Merged Config")
    print("=" * 80)
    print(pretty_print_config(cfg))

    transforms = build_transforms_from_config(cfg)
    split_csv = get_effective_split_csv_path(
        cfg=cfg,
        split=args.split,
        split_csv_override=args.split_csv_override,
    )
    split_csv_path = resolve_path(split_csv)

    if not split_csv_path.exists():
        if args.split_csv_override is not None:
            raise FileNotFoundError(
                f"Override split CSV not found for {args.split}: {split_csv_path}"
            )
        raise FileNotFoundError(f"{args.split} split CSV not found: {split_csv_path}")

    dataset_cls = get_dataset_class(cfg)
    dataset = build_single_dataset(
        dataset_cls=dataset_cls,
        csv_path=split_csv,
        root_dir=cfg.data.paths.root_dir,
        transform=transforms[args.split],
    )

    loader = build_loader(
        dataset=dataset,
        batch_size=int(cfg.data.dataloader.batch_size),
        num_workers=int(cfg.data.dataloader.num_workers),
        pin_memory=bool(cfg.data.dataloader.pin_memory),
        shuffle=get_split_shuffle(cfg, args.split),
        seed=seed,
    )

    print("=" * 80)
    print("Dataset Summary")
    print("=" * 80)
    print(f"dataset name: {getattr(cfg.data, 'name', 'unknown')}")
    print(f"dataset class: {dataset_cls.__name__}")
    print(f"split: {args.split}")
    print(f"evaluated csv: {split_csv_path}")
    print(f"size: {len(dataset)}")
    print(f"class counts: {dataset.class_counts}")

    model = build_model(cfg.model)
    model_summary = get_model_summary(model)

    print("=" * 80)
    print("Model Summary")
    print("=" * 80)
    for key, value in model_summary.items():
        print(f"{key}: {value}")

    trainer = Trainer(
        model=model,
        train_cfg=cfg.train,
        device=cfg.train.experiment.device,
    )

    checkpoint_path = (
        resolve_path(args.checkpoint)
        if args.checkpoint is not None
        else resolve_path(Path(cfg.train.experiment.output_dir) / "best.pth")
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("=" * 80)
    print("Load Checkpoint")
    print("=" * 80)
    print(f"checkpoint: {checkpoint_path}")

    checkpoint = trainer.load_checkpoint(checkpoint_path, strict=True)

    if args.temperature_json is not None and args.temperature is not None:
        raise ValueError(
            "Use only one of --temperature_json or --temperature."
        )

    temperature = None
    temperature_json_path = None
    if args.temperature_json is not None:
        temperature_json_path = resolve_path(args.temperature_json)
        if not temperature_json_path.exists():
            raise FileNotFoundError(
                f"Temperature JSON not found: {temperature_json_path}"
            )
        temperature = load_temperature_value(temperature_json_path)
    elif args.temperature is not None:
        temperature = float(args.temperature)

    threshold_value = (
        float(args.threshold_override)
        if args.threshold_override is not None
        else float(cfg.train.evaluation.threshold)
    )

    print("=" * 80)
    print(f"Evaluate: {args.split}")
    print("=" * 80)

    if temperature is not None:
        print(f"temperature: {temperature:.6f}")
    else:
        print("temperature: none")
    print(f"threshold: {threshold_value:.4f}")

    metrics = trainer.evaluate(
        loader,
        split=args.split,
        temperature=temperature,
        threshold=threshold_value,
    )
    print(format_metrics(metrics))

    result = {
        "dataset_name": getattr(cfg.data, "name", "unknown"),
        "dataset_class": dataset_cls.__name__,
        "split": args.split,
        "evaluated_csv": split_csv_path.as_posix(),
        "split_csv_override_used": args.split_csv_override is not None,
        "generator_name": (
            infer_generator_name_from_csv(split_csv_path)
            if args.split_csv_override is not None and args.split == "test"
            else None
        ),
        "checkpoint": checkpoint_path.as_posix(),
        "temperature": temperature,
        "temperature_json": (
            temperature_json_path.as_posix()
            if temperature_json_path is not None
            else None
        ),
        "threshold_used": threshold_value,
        "metrics": metrics,
        "dataset_size": len(dataset),
        "class_counts": dataset.class_counts,
        "best_score_in_checkpoint": checkpoint.get("best_score", None),
        "best_epoch_in_checkpoint": checkpoint.get("best_epoch", None),
        "saved_epoch": checkpoint.get("epoch", None),
    }

    output_json = (
        resolve_path(args.output_json)
        if args.output_json is not None
        else resolve_path(Path(cfg.train.experiment.output_dir) / f"eval_{args.split}.json")
    )
    save_json(result, output_json)

    print("=" * 80)
    print("Evaluation Finished")
    print("=" * 80)
    print(f"Saved result to: {output_json}")


if __name__ == "__main__":
    main()