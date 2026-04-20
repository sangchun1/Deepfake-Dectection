from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import torch
from torch.utils.data import DataLoader

from deepfake_fusion.datasets.cifake_dataset import CIFAKEDataset
from deepfake_fusion.datasets.face130k_dataset import FACE130KDataset
from deepfake_fusion.datasets.openfake_dataset import OpenFakeDataset
from deepfake_fusion.engine.trainer import Trainer
from deepfake_fusion.metrics.classification import ClassificationMeter
from deepfake_fusion.models.build_model import build_model, get_model_summary
from deepfake_fusion.transforms.image_aug import build_transforms_from_config
from deepfake_fusion.utils.calibration import TemperatureScaler
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
    "cifake": CIFAKEDataset,
    "CIFAKEDataset": CIFAKEDataset,
    "face130k": FACE130KDataset,
    "FACE130KDataset": FACE130KDataset,
    "openfake": OpenFakeDataset,
    "OpenFakeDataset": OpenFakeDataset,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a post-hoc temperature scaler on a validation split."
    )

    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data/openfake.yaml",
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
        default="configs/train/spatial_resnet_openfake.yaml",
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
        default="val",
        choices=["train", "val", "test"],
        help="Which split to use for fitting the temperature. Usually val.",
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
        "--threshold",
        type=float,
        default=None,
        help="Threshold used only for reporting before/after metrics.",
    )
    parser.add_argument(
        "--temperature_init",
        type=float,
        default=1.0,
        help="Initial temperature value before optimization.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=50,
        help="Max LBFGS iterations for temperature fitting.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="LBFGS learning rate for temperature fitting.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save fitted temperature JSON.",
    )
    parser.add_argument(
        "--split_csv_override",
        type=str,
        default=None,
        help=(
            "Optional CSV path to use instead of cfg.data.paths.<split>_csv. "
            "Useful for per-generator validation calibration."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional fitting diagnostics.",
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


def _extract_logits_from_model_output(model_output: Any) -> torch.Tensor:
    if torch.is_tensor(model_output):
        return model_output

    if isinstance(model_output, dict):
        for key in ("logits", "fused_logits", "output", "pred"):
            value = model_output.get(key)
            if torch.is_tensor(value):
                return value

    raise TypeError(
        "Model output must be a tensor or a dict containing one of: "
        "'logits', 'fused_logits', 'output', 'pred'."
    )


@torch.no_grad()
def collect_logits_and_targets(
    trainer: Trainer,
    loader: DataLoader,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    trainer.model.eval()
    meter = ClassificationMeter()
    logits_list = []
    targets_list = []

    iterator = loader
    if trainer.use_tqdm:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(loader, desc="Collect logits", leave=False)
        except Exception:
            pass

    for batch in iterator:
        batch = trainer._move_batch_to_device(batch)
        images = batch["image"]
        labels = batch["label"]

        with trainer._autocast_context():
            model_output = trainer.model(images)
            logits = _extract_logits_from_model_output(model_output)
            loss = trainer.criterion(logits, labels)

        logits_cpu = logits.detach().cpu()
        labels_cpu = labels.detach().cpu()
        logits_list.append(logits_cpu)
        targets_list.append(labels_cpu)

        meter.update(
            logits=logits_cpu,
            targets=labels_cpu,
            loss=loss.detach().cpu(),
            threshold=trainer.threshold,
        )

        if trainer.use_tqdm and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{meter.loss_meter.avg:.4f}")

    if not logits_list:
        raise ValueError("No batches were processed while collecting logits.")

    logits_all = torch.cat(logits_list, dim=0)
    targets_all = torch.cat(targets_list, dim=0)
    metrics = meter.compute()
    return logits_all, targets_all, metrics


@torch.no_grad()
def compute_metrics_with_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    temperature_scaler: TemperatureScaler,
    threshold: float,
    loss_value: Optional[float] = None,
) -> Dict[str, float]:
    probs = temperature_scaler.predict_proba(logits).detach().cpu()
    meter = ClassificationMeter()
    meter.update(
        probs=probs,
        targets=targets.detach().cpu(),
        loss=loss_value,
        threshold=threshold,
    )
    return meter.compute()


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

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(getattr(cfg.train.evaluation, "threshold", 0.5))
    )

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
    print(f"fit split: {args.split}")
    print(f"csv: {split_csv_path}")
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

    print("=" * 80)
    print(f"Collect Logits: {args.split}")
    print("=" * 80)
    logits, targets, metrics_before = collect_logits_and_targets(trainer, loader)
    print("before calibration:")
    print(format_metrics(metrics_before))

    scaler = TemperatureScaler(temperature=args.temperature_init).to(trainer.device)
    fit_summary = scaler.fit(
        logits=logits.to(trainer.device),
        targets=targets.to(trainer.device),
        max_iter=args.max_iter,
        lr=args.lr,
        verbose=args.verbose,
    )

    nll_after = fit_summary["nll_after"]
    metrics_after = compute_metrics_with_temperature(
        logits=logits,
        targets=targets,
        temperature_scaler=scaler.cpu(),
        threshold=threshold,
        loss_value=nll_after,
    )

    print("=" * 80)
    print("Calibration Result")
    print("=" * 80)
    print(f"temperature={fit_summary['temperature']:.6f}")
    print(f"nll_before={fit_summary['nll_before']:.6f}")
    print(f"nll_after={fit_summary['nll_after']:.6f}")
    print("after calibration:")
    print(format_metrics(metrics_after))

    default_output_name = "temperature.json" if args.split == "val" else f"temperature_{args.split}.json"
    output_json = (
        resolve_path(args.output_json)
        if args.output_json is not None
        else resolve_path(Path(cfg.train.experiment.output_dir) / default_output_name)
    )

    result = {
        "dataset_name": getattr(cfg.data, "name", "unknown"),
        "dataset_class": dataset_cls.__name__,
        "fit_split": args.split,
        "evaluated_csv": split_csv_path.as_posix(),
        "split_csv_override_used": args.split_csv_override is not None,
        "generator_name": (
            infer_generator_name_from_csv(split_csv_path)
            if args.split_csv_override is not None and args.split in {"val", "test"}
            else None
        ),
        "checkpoint": checkpoint_path.as_posix(),
        "threshold_for_reporting": threshold,
        "temperature_init": float(args.temperature_init),
        "temperature": fit_summary["temperature"],
        "nll_before": fit_summary["nll_before"],
        "nll_after": fit_summary["nll_after"],
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "dataset_size": len(dataset),
        "class_counts": dataset.class_counts,
        "best_score_in_checkpoint": checkpoint.get("best_score", None),
        "best_epoch_in_checkpoint": checkpoint.get("best_epoch", None),
        "saved_epoch": checkpoint.get("epoch", None),
    }
    save_json(result, output_json)

    print("=" * 80)
    print("Temperature Calibration Finished")
    print("=" * 80)
    print(f"Saved result to: {output_json}")


if __name__ == "__main__":
    main()
