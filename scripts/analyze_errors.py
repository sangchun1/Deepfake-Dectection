from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from deepfake_fusion.datasets.cifake_dataset import CIFAKEDataset
from deepfake_fusion.datasets.face130k_dataset import FACE130KDataset
from deepfake_fusion.datasets.openfake_dataset import OpenFakeDataset
from deepfake_fusion.models.build_model import build_model
from deepfake_fusion.transforms.robustness import (
    build_clean_eval_transform,
    build_corrupted_eval_transform,
    get_corruption_params,
)
from deepfake_fusion.utils.config import (
    load_experiment_config,
    load_yaml,
    pretty_print_config,
    resolve_path,
)
from deepfake_fusion.utils.seed import (
    get_torch_generator,
    seed_everything,
    seed_worker,
)
from deepfake_fusion.visualization.attention_rollout import AttentionRollout
from deepfake_fusion.visualization.error_analysis import (
    build_predictions_dataframe,
    categorize_case,
    compute_group_metrics_tables,
    compute_overall_metrics_from_df,
    ensure_dir,
    infer_group_columns,
    rank_error_cases,
    save_case_montage,
    save_confusion_matrix_plot,
    save_dataframe_csv,
    save_json,
    save_split_audit_artifacts,
    short_label_name,
)
from deepfake_fusion.visualization.frequency_visualize import (
    build_frequency_metrics,
    build_frequency_visuals,
    save_frequency_run_artifacts,
    save_frequency_sample_artifacts,
)
from deepfake_fusion.visualization.gradcam import (
    GradCAM,
    apply_colormap_to_cam,
    denormalize_image_tensor,
    make_gradcam_panel,
    overlay_cam_on_image,
    resolve_target_layer,
    save_rgb_image,
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
    parser = argparse.ArgumentParser(description="Run confusion-matrix and error-case analysis.")
    parser.add_argument("--data_config", type=str, default="configs/data/openfake.yaml")
    parser.add_argument("--model_config", type=str, default="configs/model/resnet18.yaml")
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train/spatial_resnet_openfake.yaml",
    )
    parser.add_argument(
        "--robustness_config",
        type=str,
        default="configs/train/robustness.yaml",
        help="Used only when corruption != clean.",
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
        default=None,
        choices=["train", "val", "test", "custom"],
        help="Split label. If --split_csv is omitted, defaults to test. If --split_csv is provided and --split is omitted, custom is used.",
    )
    parser.add_argument(
        "--split_csv",
        type=str,
        default=None,
        help="Direct path to the evaluation split CSV. Overrides cfg.data.paths.<split>_csv.",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu / mps")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override evaluation threshold. If omitted, train.evaluation.threshold is used.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "gradcam", "rollout", "frequency"],
        help="Explanation method for top error cases.",
    )
    parser.add_argument(
        "--target_type",
        type=str,
        default="pred",
        choices=["pred", "true"],
        help="Which class to explain for Grad-CAM/rollout panels.",
    )
    parser.add_argument("--target_layer", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--head_fusion", type=str, default="mean", choices=["mean", "max", "min"])
    parser.add_argument("--discard_ratio", type=float, default=0.0)
    parser.add_argument("--start_layer", type=int, default=0)
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of hardest false positives and false negatives to save.",
    )
    parser.add_argument(
        "--corruption",
        type=str,
        default="clean",
        help="clean, jpeg, gaussian_blur, gaussian_noise, resize_down_up, ...",
    )
    parser.add_argument("--severity", type=int, default=1)
    parser.add_argument(
        "--save_individual_frequency_images",
        action="store_true",
        help="Also save input/low/high/logmag images for frequency panels.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/error_analysis",
        help="Base directory. Final output becomes save_dir/<exp_name>/<condition>/<split>.",
    )
    return parser.parse_args()


def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
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


def _to_plain_python(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_plain_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_python(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if hasattr(value, "item") and callable(getattr(value, "item", None)):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _normalize_corruption_name(name: Optional[str]) -> str:
    if name is None:
        return "clean"
    name = str(name).strip().lower()
    aliases = {
        "clean": "clean",
        "none": "clean",
        "jpeg": "jpeg",
        "jpg": "jpeg",
        "resize": "resize_down_up",
        "resize_down_up": "resize_down_up",
        "down_up_resize": "resize_down_up",
        "blur": "gaussian_blur",
        "gaussian_blur": "gaussian_blur",
        "noise": "gaussian_noise",
        "gaussian_noise": "gaussian_noise",
        "brightness_contrast": "brightness_contrast",
        "color": "brightness_contrast",
    }
    return aliases.get(name, name)


def is_clean_corruption(name: Optional[str]) -> bool:
    return _normalize_corruption_name(name) == "clean"


def build_condition_name(corruption_name: str, severity: int) -> str:
    normalized = _normalize_corruption_name(corruption_name)
    if normalized == "clean":
        return "clean"
    return f"{normalized}_s{int(severity)}"


def get_frequency_cfg_dict(cfg: Any) -> Dict[str, Any]:
    model_name = str(_cfg_get(cfg, "model", "name", default="")).lower()
    if model_name == "fusion":
        freq_cfg = _cfg_get(cfg, "model", "spectral", "frequency", default=None)
    else:
        freq_cfg = _cfg_get(cfg, "model", "frequency", default=None)

    if freq_cfg is None:
        return {}

    return {
        "mask_mode": _to_plain_python(_cfg_get(freq_cfg, "mask_mode", default="radial")),
        "radius_ratio": _to_plain_python(_cfg_get(freq_cfg, "radius_ratio", default=0.25)),
        "fft_norm": _to_plain_python(_cfg_get(freq_cfg, "fft_norm", default="ortho")),
        "high_from_residual": _to_plain_python(
            _cfg_get(freq_cfg, "high_from_residual", default=True)
        ),
        "clamp_output": _to_plain_python(_cfg_get(freq_cfg, "clamp_output", default=False)),
        "eps": _to_plain_python(_cfg_get(freq_cfg, "eps", default=1e-8)),
    }


def resolve_device(device_name: Optional[str]) -> torch.device:
    device_name = (device_name or "cuda").lower()
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_name == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dataset_class(cfg: Any):
    dataset_key = None
    if getattr(cfg.data, "dataset_class", None) is not None:
        dataset_key = str(cfg.data.dataset_class)
    elif getattr(cfg.data, "name", None) is not None:
        dataset_key = str(cfg.data.name)

    if dataset_key is None:
        raise ValueError(
            "Could not determine dataset class. Set cfg.data.dataset_class or cfg.data.name."
        )
    if dataset_key not in DATASET_REGISTRY:
        supported = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unsupported dataset '{dataset_key}'. Supported values: {supported}")
    return DATASET_REGISTRY[dataset_key]


def resolve_effective_split(split: Optional[str], split_csv: Optional[str]) -> str:
    if split_csv is not None:
        return str(split or "custom")
    return str(split or "test")


def get_split_csv_path(cfg: Any, split: str, split_csv: Optional[str] = None) -> str:
    if split_csv is not None:
        return str(split_csv)
    if split == "train":
        return cfg.data.paths.train_csv
    if split == "val":
        return cfg.data.paths.val_csv
    if split == "test":
        return cfg.data.paths.test_csv
    raise ValueError(
        f"Unsupported split: {split}. When using a custom split name, provide --split_csv."
    )


def resolve_robustness_cfg(args: argparse.Namespace) -> Optional[Any]:
    corruption_name = _normalize_corruption_name(args.corruption)
    if corruption_name == "clean":
        return None
    robustness_path = resolve_path(args.robustness_config)
    if not robustness_path.exists():
        raise FileNotFoundError(f"Robustness config not found: {robustness_path}")
    return load_yaml(robustness_path)


def resolve_corruption_params(
    corruption_name: str,
    severity: int,
    robustness_cfg: Optional[Any],
) -> Dict[str, Any]:
    normalized = _normalize_corruption_name(corruption_name)
    if normalized == "clean":
        return {"name": "clean"}
    if robustness_cfg is None:
        raise ValueError("robustness_cfg is required when corruption != clean")
    return get_corruption_params(
        robustness_cfg=robustness_cfg,
        corruption_name=normalized,
        severity=int(severity),
    )


def build_dataset(
    cfg: Any,
    split: str,
    *,
    split_csv: Optional[str] = None,
    corruption_name: str = "clean",
    severity: int = 1,
    robustness_cfg: Optional[Any] = None,
):
    normalized_corruption = _normalize_corruption_name(corruption_name)
    if normalized_corruption == "clean":
        transform = build_clean_eval_transform(cfg.data)
    else:
        transform = build_corrupted_eval_transform(
            data_cfg=cfg.data,
            corruption_name=normalized_corruption,
            severity=int(severity),
            robustness_cfg=robustness_cfg,
        )

    csv_path = get_split_csv_path(cfg, split, split_csv=split_csv)
    csv_path_resolved = resolve_path(csv_path)
    if not csv_path_resolved.exists():
        raise FileNotFoundError(f"{split} split CSV not found: {csv_path_resolved}")

    dataset_cls = get_dataset_class(cfg)
    dataset = dataset_cls(
        csv_path=csv_path,
        root_dir=cfg.data.paths.root_dir,
        transform=transform,
    )
    return dataset_cls, dataset


def build_loader(
    dataset: Any,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    use_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker if use_workers else None,
        generator=get_torch_generator(seed),
        persistent_workers=use_workers,
    )


def build_split_to_csv_map(
    cfg: Any,
    *,
    effective_split: str,
    split_csv: Optional[str],
) -> Dict[str, Path]:
    split_to_csv = build_split_to_csv_map(
        cfg,
        effective_split=effective_split,
        split_csv=args.split_csv,
    )
    if split_csv is not None:
        split_to_csv[effective_split] = resolve_path(split_csv)
    return split_to_csv


def load_checkpoint_to_model(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
    strict: bool = True,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def batch_logits_to_prob_fake_pred(
    logits: torch.Tensor,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if logits.ndim == 1:
        prob_fake = torch.sigmoid(logits)
        pred = (prob_fake >= threshold).long()
        return prob_fake.detach().cpu().numpy(), pred.detach().cpu().numpy()

    if logits.ndim == 2 and logits.size(1) == 1:
        prob_fake = torch.sigmoid(logits[:, 0])
        pred = (prob_fake >= threshold).long()
        return prob_fake.detach().cpu().numpy(), pred.detach().cpu().numpy()

    if logits.ndim == 2 and logits.size(1) == 2:
        probs = torch.softmax(logits, dim=1)
        prob_fake = probs[:, 1]
        pred = (prob_fake >= threshold).long()
        return prob_fake.detach().cpu().numpy(), pred.detach().cpu().numpy()

    raise ValueError(
        f"Expected binary logits shape [B], [B,1], or [B,2], got {tuple(logits.shape)}"
    )


def single_logits_to_prob_fake_pred(logits: torch.Tensor, threshold: float) -> Tuple[float, int]:
    probs, preds = batch_logits_to_prob_fake_pred(logits, threshold=threshold)
    return float(probs[0]), int(preds[0])


def infer_explain_method(cfg: Any, requested_method: str) -> str:
    if requested_method != "auto":
        return requested_method

    model_name = str(getattr(cfg.model, "name", "")).lower()
    backbone = getattr(cfg.model, "backbone", None)
    backbone_name = str(getattr(backbone, "name", "")).lower()

    if model_name == "spai":
        return "frequency"
    if model_name == "fusion":
        return "gradcam"
    if model_name == "vit" or backbone_name.startswith("vit"):
        return "rollout"
    return "gradcam"


def is_fusion_model(model: torch.nn.Module) -> bool:
    return hasattr(model, "spatial_branch") and hasattr(model, "spectral_branch")


def resolve_frequency_explain_components(
    model: torch.nn.Module,
) -> Tuple[torch.nn.Module, Any]:
    if hasattr(model, "frequency_encoder") and hasattr(model, "extract_features"):
        return model, model.frequency_encoder

    if (
        hasattr(model, "spectral_branch")
        and hasattr(model, "extract_features")
        and hasattr(model.spectral_branch, "frequency_encoder")
    ):
        return model, model.spectral_branch.frequency_encoder

    raise ValueError(
        "Frequency explanation requires a SPAI-like model or fusion model with spectral_branch."
    )


def build_explainer(model: torch.nn.Module, method: str, args: argparse.Namespace):
    if method == "gradcam":
        target_layer = resolve_target_layer(model, args.target_layer)
        return GradCAM(model=model, target_layer=target_layer), target_layer

    if method == "rollout":
        if is_fusion_model(model):
            raise ValueError("Attention rollout is not supported for the current fusion model.")
        return (
            AttentionRollout(
                model=model,
                head_fusion=args.head_fusion,
                discard_ratio=args.discard_ratio,
                start_layer=args.start_layer,
            ),
            None,
        )

    if method == "frequency":
        return None, None

    raise ValueError(f"Unsupported explanation method: {method}")


def _extract_batch_item(value: Any, idx: int) -> Any:
    if torch.is_tensor(value):
        item = value[idx]
        if item.ndim == 0:
            return item.item()
        return item.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        item = value[idx]
        if np.ndim(item) == 0:
            return item.item()
        return item.tolist()
    if isinstance(value, (list, tuple)):
        return value[idx]
    return value


def collect_prediction_records(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    threshold: float,
    split: str,
    condition_name: str,
    corruption_name: str,
    severity: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    model.eval()
    sample_index = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            logits = model(images)

        prob_fake, pred_label = batch_logits_to_prob_fake_pred(logits, threshold=threshold)
        true_label = labels.detach().cpu().numpy().astype(np.int64)

        batch_size = len(true_label)
        for i in range(batch_size):
            filepath = str(_extract_batch_item(batch.get("filepath", ""), i))
            record: Dict[str, Any] = {
                "index": int(sample_index),
                "filepath": filepath,
                "filename": Path(filepath).name if filepath else None,
                "true_label": int(true_label[i]),
                "pred_label": int(pred_label[i]),
                "prob_fake": float(prob_fake[i]),
                "prob_real": float(1.0 - float(prob_fake[i])),
                "correct": bool(int(true_label[i]) == int(pred_label[i])),
                "case_group": categorize_case(int(true_label[i]), int(pred_label[i])),
                "split": split,
                "condition": condition_name,
                "corruption": corruption_name,
                "severity": int(severity),
            }

            for key, value in batch.items():
                if key in {"image", "label", "filepath"}:
                    continue
                try:
                    record[key] = _to_plain_python(_extract_batch_item(value, i))
                except Exception:
                    record[key] = _to_plain_python(value)

            records.append(record)
            sample_index += 1

    return records


def build_selected_case_specs(
    ranked_cases: Mapping[str, pd.DataFrame],
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    mapping = {
        "false_positive": "false_positive",
        "false_negative": "false_negative",
    }

    for key, df in ranked_cases.items():
        error_type = mapping.get(key, key)
        for rank, row in enumerate(df.to_dict(orient="records"), start=1):
            specs.append(
                {
                    "error_type": error_type,
                    "rank": int(rank),
                    **row,
                }
            )

    specs.sort(key=lambda x: (str(x["error_type"]), int(x["rank"])))
    return specs


def generate_error_case_artifacts(
    *,
    model: torch.nn.Module,
    dataset: Any,
    case_specs: Sequence[Mapping[str, Any]],
    device: torch.device,
    threshold: float,
    method: str,
    args: argparse.Namespace,
    save_root: Path,
    mean: Sequence[float],
    std: Sequence[float],
    frequency_cfg: Mapping[str, Any],
    condition_name: str,
    corruption_name: str,
    severity: int,
    corruption_params: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    saved_cases: List[Dict[str, Any]] = []
    explainer = None
    resolved_target_layer = None
    run_artifacts: Dict[str, str] = {}
    frequency_run_saved = False

    if method in {"gradcam", "rollout"}:
        explainer, resolved_target_layer = build_explainer(model, method, args)

    try:
        for case in case_specs:
            idx = int(case["index"])
            sample = dataset[idx]
            image_tensor = sample["image"]
            true_label = int(sample["label"].item())
            filepath = str(sample.get("filepath", case.get("filepath", "")))

            x = image_tensor.unsqueeze(0).to(device)
            target_class = true_label
            case_dir = ensure_dir(save_root / "cases" / str(case["error_type"]))

            if method == "frequency":
                with torch.no_grad():
                    logits = model(x)
                    pred_prob, pred_label = single_logits_to_prob_fake_pred(logits, threshold)
                    feature_model, frequency_encoder = resolve_frequency_explain_components(model)
                    explain_out = feature_model.extract_features(x, return_dict=True)
                    split_dict = frequency_encoder.split_spectrum(x)

                target_class = pred_label if args.target_type == "pred" else true_label

                if not frequency_run_saved:
                    mask_info = {
                        **dict(frequency_cfg),
                        "image_size": list(image_tensor.shape[-2:]),
                        "condition": condition_name,
                        "corruption": corruption_name,
                        "severity": int(severity),
                        "corruption_params": _to_plain_python(corruption_params),
                    }
                    run_artifacts = save_frequency_run_artifacts(
                        save_dir=save_root,
                        low_mask=split_dict["low_mask"],
                        high_mask=split_dict["high_mask"],
                        mask_info=mask_info,
                    )
                    frequency_run_saved = True

                x_low = explain_out["x_low"][0].detach().cpu()
                x_high = explain_out["x_high"][0].detach().cpu()
                visuals = build_frequency_visuals(
                    input_tensor=image_tensor,
                    x_low=x_low,
                    x_high=x_high,
                    mean=mean,
                    std=std,
                    fft_norm=str(frequency_cfg.get("fft_norm", "ortho")),
                    high_channel_reduce="mean",
                )
                metrics = build_frequency_metrics(
                    input_tensor=image_tensor,
                    x_low=x_low,
                    x_high=x_high,
                    explain_dict=explain_out,
                    true_label=true_label,
                    pred_label=pred_label,
                    pred_prob=pred_prob,
                    source_filepath=filepath,
                    sample_index=idx,
                    group=str(case["error_type"]),
                    frequency_cfg=frequency_cfg,
                )
                metrics["rank"] = int(case["rank"])
                metrics["error_type"] = str(case["error_type"])
                metrics["condition"] = condition_name
                metrics["corruption"] = corruption_name
                metrics["severity"] = int(severity)
                metrics["corruption_params"] = _to_plain_python(corruption_params)

                sample_name = (
                    f"{int(case['rank']):02d}"
                    f"_idx-{idx:05d}"
                    f"_true-{short_label_name(true_label)}"
                    f"_pred-{short_label_name(pred_label)}"
                    f"_p-{pred_prob:.4f}"
                )
                sample_dir = ensure_dir(case_dir / sample_name)
                saved_files = save_frequency_sample_artifacts(
                    save_dir=sample_dir,
                    visuals=visuals,
                    metrics=metrics,
                    save_individual_images=bool(args.save_individual_frequency_images),
                    save_panel=True,
                )
                panel_path = saved_files.get("panel.png")
                artifact = {
                    **{k: _to_plain_python(v) for k, v in case.items()},
                    "method": method,
                    "target_type": args.target_type,
                    "target_class": int(target_class),
                    "panel_path": panel_path,
                    "saved_dir": sample_dir.as_posix(),
                    "saved_files": saved_files,
                    "frequency_run_artifacts": run_artifacts,
                }
                save_json(artifact, sample_dir / "case_record.json")
                saved_cases.append(artifact)
                continue

            if method == "rollout":
                result = explainer.generate(x, target_class=None)
                logits = result["logits"]
                pred_prob, pred_label = single_logits_to_prob_fake_pred(logits, threshold)
                target_class = pred_label if args.target_type == "pred" else true_label
            else:
                with torch.no_grad():
                    logits = model(x)
                pred_prob, pred_label = single_logits_to_prob_fake_pred(logits, threshold)
                target_class = pred_label if args.target_type == "pred" else true_label
                result = explainer.generate(x, target_class=target_class)

            input_rgb = denormalize_image_tensor(image_tensor, mean=mean, std=std)
            cam = result["cam"]
            heatmap_rgb = apply_colormap_to_cam(cam)
            overlay_rgb = overlay_cam_on_image(input_rgb, cam, alpha=args.alpha)

            text_lines = [
                (
                    f"error_type={case['error_type']} | rank={int(case['rank'])} | "
                    f"idx={idx} | method={method}"
                ),
                (
                    f"true={short_label_name(true_label)}({true_label}) | "
                    f"pred={short_label_name(pred_label)}({pred_label}) | prob_fake={pred_prob:.4f}"
                ),
                f"target_type={args.target_type} | target_class={int(target_class)} | condition={condition_name}",
                f"path={Path(filepath).name}",
            ]
            panel = make_gradcam_panel(
                original_rgb=input_rgb,
                heatmap_rgb=heatmap_rgb,
                overlay_rgb=overlay_rgb,
                text_lines=text_lines,
            )

            filename = (
                f"{int(case['rank']):02d}"
                f"_idx-{idx:05d}"
                f"_true-{short_label_name(true_label)}"
                f"_pred-{short_label_name(pred_label)}"
                f"_p-{pred_prob:.4f}.png"
            )
            save_path = case_dir / filename
            save_rgb_image(panel, save_path)

            artifact = {
                **{k: _to_plain_python(v) for k, v in case.items()},
                "method": method,
                "target_type": args.target_type,
                "target_class": int(target_class),
                "resolved_target_layer": str(resolved_target_layer) if resolved_target_layer is not None else None,
                "panel_path": save_path.as_posix(),
            }
            save_json(artifact, save_path.with_suffix(".json"))
            saved_cases.append(artifact)

    finally:
        if explainer is not None and hasattr(explainer, "remove_hooks"):
            explainer.remove_hooks()

    return saved_cases


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

    effective_split = resolve_effective_split(args.split, args.split_csv)

    normalized_corruption = _normalize_corruption_name(args.corruption)
    robustness_cfg = resolve_robustness_cfg(args)
    corruption_params = resolve_corruption_params(
        normalized_corruption,
        int(args.severity),
        robustness_cfg,
    )
    condition_name = build_condition_name(normalized_corruption, int(args.severity))

    seed = int(cfg.train.experiment.seed)
    seed_everything(seed)

    print("=" * 80)
    print("Merged Config")
    print("=" * 80)
    print(pretty_print_config(cfg))

    dataset_cls, dataset = build_dataset(
        cfg,
        effective_split,
        split_csv=args.split_csv,
        corruption_name=normalized_corruption,
        severity=int(args.severity),
        robustness_cfg=robustness_cfg,
    )
    loader = build_loader(
        dataset=dataset,
        batch_size=int(cfg.data.dataloader.batch_size),
        num_workers=int(cfg.data.dataloader.num_workers),
        pin_memory=bool(cfg.data.dataloader.pin_memory),
        seed=seed,
    )

    print("=" * 80)
    print("Dataset Summary")
    print("=" * 80)
    print(f"dataset name: {getattr(cfg.data, 'name', 'unknown')}")
    print(f"dataset class: {dataset_cls.__name__}")
    print(f"split: {effective_split}")
    print(f"split_csv: {resolve_path(args.split_csv).as_posix() if args.split_csv is not None else resolve_path(get_split_csv_path(cfg, effective_split)).as_posix()}")
    print(f"condition: {condition_name}")
    print(f"size: {len(dataset)}")
    print(f"class counts: {dataset.class_counts}")

    model = build_model(cfg.model)
    device = resolve_device(cfg.train.experiment.device)
    model = model.to(device)
    model.eval()

    checkpoint_path = (
        resolve_path(args.checkpoint)
        if args.checkpoint is not None
        else resolve_path(Path(cfg.train.experiment.output_dir) / "best.pth")
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint_to_model(model, checkpoint_path, device=device, strict=True)
    threshold = float(
        args.threshold
        if args.threshold is not None
        else _cfg_get(cfg.train, "evaluation", "threshold", default=0.5)
    )

    experiment_name = Path(cfg.train.experiment.output_dir).name
    save_root = ensure_dir(Path(args.save_dir) / experiment_name / condition_name / effective_split)
    predictions_dir = ensure_dir(save_root / "predictions")
    metrics_dir = ensure_dir(save_root / "metrics")
    cases_dir = ensure_dir(save_root / "cases")
    split_audit_dir = ensure_dir(save_root / "split_audit")

    split_to_csv = build_split_to_csv_map(
        cfg,
        effective_split=effective_split,
        split_csv=args.split_csv,
    )
    split_audit_report = save_split_audit_artifacts(split_to_csv, split_audit_dir)

    print("=" * 80)
    print("Collect predictions")
    print("=" * 80)
    records = collect_prediction_records(
        model,
        loader,
        device=device,
        threshold=threshold,
        split=effective_split,
        condition_name=condition_name,
        corruption_name=normalized_corruption,
        severity=0 if is_clean_corruption(normalized_corruption) else int(args.severity),
    )
    predictions_df = build_predictions_dataframe(records)
    save_dataframe_csv(predictions_df, predictions_dir / "predictions.csv")

    overall = compute_overall_metrics_from_df(predictions_df)
    cm_count = np.asarray(overall["confusion"]["count"], dtype=np.int64)
    save_confusion_matrix_plot(
        cm_count,
        metrics_dir / "confusion_matrix_count.png",
        normalize=False,
        title=f"Confusion Matrix ({effective_split}, {condition_name})",
    )
    save_confusion_matrix_plot(
        cm_count,
        metrics_dir / "confusion_matrix_normalized.png",
        normalize=True,
        title=f"Confusion Matrix Normalized ({effective_split}, {condition_name})",
    )

    group_columns = infer_group_columns(predictions_df)
    group_tables = compute_group_metrics_tables(predictions_df, group_columns)
    saved_group_csvs: Dict[str, str] = {}
    for col, table in group_tables.items():
        csv_path = metrics_dir / f"metrics_by_{col}.csv"
        save_dataframe_csv(table, csv_path)
        saved_group_csvs[col] = csv_path.as_posix()

    ranked = rank_error_cases(predictions_df, top_k=int(args.top_k))
    fp_df = ranked["false_positive"]
    fn_df = ranked["false_negative"]
    if not fp_df.empty:
        save_dataframe_csv(fp_df, predictions_dir / "false_positive_topk.csv")
    if not fn_df.empty:
        save_dataframe_csv(fn_df, predictions_dir / "false_negative_topk.csv")

    method = infer_explain_method(cfg, args.method)
    mean = list(_cfg_get(cfg.data, "image", "mean", default=[0.485, 0.456, 0.406]))
    std = list(_cfg_get(cfg.data, "image", "std", default=[0.229, 0.224, 0.225]))
    frequency_cfg = get_frequency_cfg_dict(cfg)

    case_specs = build_selected_case_specs(ranked)
    saved_cases = generate_error_case_artifacts(
        model=model,
        dataset=dataset,
        case_specs=case_specs,
        device=device,
        threshold=threshold,
        method=method,
        args=args,
        save_root=cases_dir,
        mean=mean,
        std=std,
        frequency_cfg=frequency_cfg,
        condition_name=condition_name,
        corruption_name=normalized_corruption,
        severity=0 if is_clean_corruption(normalized_corruption) else int(args.severity),
        corruption_params=corruption_params,
    )

    saved_cases_df = pd.DataFrame(saved_cases)
    if not saved_cases_df.empty:
        save_dataframe_csv(saved_cases_df, cases_dir / "selected_cases.csv")

    fp_cases = [case for case in saved_cases if case.get("error_type") == "false_positive"]
    fn_cases = [case for case in saved_cases if case.get("error_type") == "false_negative"]

    fp_montage = save_case_montage(
        fp_cases,
        cases_dir / "false_positive_montage.png",
        title=f"False Positive Top-{len(fp_cases)}",
    )
    fn_montage = save_case_montage(
        fn_cases,
        cases_dir / "false_negative_montage.png",
        title=f"False Negative Top-{len(fn_cases)}",
    )

    summary = {
        "dataset_name": getattr(cfg.data, "name", "unknown"),
        "dataset_class": dataset_cls.__name__,
        "split": effective_split,
        "split_csv": resolve_path(args.split_csv).as_posix() if args.split_csv is not None else resolve_path(get_split_csv_path(cfg, effective_split)).as_posix(),
        "condition": condition_name,
        "corruption": normalized_corruption,
        "severity": 0 if is_clean_corruption(normalized_corruption) else int(args.severity),
        "corruption_params": _to_plain_python(corruption_params),
        "checkpoint": checkpoint_path.as_posix(),
        "threshold": threshold,
        "method": method,
        "target_type": args.target_type,
        "top_k": int(args.top_k),
        "dataset_size": int(len(dataset)),
        "class_counts": _to_plain_python(dataset.class_counts),
        "best_score_in_checkpoint": checkpoint.get("best_score", None),
        "best_epoch_in_checkpoint": checkpoint.get("best_epoch", None),
        "saved_epoch": checkpoint.get("epoch", None),
        "overall": overall,
        "group_metric_csvs": saved_group_csvs,
        "prediction_csv": (predictions_dir / "predictions.csv").as_posix(),
        "false_positive_csv": (predictions_dir / "false_positive_topk.csv").as_posix() if not fp_df.empty else None,
        "false_negative_csv": (predictions_dir / "false_negative_topk.csv").as_posix() if not fn_df.empty else None,
        "confusion_matrix_count_png": (metrics_dir / "confusion_matrix_count.png").as_posix(),
        "confusion_matrix_normalized_png": (metrics_dir / "confusion_matrix_normalized.png").as_posix(),
        "selected_cases_csv": (cases_dir / "selected_cases.csv").as_posix() if not saved_cases_df.empty else None,
        "false_positive_montage_png": fp_montage,
        "false_negative_montage_png": fn_montage,
        "split_audit_dir": split_audit_dir.as_posix(),
        "split_audit_report": split_audit_report,
    }
    save_json(summary, save_root / "summary.json")

    print("=" * 80)
    print("Error Analysis Finished")
    print("=" * 80)
    print(f"save_dir: {save_root}")
    print(f"prediction_csv: {predictions_dir / 'predictions.csv'}")
    print(f"method: {method}")
    print(f"false positives saved: {len(fp_cases)}")
    print(f"false negatives saved: {len(fn_cases)}")


if __name__ == "__main__":
    main()
