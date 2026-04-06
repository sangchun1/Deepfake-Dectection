from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deepfake_fusion.metrics.classification import (
    compute_classification_metrics,
    compute_confusion_details,
)


BASE_PREDICTION_COLUMNS = {
    "index",
    "filepath",
    "filename",
    "true_label",
    "pred_label",
    "prob_fake",
    "prob_real",
    "correct",
    "case_group",
    "condition",
    "corruption",
    "severity",
    "split",
}


def _to_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def ensure_dir(path: str | Path) -> Path:
    path = _to_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: str | Path) -> Path:
    path = _to_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Mapping[str, Any], path: str | Path, indent: int = 2) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, allow_nan=True)


def save_dataframe_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    path = ensure_parent(path)
    df.to_csv(path, index=index, encoding="utf-8-sig")


def short_label_name(label: int) -> str:
    return "real" if int(label) == 0 else "fake"


def categorize_case(true_label: int, pred_label: int) -> str:
    true_label = int(true_label)
    pred_label = int(pred_label)
    if true_label == 0 and pred_label == 0:
        return "correct_real"
    if true_label == 1 and pred_label == 1:
        return "correct_fake"
    if true_label == 0 and pred_label == 1:
        return "wrong_real"
    if true_label == 1 and pred_label == 0:
        return "wrong_fake"
    return "unknown"


def build_predictions_dataframe(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame([dict(record) for record in records])
    if df.empty:
        return df

    ordered = [
        "index",
        "filepath",
        "filename",
        "true_label",
        "pred_label",
        "prob_fake",
        "prob_real",
        "correct",
        "case_group",
        "split",
        "condition",
        "corruption",
        "severity",
    ]
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining]


def compute_overall_metrics_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        raise ValueError("predictions dataframe is empty")

    y_true = df["true_label"].astype(int).to_numpy()
    y_pred = df["pred_label"].astype(int).to_numpy()
    y_prob = df["prob_fake"].astype(float).to_numpy()

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        num_classes=2,
    )
    details = compute_confusion_details(y_true=y_true, y_pred=y_pred, num_classes=2)

    cm = np.asarray(details["confusion_matrix"], dtype=np.int64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(np.float64),
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums > 0,
    )

    return {
        "metrics": metrics,
        "confusion": {
            "count": cm.tolist(),
            "normalized": cm_norm.tolist(),
            "tn": int(details.get("tn", 0)),
            "fp": int(details.get("fp", 0)),
            "fn": int(details.get("fn", 0)),
            "tp": int(details.get("tp", 0)),
        },
        "num_samples": int(len(df)),
    }


def _format_cell(value: float, normalize: bool) -> str:
    if normalize:
        return f"{value:.3f}"
    return f"{int(round(value))}"


def save_confusion_matrix_plot(
    cm: np.ndarray,
    save_path: str | Path,
    *,
    class_names: Sequence[str] = ("real", "fake"),
    normalize: bool = False,
    title: Optional[str] = None,
) -> None:
    cm = np.asarray(cm, dtype=np.float64)
    if cm.shape != (2, 2):
        raise ValueError(f"Expected 2x2 confusion matrix, got {cm.shape}")

    display = cm.copy()
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(
            display,
            row_sums,
            out=np.zeros_like(display, dtype=np.float64),
            where=row_sums > 0,
        )

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(display)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1], labels=list(class_names))
    ax.set_yticks([0, 1], labels=list(class_names))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or ("Confusion Matrix (Normalized)" if normalize else "Confusion Matrix"))

    threshold = float(np.nanmax(display) / 2.0) if np.size(display) > 0 else 0.0
    for i in range(2):
        for j in range(2):
            value = float(display[i, j])
            ax.text(
                j,
                i,
                _format_cell(value, normalize=normalize),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=11,
                fontweight="bold",
            )

    fig.tight_layout()
    save_path = ensure_parent(save_path)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def infer_group_columns(
    df: pd.DataFrame,
    *,
    preferred: Optional[Sequence[str]] = None,
    max_unique_absolute: int = 50,
    max_unique_ratio: float = 0.5,
) -> List[str]:
    if df.empty:
        return []

    preferred = list(preferred or ["generator", "source", "group", "mode", "type"])
    candidates: List[str] = []

    for col in preferred:
        if col in df.columns and col not in BASE_PREDICTION_COLUMNS:
            candidates.append(col)

    for col in df.columns:
        if col in BASE_PREDICTION_COLUMNS or col in candidates:
            continue
        series = df[col]
        nunique = int(series.nunique(dropna=True))
        if nunique < 2:
            continue
        if nunique > max_unique_absolute:
            continue
        if nunique > max(2, int(math.ceil(len(df) * max_unique_ratio))):
            continue
        candidates.append(col)

    return candidates


def _group_metrics_rows(df: pd.DataFrame, group_col: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if group_col not in df.columns:
        return rows

    for group_name, group_df in df.groupby(group_col, dropna=False):
        group_df = group_df.reset_index(drop=True)
        if group_df.empty:
            continue

        y_true = group_df["true_label"].astype(int).to_numpy()
        y_pred = group_df["pred_label"].astype(int).to_numpy()
        y_prob = group_df["prob_fake"].astype(float).to_numpy()

        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            num_classes=2,
        )
        details = compute_confusion_details(y_true=y_true, y_pred=y_pred, num_classes=2)
        cm = np.asarray(details["confusion_matrix"], dtype=np.int64)

        row: Dict[str, Any] = {
            "group_column": group_col,
            group_col: group_name,
            "num_samples": int(len(group_df)),
            "num_real": int((group_df["true_label"] == 0).sum()),
            "num_fake": int((group_df["true_label"] == 1).sum()),
            "accuracy": float(metrics.get("accuracy", math.nan)),
            "precision": float(metrics.get("precision", math.nan)),
            "recall": float(metrics.get("recall", math.nan)),
            "f1": float(metrics.get("f1", math.nan)),
            "auc": float(metrics.get("auc", math.nan)),
            "tn": int(details.get("tn", 0)),
            "fp": int(details.get("fp", 0)),
            "fn": int(details.get("fn", 0)),
            "tp": int(details.get("tp", 0)),
            "confusion_matrix": json.dumps(cm.tolist(), ensure_ascii=False),
        }
        rows.append(row)

    rows.sort(key=lambda x: (-int(x["num_samples"]), str(x.get(group_col))))
    return rows


def compute_group_metrics_tables(
    df: pd.DataFrame,
    group_columns: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for col in group_columns:
        rows = _group_metrics_rows(df, col)
        if rows:
            tables[col] = pd.DataFrame(rows)
    return tables


def rank_error_cases(df: pd.DataFrame, top_k: int = 20) -> Dict[str, pd.DataFrame]:
    if df.empty:
        empty = df.copy()
        return {"false_positive": empty, "false_negative": empty}

    fp = df[(df["true_label"] == 0) & (df["pred_label"] == 1)].copy()
    fn = df[(df["true_label"] == 1) & (df["pred_label"] == 0)].copy()

    if not fp.empty:
        fp = fp.sort_values(
            by=["prob_fake", "index"],
            ascending=[False, True],
        ).head(top_k)

    if not fn.empty:
        fn = fn.sort_values(
            by=["prob_fake", "index"],
            ascending=[True, True],
        ).head(top_k)

    return {
        "false_positive": fp.reset_index(drop=True),
        "false_negative": fn.reset_index(drop=True),
    }


def summarize_split_csvs(
    split_to_csv: Mapping[str, str | Path],
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    summary_rows: List[Dict[str, Any]] = []

    for split, csv_path in split_to_csv.items():
        path = _to_path(csv_path)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        frames[split] = df.copy()

        label_counts: Dict[Any, int] = {}
        if "label" in df.columns:
            label_counts = df["label"].value_counts(dropna=False).to_dict()

        summary_rows.append(
            {
                "split": split,
                "csv_path": path.as_posix(),
                "num_rows": int(len(df)),
                "num_unique_filepath": int(df["filepath"].nunique(dropna=True)) if "filepath" in df.columns else math.nan,
                "num_duplicate_filepath": int(max(len(df) - df["filepath"].nunique(dropna=True), 0)) if "filepath" in df.columns else math.nan,
                "num_real": int(label_counts.get(0, label_counts.get("0", 0))) if label_counts else math.nan,
                "num_fake": int(label_counts.get(1, label_counts.get("1", 0))) if label_counts else math.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return frames, summary_df


def compute_split_overlap_tables(frames: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    split_names = list(frames.keys())
    rows_filepath: List[Dict[str, Any]] = []
    rows_filename: List[Dict[str, Any]] = []

    for i, split_a in enumerate(split_names):
        for split_b in split_names[i + 1 :]:
            df_a = frames[split_a]
            df_b = frames[split_b]

            if "filepath" in df_a.columns and "filepath" in df_b.columns:
                set_a = set(df_a["filepath"].astype(str).tolist())
                set_b = set(df_b["filepath"].astype(str).tolist())
                overlap = sorted(set_a & set_b)
                rows_filepath.append(
                    {
                        "split_a": split_a,
                        "split_b": split_b,
                        "overlap_count": int(len(overlap)),
                        "example_1": overlap[0] if len(overlap) > 0 else None,
                        "example_2": overlap[1] if len(overlap) > 1 else None,
                        "example_3": overlap[2] if len(overlap) > 2 else None,
                    }
                )

                name_a = set(Path(x).name for x in set_a)
                name_b = set(Path(x).name for x in set_b)
                overlap_name = sorted(name_a & name_b)
                rows_filename.append(
                    {
                        "split_a": split_a,
                        "split_b": split_b,
                        "overlap_count": int(len(overlap_name)),
                        "example_1": overlap_name[0] if len(overlap_name) > 0 else None,
                        "example_2": overlap_name[1] if len(overlap_name) > 1 else None,
                        "example_3": overlap_name[2] if len(overlap_name) > 2 else None,
                    }
                )

    return {
        "filepath_overlap": pd.DataFrame(rows_filepath),
        "filename_overlap": pd.DataFrame(rows_filename),
    }


def compute_split_distribution_tables(
    frames: Mapping[str, pd.DataFrame],
    candidate_columns: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for col in candidate_columns:
        rows: List[Dict[str, Any]] = []
        for split, df in frames.items():
            if col not in df.columns:
                continue
            vc = df[col].astype(str).value_counts(dropna=False)
            for value, count in vc.items():
                rows.append(
                    {
                        "split": split,
                        "column": col,
                        col: value,
                        "count": int(count),
                    }
                )
        if rows:
            tables[col] = pd.DataFrame(rows)
    return tables


def save_split_audit_artifacts(
    split_to_csv: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    preferred_columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    frames, summary_df = summarize_split_csvs(split_to_csv)
    overlap_tables = compute_split_overlap_tables(frames)

    candidate_columns: List[str] = []
    preferred_columns = list(preferred_columns or ["generator", "source", "group", "mode", "type"])
    for split_df in frames.values():
        for col in preferred_columns:
            if col in split_df.columns and col not in candidate_columns:
                candidate_columns.append(col)

    distribution_tables = compute_split_distribution_tables(frames, candidate_columns)

    if not summary_df.empty:
        save_dataframe_csv(summary_df, output_dir / "split_summary.csv")

    for name, table in overlap_tables.items():
        if not table.empty:
            save_dataframe_csv(table, output_dir / f"{name}.csv")

    for col, table in distribution_tables.items():
        if not table.empty:
            save_dataframe_csv(table, output_dir / f"distribution_by_{col}.csv")

    report = {
        "available_splits": list(frames.keys()),
        "summary_csv": (output_dir / "split_summary.csv").as_posix() if not summary_df.empty else None,
        "overlap_csvs": {
            name: (output_dir / f"{name}.csv").as_posix()
            for name, table in overlap_tables.items()
            if not table.empty
        },
        "distribution_csvs": {
            col: (output_dir / f"distribution_by_{col}.csv").as_posix()
            for col, table in distribution_tables.items()
            if not table.empty
        },
        "num_exact_filepath_overlaps": {
            f"{row['split_a']}__{row['split_b']}": int(row["overlap_count"])
            for _, row in overlap_tables["filepath_overlap"].iterrows()
        }
        if not overlap_tables["filepath_overlap"].empty
        else {},
        "num_filename_overlaps": {
            f"{row['split_a']}__{row['split_b']}": int(row["overlap_count"])
            for _, row in overlap_tables["filename_overlap"].iterrows()
        }
        if not overlap_tables["filename_overlap"].empty
        else {},
    }
    save_json(report, output_dir / "split_audit_report.json")
    return report


def _read_image_rgb(path: str | Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _fit_to_canvas(image_rgb: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    image_h, image_w = image_rgb.shape[:2]
    scale = min(target_w / max(image_w, 1), target_h / max(image_h, 1))
    new_w = max(1, int(round(image_w * scale)))
    new_h = max(1, int(round(image_h * scale)))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def save_case_montage(
    cases: Sequence[Mapping[str, Any]],
    save_path: str | Path,
    *,
    image_key: str = "panel_path",
    title: Optional[str] = None,
    ncols: int = 2,
    cell_hw: Tuple[int, int] = (360, 720),
) -> Optional[str]:
    image_paths = [str(case.get(image_key)) for case in cases if case.get(image_key)]
    if len(image_paths) == 0:
        return None

    ncols = max(1, int(ncols))
    nrows = int(math.ceil(len(image_paths) / ncols))
    cell_h, cell_w = int(cell_hw[0]), int(cell_hw[1])
    header_h = 50 if title else 0
    gap = 16

    canvas_h = header_h + nrows * cell_h + max(0, nrows - 1) * gap
    canvas_w = ncols * cell_w + max(0, ncols - 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    if title:
        cv2.putText(
            canvas,
            title,
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    for idx, case in enumerate(cases):
        image_path = case.get(image_key)
        if not image_path:
            continue
        image = _read_image_rgb(image_path)
        fitted = _fit_to_canvas(image, (cell_h, cell_w))
        row = idx // ncols
        col = idx % ncols
        y0 = header_h + row * (cell_h + gap)
        x0 = col * (cell_w + gap)
        canvas[y0 : y0 + cell_h, x0 : x0 + cell_w] = fitted

    save_path = ensure_parent(save_path)
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(save_path), canvas_bgr):
        raise RuntimeError(f"Failed to save montage: {save_path}")
    return _to_path(save_path).as_posix()
