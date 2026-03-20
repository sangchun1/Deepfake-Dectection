from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _to_numpy(x: Any) -> np.ndarray:
    """
    torch.Tensor / list / np.ndarray 를 numpy 배열로 변환.
    """
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_float(value: Any) -> float:
    """
    torch scalar / numpy scalar / python number 를 float로 변환.
    """
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def infer_num_classes_from_logits(logits: torch.Tensor) -> int:
    """
    logits shape으로부터 클래스 수 추론.

    지원:
    - [B] 또는 [B, 1] -> binary
    - [B, C] -> C-class
    """
    if logits.ndim == 1:
        return 2
    if logits.ndim == 2 and logits.size(1) == 1:
        return 2
    if logits.ndim == 2:
        return int(logits.size(1))

    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


def logits_to_probs(logits: torch.Tensor) -> np.ndarray:
    """
    logits를 probability로 변환.

    반환:
    - binary: shape [B], positive class probability
    - multiclass: shape [B, C], class probabilities
    """
    if logits.ndim == 1:
        probs = torch.sigmoid(logits)
        return probs.detach().cpu().numpy()

    if logits.ndim == 2 and logits.size(1) == 1:
        probs = torch.sigmoid(logits.squeeze(1))
        return probs.detach().cpu().numpy()

    if logits.ndim == 2:
        probs = torch.softmax(logits, dim=1)
        if logits.size(1) == 2:
            return probs[:, 1].detach().cpu().numpy()
        return probs.detach().cpu().numpy()

    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


def probs_to_preds(probs: Any, threshold: float = 0.5) -> np.ndarray:
    """
    probability를 class prediction으로 변환.

    입력:
    - binary: [B]
    - multiclass: [B, C]
    """
    probs = _to_numpy(probs)

    if probs.ndim == 1:
        return (probs >= threshold).astype(np.int64)

    if probs.ndim == 2:
        return np.argmax(probs, axis=1).astype(np.int64)

    raise ValueError(f"Unsupported probability shape: {probs.shape}")


def extract_positive_probs(logits: torch.Tensor) -> np.ndarray:
    """
    binary classification용 positive probability 추출.
    """
    probs = logits_to_probs(logits)
    probs = _to_numpy(probs)

    if probs.ndim != 1:
        raise ValueError("Positive probability extraction is only valid for binary classification.")

    return probs


def compute_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_prob: Optional[Any] = None,
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """
    분류 지표 계산.

    binary:
    - accuracy, precision, recall, f1, auc

    multiclass:
    - accuracy, precision, recall, f1, auc(가능한 경우 macro-ovr)
    """
    y_true = _to_numpy(y_true).astype(np.int64)
    y_pred = _to_numpy(y_pred).astype(np.int64)
    y_prob_np = None if y_prob is None else _to_numpy(y_prob)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")

    if len(y_true) == 0:
        raise ValueError("Empty targets are not allowed.")

    if num_classes is None:
        inferred_max = max(int(y_true.max()), int(y_pred.max()))
        num_classes = inferred_max + 1

        if y_prob_np is not None and y_prob_np.ndim == 2:
            num_classes = max(num_classes, int(y_prob_np.shape[1]))
        elif y_prob_np is not None and y_prob_np.ndim == 1:
            num_classes = max(num_classes, 2)

    is_binary = num_classes == 2

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    if is_binary:
        metrics["precision"] = float(
            precision_score(y_true, y_pred, average="binary", zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(y_true, y_pred, average="binary", zero_division=0)
        )
        metrics["f1"] = float(
            f1_score(y_true, y_pred, average="binary", zero_division=0)
        )

        auc = np.nan
        if y_prob_np is not None:
            try:
                if len(np.unique(y_true)) >= 2:
                    auc = float(roc_auc_score(y_true, y_prob_np))
            except ValueError:
                pass

        metrics["auc"] = float(auc) if not np.isnan(auc) else np.nan

    else:
        metrics["precision"] = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["f1"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )

        auc = np.nan
        if y_prob_np is not None and y_prob_np.ndim == 2:
            try:
                if len(np.unique(y_true)) >= 2:
                    auc = float(
                        roc_auc_score(
                            y_true,
                            y_prob_np,
                            multi_class="ovr",
                            average="macro",
                        )
                    )
            except ValueError:
                pass

        metrics["auc"] = float(auc) if not np.isnan(auc) else np.nan

    return metrics


def compute_confusion_details(
    y_true: Any,
    y_pred: Any,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    confusion matrix 및 기본 개수 계산.
    binary면 tn/fp/fn/tp도 같이 반환.
    """
    y_true = _to_numpy(y_true).astype(np.int64)
    y_pred = _to_numpy(y_pred).astype(np.int64)

    if num_classes is None:
        num_classes = max(int(y_true.max()), int(y_pred.max())) + 1

    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    result: Dict[str, Any] = {
        "confusion_matrix": cm,
    }

    if num_classes == 2 and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        result.update(
            {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    return result


def compute_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    loss: Optional[Any] = None,
) -> Dict[str, float]:
    """
    한 배치 기준으로 logits -> metrics 계산.

    지원 logits:
    - [B]
    - [B, 1]
    - [B, 2]
    - [B, C]
    """
    if not torch.is_tensor(logits):
        raise TypeError("logits must be a torch.Tensor")
    if not torch.is_tensor(targets):
        raise TypeError("targets must be a torch.Tensor")

    targets_np = targets.detach().cpu().numpy().astype(np.int64)
    num_classes = infer_num_classes_from_logits(logits)

    probs = logits_to_probs(logits)
    preds = probs_to_preds(probs, threshold=threshold)

    metrics = compute_classification_metrics(
        y_true=targets_np,
        y_pred=preds,
        y_prob=probs,
        num_classes=num_classes,
    )

    if loss is not None:
        metrics["loss"] = _safe_float(loss)

    return metrics


@dataclass
class AverageMeter:
    """
    scalar 평균 추적기.
    """
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: Any, n: int = 1) -> None:
        value = _safe_float(value)
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class ClassificationMeter:
    """
    epoch 단위 분류 metric 누적기.

    예:
        meter = ClassificationMeter()
        meter.update(logits=logits, targets=labels, loss=loss)
        ...
        metrics = meter.compute()
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.targets: List[np.ndarray] = []
        self.preds: List[np.ndarray] = []
        self.probs: List[np.ndarray] = []
        self.loss_meter = AverageMeter()

    def update(
        self,
        targets: Any,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[Any] = None,
        preds: Optional[Any] = None,
        loss: Optional[Any] = None,
        threshold: float = 0.5,
    ) -> None:
        """
        배치 결과 누적.

        입력 방식:
        1) logits + targets
        2) probs + targets
        3) preds + targets
        """
        targets_np = _to_numpy(targets).astype(np.int64).reshape(-1)

        if logits is not None:
            probs_np = _to_numpy(logits_to_probs(logits))
            preds_np = probs_to_preds(probs_np, threshold=threshold)
        else:
            probs_np = None if probs is None else _to_numpy(probs)
            preds_np = None if preds is None else _to_numpy(preds).astype(np.int64)

            if preds_np is None and probs_np is not None:
                preds_np = probs_to_preds(probs_np, threshold=threshold)

        if preds_np is None:
            raise ValueError("At least one of logits / probs / preds must be provided.")

        self.targets.append(targets_np)
        self.preds.append(preds_np.reshape(-1))

        if probs_np is not None:
            self.probs.append(probs_np)

        if loss is not None:
            self.loss_meter.update(loss, n=len(targets_np))

    def compute(self, num_classes: Optional[int] = None) -> Dict[str, float]:
        if not self.targets:
            raise ValueError("No samples accumulated in ClassificationMeter.")

        y_true = np.concatenate(self.targets, axis=0)
        y_pred = np.concatenate(self.preds, axis=0)

        y_prob = None
        if self.probs:
            first_prob = self.probs[0]
            if first_prob.ndim == 1:
                y_prob = np.concatenate([p.reshape(-1) for p in self.probs], axis=0)
            else:
                y_prob = np.concatenate(self.probs, axis=0)

        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            num_classes=num_classes,
        )

        if self.loss_meter.count > 0:
            metrics["loss"] = float(self.loss_meter.avg)

        return metrics

    def compute_with_details(self, num_classes: Optional[int] = None) -> Dict[str, Any]:
        if not self.targets:
            raise ValueError("No samples accumulated in ClassificationMeter.")

        y_true = np.concatenate(self.targets, axis=0)
        y_pred = np.concatenate(self.preds, axis=0)

        y_prob = None
        if self.probs:
            first_prob = self.probs[0]
            if first_prob.ndim == 1:
                y_prob = np.concatenate([p.reshape(-1) for p in self.probs], axis=0)
            else:
                y_prob = np.concatenate(self.probs, axis=0)

        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            num_classes=num_classes,
        )
        details = compute_confusion_details(
            y_true=y_true,
            y_pred=y_pred,
            num_classes=num_classes,
        )

        if self.loss_meter.count > 0:
            metrics["loss"] = float(self.loss_meter.avg)

        return {
            "metrics": metrics,
            "details": details,
        }