from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    MultiStepLR,
    ReduceLROnPlateau,
    StepLR,
)

from ..metrics.classification import ClassificationMeter


try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


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


def _to_plain_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if hasattr(obj, "items"):
        try:
            return {k: _to_plain_dict(v) for k, v in obj.items()}
        except Exception:
            pass
    if isinstance(obj, list):
        return [_to_plain_dict(v) for v in obj]
    return obj


def resolve_device(device_name: Optional[str] = None) -> torch.device:
    """
    device 문자열을 실제 torch.device로 변환.
    """
    device_name = (device_name or "cuda").lower()

    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    if device_name == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_criterion(train_cfg: Any) -> nn.Module:
    """
    loss 함수 생성.
    현재 기본은 cross_entropy.
    """
    loss_name = str(_cfg_get(train_cfg, "loss", "name", default="cross_entropy")).lower()

    if loss_name == "cross_entropy":
        label_smoothing = float(_cfg_get(train_cfg, "loss", "label_smoothing", default=0.0))
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if loss_name in {"bce", "bce_with_logits", "binary_cross_entropy_with_logits"}:
        return nn.BCEWithLogitsLoss()

    raise ValueError(f"Unsupported loss: {loss_name}")


def build_optimizer(model: nn.Module, train_cfg: Any) -> torch.optim.Optimizer:
    """
    optimizer 생성.
    """
    optimizer_name = str(_cfg_get(train_cfg, "optimizer", "name", default="adamw")).lower()
    lr = float(_cfg_get(train_cfg, "optimizer", "lr", default=3e-4))
    weight_decay = float(_cfg_get(train_cfg, "optimizer", "weight_decay", default=1e-4))
    betas = _cfg_get(train_cfg, "optimizer", "betas", default=[0.9, 0.999])
    momentum = float(_cfg_get(train_cfg, "optimizer", "momentum", default=0.9))

    params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay, betas=tuple(betas))

    if optimizer_name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay, betas=tuple(betas))

    if optimizer_name == "sgd":
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: Any,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    scheduler 생성.
    """
    scheduler_name = str(_cfg_get(train_cfg, "scheduler", "name", default="none")).lower()

    if scheduler_name in {"none", "null", ""}:
        return None

    if scheduler_name == "cosine":
        t_max = int(_cfg_get(train_cfg, "scheduler", "t_max", default=10))
        eta_min = float(_cfg_get(train_cfg, "scheduler", "eta_min", default=1e-6))
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    if scheduler_name == "step":
        step_size = int(_cfg_get(train_cfg, "scheduler", "step_size", default=5))
        gamma = float(_cfg_get(train_cfg, "scheduler", "gamma", default=0.1))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    if scheduler_name == "multistep":
        milestones = list(_cfg_get(train_cfg, "scheduler", "milestones", default=[10, 20]))
        gamma = float(_cfg_get(train_cfg, "scheduler", "gamma", default=0.1))
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    if scheduler_name == "plateau":
        mode = str(_cfg_get(train_cfg, "scheduler", "mode", default="max")).lower()
        factor = float(_cfg_get(train_cfg, "scheduler", "factor", default=0.5))
        patience = int(_cfg_get(train_cfg, "scheduler", "patience", default=2))
        min_lr = float(_cfg_get(train_cfg, "scheduler", "min_lr", default=1e-6))
        return ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_cfg: Any,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[str] = None,
        wandb_run: Optional[Any] = None,
    ) -> None:
        self.cfg = train_cfg
        self.device = resolve_device(device or _cfg_get(train_cfg, "experiment", "device", default="cuda"))
        self.model = model.to(self.device)

        self.criterion = criterion if criterion is not None else build_criterion(train_cfg)
        self.criterion = self.criterion.to(self.device)

        self.optimizer = optimizer if optimizer is not None else build_optimizer(model, train_cfg)
        self.scheduler = scheduler if scheduler is not None else build_scheduler(self.optimizer, train_cfg)

        self.epochs = int(_cfg_get(train_cfg, "train", "epochs", default=1))
        self.grad_accum_steps = int(_cfg_get(train_cfg, "train", "grad_accum_steps", default=1))
        self.clip_grad_norm = _cfg_get(train_cfg, "train", "clip_grad_norm", default=None)

        self.use_amp = bool(_cfg_get(train_cfg, "experiment", "use_amp", default=True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.threshold = float(_cfg_get(train_cfg, "evaluation", "threshold", default=0.5))
        self.log_interval = int(_cfg_get(train_cfg, "logging", "log_interval", default=50))
        self.val_interval = int(_cfg_get(train_cfg, "logging", "val_interval", default=1))
        # self.save_interval = int(_cfg_get(train_cfg, "logging", "save_interval", default=1))
        self.use_tqdm = bool(_cfg_get(train_cfg, "logging", "use_tqdm", default=True))

        self.output_dir = Path(_cfg_get(train_cfg, "experiment", "output_dir", default="outputs/default"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = str(_cfg_get(train_cfg, "checkpoint", "monitor", default="val_auc"))
        self.monitor_mode = str(_cfg_get(train_cfg, "checkpoint", "mode", default="max")).lower()

        self.best_score = -math.inf if self.monitor_mode == "max" else math.inf
        self.best_epoch = 0
        self.no_improve_count = 0
        self.early_stopping_enabled = bool(
            _cfg_get(train_cfg, "checkpoint", "early_stopping", "enabled", default=False)
        )
        self.early_stopping_patience = int(
            _cfg_get(train_cfg, "checkpoint", "early_stopping", "patience", default=5)
        )
        self.early_stopping_min_delta = float(
            _cfg_get(train_cfg, "checkpoint", "early_stopping", "min_delta", default=0.0)
        )
        self.history = []

        self.wandb_run = wandb_run

        # ------------------------------------------------------------------
        # Auxiliary branch loss 설정
        # ------------------------------------------------------------------
        self.aux_enabled = bool(
            _cfg_get(train_cfg, "loss", "auxiliary", "enabled", default=False)
        )
        self.aux_spatial_weight = float(
            _cfg_get(train_cfg, "loss", "auxiliary", "spatial_weight", default=0.0)
        )
        self.aux_spectral_weight = float(
            _cfg_get(train_cfg, "loss", "auxiliary", "spectral_weight", default=0.0)
        )

    def _autocast_context(self):
        return torch.amp.autocast("cuda",enabled=self.use_amp)

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        moved = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(self.device, non_blocking=True)
            else:
                moved[key] = value
        return moved

    def _compute_loss(
        self,
        model_output: Any,
        labels: torch.Tensor,
        logits_override: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        fused loss + auxiliary branch loss 계산.

        반환:
            total_loss, loss_dict
        """
        logits = logits_override if logits_override is not None else self._extract_logits(model_output)
        fused_loss = self.criterion(logits, labels)

        loss_dict: Dict[str, torch.Tensor] = {
            "loss_total": fused_loss,
            "loss_fused": fused_loss.detach(),
        }

        if not self.aux_enabled or not isinstance(model_output, Mapping):
            return fused_loss, loss_dict

        total_loss = fused_loss

        spatial_aux_logits = model_output.get("spatial_aux_logits", None)
        if torch.is_tensor(spatial_aux_logits):
            spatial_aux_loss = self.criterion(spatial_aux_logits, labels)
            total_loss = total_loss + self.aux_spatial_weight * spatial_aux_loss
            loss_dict["loss_spatial_aux"] = spatial_aux_loss.detach()
        else:
            spatial_aux_loss = None

        spectral_aux_logits = model_output.get("spectral_aux_logits", None)
        if torch.is_tensor(spectral_aux_logits):
            spectral_aux_loss = self.criterion(spectral_aux_logits, labels)
            total_loss = total_loss + self.aux_spectral_weight * spectral_aux_loss
            loss_dict["loss_spectral_aux"] = spectral_aux_loss.detach()
        else:
            spectral_aux_loss = None

        loss_dict["loss_total"] = total_loss
        return total_loss, loss_dict

    def _extract_logits(self, model_output: Any) -> torch.Tensor:
        """
        모델 출력에서 실제 분류 logits를 꺼낸다.
        - 일반 모델: Tensor logits 반환
        - dict 반환 모델: 아래 키 중 하나를 우선 사용
          logits / fused_logits / output / pred
        """
        if torch.is_tensor(model_output):
            return model_output

        if isinstance(model_output, Mapping):
            for key in ("logits", "fused_logits", "output", "pred"):
                value = model_output.get(key, None)
                if torch.is_tensor(value):
                    return value

        raise TypeError(
            "Model output must be a tensor or a mapping containing one of: "
            "'logits', 'fused_logits', 'output', 'pred'."
        )

    def _extract_monitor_value(self, metrics: Dict[str, float], name: str) -> float:
        """
        monitor 이름 예:
        - val_auc
        - val_loss
        - accuracy
        - train_loss
        """
        if name in metrics:
            return float(metrics[name])

        if name.startswith("val_"):
            key = name.replace("val_", "", 1)
            if key in metrics:
                return float(metrics[key])

        if name.startswith("train_"):
            key = name.replace("train_", "", 1)
            if key in metrics:
                return float(metrics[key])

        raise KeyError(f"Monitor metric '{name}' not found in metrics: {list(metrics.keys())}")

    def _is_better(self, score: float, best_score: float, mode: str) -> bool:
        min_delta = self.early_stopping_min_delta
        if mode == "max":
            return score > (best_score + min_delta)
        if mode == "min":
            return score < (best_score - min_delta)
        raise ValueError(f"Unsupported monitor mode: {mode}")

    def _get_current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Dict[str, float],
    ) -> Path:
        path = self.output_dir / filename

        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "metrics": metrics,
            "config": _to_plain_dict(self.cfg),
        }
        torch.save(state, path)
        return path

    def load_checkpoint(self, checkpoint_path: str | Path, strict: bool = True) -> Dict[str, Any]:
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            sched_state = checkpoint["scheduler_state_dict"]
            if sched_state is not None:
                self.scheduler.load_state_dict(sched_state)

        if self.use_amp and "scaler_state_dict" in checkpoint:
            scaler_state = checkpoint["scaler_state_dict"]
            if scaler_state is not None:
                self.scaler.load_state_dict(scaler_state)

        self.best_score = checkpoint.get("best_score", self.best_score)
        self.best_epoch = checkpoint.get("best_epoch", self.best_epoch)

        return checkpoint

    def train_one_epoch(self, loader, epoch: int) -> Dict[str, float]:
        self.model.train()
        meter = ClassificationMeter()

        self.optimizer.zero_grad(set_to_none=True)
        aux_loss_total = 0.0
        spatial_aux_loss_total = 0.0
        spectral_aux_loss_total = 0.0
        num_steps = 0

        iterator = loader
        if self.use_tqdm and tqdm is not None:
            iterator = tqdm(loader, desc=f"Train {epoch}/{self.epochs}", leave=False)

        for step, batch in enumerate(iterator, start=1):
            batch = self._move_batch_to_device(batch)
            images = batch["image"]
            labels = batch["label"]

            with self._autocast_context():
                model_output = self.model(images)
                logits = self._extract_logits(model_output)
                loss, loss_dict = self._compute_loss(model_output, labels)
            loss_to_backward = loss / self.grad_accum_steps

            if self.use_amp:
                self.scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            should_step = (step % self.grad_accum_steps == 0) or (step == len(loader))
            if should_step:
                if self.clip_grad_norm is not None:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), max_norm=float(self.clip_grad_norm))

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

            meter.update(
                logits=logits.detach(),
                targets=labels.detach(),
                loss=loss.detach(),
                threshold=self.threshold,
            )
            num_steps += 1
            aux_loss_total += float(loss_dict["loss_total"].detach().item())
            if "loss_spatial_aux" in loss_dict:
                spatial_aux_loss_total += float(loss_dict["loss_spatial_aux"].item())
            if "loss_spectral_aux" in loss_dict:
                spectral_aux_loss_total += float(loss_dict["loss_spectral_aux"].item())

            if self.use_tqdm and tqdm is not None:
                iterator.set_postfix(loss=f"{meter.loss_meter.avg:.4f}", lr=f"{self._get_current_lr():.2e}")
            elif step % self.log_interval == 0:
                print(
                    f"[Train] epoch={epoch} step={step}/{len(loader)} "
                    f"loss={meter.loss_meter.avg:.4f} lr={self._get_current_lr():.2e}"
                )

        metrics = meter.compute()
        metrics["lr"] = self._get_current_lr()
        if num_steps > 0:
            metrics["loss_total"] = aux_loss_total / num_steps
            if self.aux_enabled:
                metrics["loss_spatial_aux"] = spatial_aux_loss_total / num_steps
                metrics["loss_spectral_aux"] = spectral_aux_loss_total / num_steps
        return metrics

    @torch.no_grad()
    def evaluate(self, loader, split: str = "val", temperature: Optional[float] = None, threshold: Optional[float] = None,) -> Dict[str, float]:
        self.model.eval()
        meter = ClassificationMeter()
        aux_loss_total = 0.0
        spatial_aux_loss_total = 0.0
        spectral_aux_loss_total = 0.0
        num_steps = 0
        threshold_value = self.threshold if threshold is None else float(threshold)
        temperature_value = None if temperature is None else float(temperature)

        if temperature_value is not None and temperature_value <= 0:
            raise ValueError("temperature must be > 0.")

        iterator = loader
        if self.use_tqdm and tqdm is not None:
            iterator = tqdm(loader, desc=f"{split.capitalize()} ", leave=False)

        for batch in iterator:
            batch = self._move_batch_to_device(batch)
            images = batch["image"]
            labels = batch["label"]

            with self._autocast_context():
                model_output = self.model(images)
                raw_logits = self._extract_logits(model_output)
                logits = raw_logits

                if temperature_value is not None and temperature_value != 1.0:
                   logits = logits / temperature_value
                
                loss, loss_dict = self._compute_loss(
                    model_output,
                    labels,
                    logits_override=logits,
                )

            meter.update(
                logits=logits.detach(),
                targets=labels.detach(),
                loss=loss.detach(),
                threshold=threshold_value,
            )
            num_steps += 1
            aux_loss_total += float(loss_dict["loss_total"].detach().item())
            if "loss_spatial_aux" in loss_dict:
                spatial_aux_loss_total += float(loss_dict["loss_spatial_aux"].item())
            if "loss_spectral_aux" in loss_dict:
                spectral_aux_loss_total += float(loss_dict["loss_spectral_aux"].item())

            if self.use_tqdm and tqdm is not None:
                iterator.set_postfix(loss=f"{meter.loss_meter.avg:.4f}")

        metrics = meter.compute()
        if temperature_value is not None:
            metrics["temperature"] = temperature_value
        metrics["threshold"] = threshold_value
        if num_steps > 0:
            metrics["loss_total"] = aux_loss_total / num_steps
            if self.aux_enabled:
                metrics["loss_spatial_aux"] = spatial_aux_loss_total / num_steps
                metrics["loss_spectral_aux"] = spectral_aux_loss_total / num_steps
        return metrics

    def _step_scheduler(self, metrics_for_scheduler: Optional[Dict[str, float]] = None) -> None:
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metrics_for_scheduler is None:
                raise ValueError("ReduceLROnPlateau requires metrics_for_scheduler.")
            monitor_name = self.monitor
            score = self._extract_monitor_value(metrics_for_scheduler, monitor_name)
            self.scheduler.step(score)
        else:
            self.scheduler.step()

    def fit(self, train_loader, val_loader=None) -> list[Dict[str, float]]:
        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_one_epoch(train_loader, epoch)
            epoch_log: Dict[str, float] = {f"train_{k}": v for k, v in train_metrics.items()}

            val_metrics = None
            if val_loader is not None and epoch % self.val_interval == 0:
                val_metrics = self.evaluate(val_loader, split="val")
                epoch_log.update({f"val_{k}": v for k, v in val_metrics.items()})

            monitor_source = val_metrics if val_metrics is not None else train_metrics
            monitor_value = self._extract_monitor_value(monitor_source, self.monitor)
            improved = self._is_better(monitor_value, self.best_score, self.monitor_mode)

            if improved:
                self.best_score = monitor_value
                self.best_epoch = epoch
                self.no_improve_count = 0
                self._save_checkpoint("best.pth", epoch, epoch_log)

                if self.wandb_run is not None:
                    self.wandb_run.summary["best_epoch"] = epoch
                    self.wandb_run.summary[f"best_{self.monitor}"] = monitor_value
            else:
                self.no_improve_count += 1

            self._step_scheduler(metrics_for_scheduler=monitor_source)

            epoch_log["epoch"] = epoch
            epoch_log["best_score"] = self.best_score
            epoch_log["best_epoch"] = self.best_epoch
            self.history.append(epoch_log)

            if self.wandb_run is not None:
                self.wandb_run.log(epoch_log, step=epoch)

            summary_parts = [f"epoch={epoch}/{self.epochs}"]
            if "train_loss" in epoch_log:
                summary_parts.append(f"train_loss={epoch_log['train_loss']:.4f}")
            if "train_auc" in epoch_log:
                summary_parts.append(f"train_auc={epoch_log['train_auc']:.4f}")
            if "val_loss" in epoch_log:
                summary_parts.append(f"val_loss={epoch_log['val_loss']:.4f}")
            if "val_auc" in epoch_log:
                summary_parts.append(f"val_auc={epoch_log['val_auc']:.4f}")

            summary_parts.append(f"best_{self.monitor}={self.best_score:.4f}")
            summary_parts.append(f"best_epoch={self.best_epoch}")
            summary_parts.append(f"no_improve={self.no_improve_count}")
            summary_parts.append(f"lr={self._get_current_lr():.2e}")
            print(" | ".join(summary_parts))

            if (
                self.early_stopping_enabled
                and self.no_improve_count >= self.early_stopping_patience
            ):
                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"No improvement in {self.monitor} for "
                    f"{self.early_stopping_patience} consecutive validation checks."
                )
                break

        if self.wandb_run is not None:
            self.wandb_run.summary["final_best_epoch"] = self.best_epoch
            self.wandb_run.summary["final_best_score"] = self.best_score

        print(
            f"Training finished. Best checkpoint: {self.output_dir / 'best.pth'} "
            f"(epoch={self.best_epoch}, {self.monitor}={self.best_score:.6f})"
        )
        return self.history