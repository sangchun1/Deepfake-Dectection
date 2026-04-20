from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling for classification logits.

    Supports the logit shapes used in this repository:
    - [B] or [B, 1] for binary classification
    - [B, 2] for binary classification with two-class logits
    - [B, C] for multi-class classification

    The scaler learns a single positive scalar ``T`` on a validation split and
    then uses ``logits / T`` for calibrated probability estimation.
    """

    def __init__(self, temperature: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.eps = float(eps)
        self.log_temperature = nn.Parameter(torch.tensor(float(math.log(temperature))))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp_min(self.eps)

    def extra_repr(self) -> str:
        return f"temperature={self.temperature.detach().item():.6f}"

    def set_temperature(self, value: float) -> "TemperatureScaler":
        if value <= 0:
            raise ValueError("temperature must be > 0")
        with torch.no_grad():
            self.log_temperature.copy_(torch.tensor(math.log(float(value)), device=self.log_temperature.device))
        return self

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.scale_logits(logits)

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(logits):
            raise TypeError("logits must be a torch.Tensor")
        return logits / self.temperature.to(device=logits.device, dtype=logits.dtype)

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        scaled_logits = self.scale_logits(logits)
        if scaled_logits.ndim == 1:
            return torch.sigmoid(scaled_logits)
        if scaled_logits.ndim == 2 and scaled_logits.size(1) == 1:
            return torch.sigmoid(scaled_logits.squeeze(1))
        if scaled_logits.ndim == 2:
            probs = torch.softmax(scaled_logits, dim=1)
            if scaled_logits.size(1) == 2:
                return probs[:, 1]
            return probs
        raise ValueError(f"Unsupported logits shape: {tuple(scaled_logits.shape)}")

    def nll(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        scaled_logits = self.scale_logits(logits)
        return _classification_nll_from_logits(scaled_logits, targets)

    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        max_iter: int = 50,
        lr: float = 0.01,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Fit a scalar temperature on validation logits and labels.

        Returns a small summary dictionary with before/after NLL values and the
        learned temperature.
        """
        if not torch.is_tensor(logits):
            logits = torch.as_tensor(logits)
        if not torch.is_tensor(targets):
            targets = torch.as_tensor(targets)

        if logits.ndim not in {1, 2}:
            raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")

        device = self.log_temperature.device
        logits = logits.detach().to(device=device, dtype=torch.float32)
        targets = targets.detach().to(device=device)

        before_nll = float(_classification_nll_from_logits(logits, targets).detach().cpu().item())

        optimizer = torch.optim.LBFGS(
            [self.log_temperature],
            lr=float(lr),
            max_iter=int(max_iter),
            tolerance_grad=float(tolerance_grad),
            tolerance_change=float(tolerance_change),
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = self.nll(logits, targets)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            after_nll = float(self.nll(logits, targets).detach().cpu().item())
            learned_temperature = float(self.temperature.detach().cpu().item())

        if verbose:
            print(
                "[TemperatureScaler] "
                f"temperature={learned_temperature:.6f} | "
                f"nll_before={before_nll:.6f} | nll_after={after_nll:.6f}"
            )

        return {
            "temperature": learned_temperature,
            "nll_before": before_nll,
            "nll_after": after_nll,
        }

    def to_dict(self) -> Dict[str, float]:
        return {
            "temperature": float(self.temperature.detach().cpu().item()),
            "eps": float(self.eps),
        }

    def save_json(self, path: str | Path, metadata: Optional[Dict[str, Any]] = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = self.to_dict()
        if metadata:
            payload["metadata"] = metadata
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemperatureScaler":
        return cls(
            temperature=float(data.get("temperature", 1.0)),
            eps=float(data.get("eps", 1e-6)),
        )

    @classmethod
    def load_json(cls, path: str | Path, map_location: str | torch.device | None = None) -> "TemperatureScaler":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        scaler = cls.from_dict(data)
        if map_location is not None:
            scaler = scaler.to(map_location)
        return scaler


def load_temperature_value(path: str | Path) -> float:
    """Convenience helper for places that only need the scalar T."""
    scaler = TemperatureScaler.load_json(path)
    return float(scaler.temperature.detach().cpu().item())


def _classification_nll_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return F.binary_cross_entropy_with_logits(logits, targets.float())

    if logits.ndim == 2 and logits.size(1) == 1:
        return F.binary_cross_entropy_with_logits(logits.squeeze(1), targets.float())

    if logits.ndim == 2:
        return F.cross_entropy(logits, targets.long())

    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")
