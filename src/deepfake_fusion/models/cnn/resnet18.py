from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models


def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """
    dict / attribute 접근을 모두 지원하는 config getter.
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


def _load_torchvision_resnet18(pretrained: bool = True) -> nn.Module:
    """
    torchvision 버전 차이를 고려해서 resnet18 로드.
    """
    try:
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet18(weights=weights)
    except AttributeError:
        model = tv_models.resnet18(pretrained=pretrained)

    return model


def _adapt_first_conv_weight(
    original_weight: torch.Tensor,
    in_channels: int,
) -> torch.Tensor:
    """
    pretrained conv1 weight를 입력 채널 수에 맞게 변환.

    - 1채널: RGB 평균
    - 3채널: 그대로 사용
    - 3채널 초과: 반복 후 자르기
    - 2채널 등 기타: 평균 채널을 반복해서 맞춤
    """
    out_channels, old_in_channels, kh, kw = original_weight.shape

    if in_channels == old_in_channels:
        return original_weight

    if in_channels == 1:
        return original_weight.mean(dim=1, keepdim=True)

    if in_channels > old_in_channels:
        repeat = (in_channels + old_in_channels - 1) // old_in_channels
        expanded = original_weight.repeat(1, repeat, 1, 1)
        return expanded[:, :in_channels, :, :] / repeat

    # in_channels가 2 같은 경우
    mean_weight = original_weight.mean(dim=1, keepdim=True)
    return mean_weight.repeat(1, in_channels, 1, 1)


class ResNet18Classifier(nn.Module):
    """
    ResNet-18 기반 이미지 분류기.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        in_channels: int = 3,
        dropout: float = 0.0,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got: {num_classes}")
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got: {in_channels}")
        if dropout < 0.0:
            raise ValueError(f"dropout must be >= 0, got: {dropout}")

        backbone = _load_torchvision_resnet18(pretrained=pretrained)

        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

            with torch.no_grad():
                adapted_weight = _adapt_first_conv_weight(
                    original_weight=old_conv.weight.data,
                    in_channels=in_channels,
                )
                new_conv.weight.copy_(adapted_weight)

                if old_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias.data)

            backbone.conv1 = new_conv

        self.feature_dim = backbone.fc.in_features
        self.backbone = backbone

        if dropout > 0.0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(self.feature_dim, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(self.feature_dim, num_classes)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """
        classifier head(fc)를 제외한 backbone 파라미터를 freeze.
        """
        for name, param in self.backbone.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """
        전체 파라미터 학습 가능하도록 설정.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        fc 직전 global pooled feature 추출.
        shape: [B, feature_dim]
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_resnet18(model_cfg: Any) -> ResNet18Classifier:
    """
    config 기반 ResNet18 생성 함수.

    기대하는 config 예시:
        cfg.model.backbone.pretrained
        cfg.model.backbone.in_channels
        cfg.model.backbone.freeze
        cfg.model.head.num_classes
        cfg.model.head.dropout
    """
    pretrained = bool(_cfg_get(model_cfg, "backbone", "pretrained", default=True))
    in_channels = int(_cfg_get(model_cfg, "backbone", "in_channels", default=3))
    freeze_backbone = bool(_cfg_get(model_cfg, "backbone", "freeze", default=False))
    num_classes = int(_cfg_get(model_cfg, "head", "num_classes", default=2))
    dropout = float(_cfg_get(model_cfg, "head", "dropout", default=0.0))

    model = ResNet18Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
    return model