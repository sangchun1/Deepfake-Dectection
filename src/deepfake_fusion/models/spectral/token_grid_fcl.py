from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

from .frequency_encoder import fft2_image, ifft2_image


__all__ = ["TokenGridFCL", "build_token_grid_fcl"]


def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """dict / attribute 접근을 모두 지원하는 config getter."""
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


class _SpectrumConv2d(nn.Module):
    """Amplitude / phase spectrum용 가벼운 2D conv block.

    기본은 depthwise 3x3 + pointwise 1x1 조합으로 ViT patch-grid에 맞게
    가볍게 구성되어 있다. depthwise / pointwise 설정을 조정하면 단일 conv처럼도
    사용할 수 있다.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        *,
        depthwise: bool = True,
        pointwise: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )

        layers: list[nn.Module] = []
        if depthwise:
            layers.append(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    groups=channels,
                    bias=bias,
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    groups=1,
                    bias=bias,
                )
            )

        if pointwise:
            layers.append(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=1,
                    padding=0,
                    groups=1,
                    bias=bias,
                )
            )

        self.net = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenGridFCL(nn.Module):
    """ViT patch token grid용 Frequency Convolutional Layer.

    논문의 FCL 아이디어를 ViT patch token에 맞게 옮긴 버전이다.

    흐름:
        [B, N(+prefix), C]
        -> prefix / patch 분리
        -> patch tokens 를 [B, C, H, W] grid로 reshape
        -> FFT2
        -> amplitude / phase 분리
        -> amplitude conv / phase conv
        -> complex spectrum 재구성
        -> iFFT2
        -> patch tokens 복원
        -> residual add (optional)
        -> prefix token 다시 concat

    Notes:
        - CLS 등 prefix token은 변경하지 않는다.
        - 입력이 [B, C, H, W]인 경우 grid tensor에 직접 FCL을 적용할 수도 있다.
        - amplitude는 항상 양수가 되도록 residual update 후 clamp_min 처리한다.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_prefix_tokens: int = 1,
        grid_size: Optional[tuple[int, int]] = None,
        fft_norm: str = "ortho",
        use_fftshift: bool = True,
        use_pre_norm: bool = True,
        residual: bool = True,
        residual_scale_init: float = 0.0,
        eps: float = 1e-6,
        amplitude_kernel_size: int = 3,
        amplitude_depthwise: bool = True,
        amplitude_pointwise: bool = True,
        phase_kernel_size: int = 3,
        phase_depthwise: bool = True,
        phase_pointwise: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {embed_dim}")
        if num_prefix_tokens < 0:
            raise ValueError(
                f"num_prefix_tokens must be >= 0, got {num_prefix_tokens}"
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")

        if grid_size is not None:
            gh, gw = int(grid_size[0]), int(grid_size[1])
            if gh <= 0 or gw <= 0:
                raise ValueError(f"grid_size must be positive, got {grid_size}")
            self.grid_size: Optional[tuple[int, int]] = (gh, gw)
        else:
            self.grid_size = None

        self.embed_dim = int(embed_dim)
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.fft_norm = str(fft_norm)
        self.use_fftshift = bool(use_fftshift)
        self.use_pre_norm = bool(use_pre_norm)
        self.residual = bool(residual)
        self.eps = float(eps)

        self.norm = nn.LayerNorm(self.embed_dim) if self.use_pre_norm else nn.Identity()
        self.amplitude_conv = _SpectrumConv2d(
            channels=self.embed_dim,
            kernel_size=int(amplitude_kernel_size),
            depthwise=bool(amplitude_depthwise),
            pointwise=bool(amplitude_pointwise),
            bias=bool(bias),
        )
        self.phase_conv = _SpectrumConv2d(
            channels=self.embed_dim,
            kernel_size=int(phase_kernel_size),
            depthwise=bool(phase_depthwise),
            pointwise=bool(phase_pointwise),
            bias=bool(bias),
        )

        if self.residual:
            self.gamma = nn.Parameter(torch.tensor(float(residual_scale_init)))
        else:
            self.register_parameter("gamma", None)

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"num_prefix_tokens={self.num_prefix_tokens}, "
            f"grid_size={self.grid_size}, "
            f"fft_norm='{self.fft_norm}', "
            f"use_fftshift={self.use_fftshift}, "
            f"use_pre_norm={self.use_pre_norm}, "
            f"residual={self.residual}"
        )

    @staticmethod
    def _normalize_grid_size(
        grid_size: Optional[tuple[int, int] | list[int]],
    ) -> Optional[tuple[int, int]]:
        if grid_size is None:
            return None
        if len(grid_size) != 2:
            raise ValueError(f"grid_size must have length 2, got {grid_size}")
        gh, gw = int(grid_size[0]), int(grid_size[1])
        if gh <= 0 or gw <= 0:
            raise ValueError(f"grid_size values must be positive, got {grid_size}")
        return gh, gw

    def _resolve_grid_size(
        self,
        num_patches: int,
        grid_size: Optional[tuple[int, int] | list[int]] = None,
    ) -> tuple[int, int]:
        grid = self._normalize_grid_size(grid_size) or self.grid_size
        if grid is not None:
            gh, gw = grid
            if gh * gw != num_patches:
                raise ValueError(
                    "grid_size does not match number of patch tokens: "
                    f"grid_size={grid}, num_patches={num_patches}"
                )
            return gh, gw

        side = int(num_patches**0.5)
        if side * side != num_patches:
            raise ValueError(
                "Could not infer square grid size from patch token count. "
                f"num_patches={num_patches}. Pass grid_size explicitly."
            )
        return side, side

    def _split_prefix(
        self,
        x: torch.Tensor,
        num_prefix_tokens: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected token tensor [B, N, C], got {tuple(x.shape)}")
        n_prefix = self.num_prefix_tokens if num_prefix_tokens is None else int(num_prefix_tokens)
        if n_prefix < 0:
            raise ValueError(f"num_prefix_tokens must be >= 0, got {n_prefix}")
        if n_prefix > x.size(1):
            raise ValueError(
                "num_prefix_tokens cannot exceed sequence length: "
                f"num_prefix_tokens={n_prefix}, seq_len={x.size(1)}"
            )
        prefix = x[:, :n_prefix, :] if n_prefix > 0 else x[:, :0, :]
        patch_tokens = x[:, n_prefix:, :]
        return prefix, patch_tokens

    def _tokens_to_grid(
        self,
        patch_tokens: torch.Tensor,
        *,
        grid_size: Optional[tuple[int, int] | list[int]] = None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if patch_tokens.ndim != 3:
            raise ValueError(
                f"Expected patch token tensor [B, N, C], got {tuple(patch_tokens.shape)}"
            )
        bsz, num_patches, channels = patch_tokens.shape
        if channels != self.embed_dim:
            raise ValueError(
                f"Expected embedding dim {self.embed_dim}, got {channels}"
            )
        gh, gw = self._resolve_grid_size(num_patches=num_patches, grid_size=grid_size)
        grid = patch_tokens.transpose(1, 2).reshape(bsz, channels, gh, gw)
        return grid, (gh, gw)

    @staticmethod
    def _grid_to_tokens(grid: torch.Tensor) -> torch.Tensor:
        if grid.ndim != 4:
            raise ValueError(f"Expected grid tensor [B, C, H, W], got {tuple(grid.shape)}")
        return grid.flatten(2).transpose(1, 2)

    def _forward_grid_impl(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected grid tensor [B, C, H, W], got {tuple(x.shape)}")
        if x.size(1) != self.embed_dim:
            raise ValueError(
                f"Expected channel dim {self.embed_dim}, got {x.size(1)}"
            )

        spectrum = fft2_image(x, norm=self.fft_norm, shift=self.use_fftshift)
        amplitude = spectrum.abs().clamp_min(self.eps)
        phase = torch.angle(spectrum)

        amp_delta = self.amplitude_conv(amplitude)
        phase_delta = self.phase_conv(phase)

        learned_amplitude = (amplitude + amp_delta).clamp_min(self.eps)
        learned_phase = phase + phase_delta
        learned_spectrum = torch.polar(learned_amplitude, learned_phase)

        recon = ifft2_image(
            learned_spectrum,
            norm=self.fft_norm,
            shift=self.use_fftshift,
        )
        return recon

    def forward_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Grid tensor [B, C, H, W]에 직접 FCL 적용."""
        transformed = self._forward_grid_impl(x)
        if not self.residual:
            return transformed
        return x + (self.gamma * transformed)

    def forward(
        self,
        x: torch.Tensor,
        *,
        grid_size: Optional[tuple[int, int] | list[int]] = None,
        num_prefix_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """FCL forward.

        Args:
            x:
                - [B, N, C] ViT token sequence, or
                - [B, C, H, W] patch grid tensor
            grid_size:
                token sequence 입력일 때 patch grid size. None이면 init 시 제공한
                self.grid_size를 사용하고, 그것도 없으면 정사각형 grid로 추론.
            num_prefix_tokens:
                CLS token 수. None이면 self.num_prefix_tokens 사용.
        """
        if x.ndim == 4:
            return self.forward_grid(x)

        if x.ndim != 3:
            raise ValueError(
                "TokenGridFCL expects either [B, N, C] tokens or [B, C, H, W] grid, "
                f"got {tuple(x.shape)}"
            )

        prefix_tokens, patch_tokens = self._split_prefix(
            x,
            num_prefix_tokens=num_prefix_tokens,
        )

        if patch_tokens.numel() == 0:
            return x

        normed_tokens = self.norm(x)
        _, normed_patch_tokens = self._split_prefix(
            normed_tokens,
            num_prefix_tokens=num_prefix_tokens,
        )
        patch_grid, _ = self._tokens_to_grid(normed_patch_tokens, grid_size=grid_size)
        transformed_grid = self._forward_grid_impl(patch_grid)
        transformed_patch_tokens = self._grid_to_tokens(transformed_grid)

        if self.residual:
            updated_patch_tokens = patch_tokens + (self.gamma * transformed_patch_tokens)
        else:
            updated_patch_tokens = transformed_patch_tokens

        if prefix_tokens.numel() == 0:
            return updated_patch_tokens
        return torch.cat([prefix_tokens, updated_patch_tokens], dim=1)


def build_token_grid_fcl(
    cfg: Any,
    *,
    embed_dim: int,
    num_prefix_tokens: int = 1,
    grid_size: Optional[tuple[int, int]] = None,
) -> TokenGridFCL:
    """config 기반 TokenGridFCL 생성 helper.

    기대하는 config 예시:
        cfg.fcl.fft_norm
        cfg.fcl.use_fftshift
        cfg.fcl.use_pre_norm
        cfg.fcl.residual
        cfg.fcl.residual_scale_init
        cfg.fcl.eps
        cfg.fcl.amplitude.kernel_size
        cfg.fcl.amplitude.depthwise
        cfg.fcl.amplitude.pointwise
        cfg.fcl.phase.kernel_size
        cfg.fcl.phase.depthwise
        cfg.fcl.phase.pointwise

    model_cfg 전체 또는 model_cfg.fcl 둘 다 입력 가능하도록 처리.
    """
    fcl_cfg = _cfg_get(cfg, "fcl", default=cfg)
    amp_cfg = _cfg_get(fcl_cfg, "amplitude", default=fcl_cfg)
    phase_cfg = _cfg_get(fcl_cfg, "phase", default=fcl_cfg)

    return TokenGridFCL(
        embed_dim=int(embed_dim),
        num_prefix_tokens=int(
            _cfg_get(fcl_cfg, "num_prefix_tokens", default=num_prefix_tokens)
        ),
        grid_size=grid_size,
        fft_norm=str(_cfg_get(fcl_cfg, "fft_norm", default="ortho")),
        use_fftshift=bool(_cfg_get(fcl_cfg, "use_fftshift", default=True)),
        use_pre_norm=bool(_cfg_get(fcl_cfg, "use_pre_norm", default=True)),
        residual=bool(_cfg_get(fcl_cfg, "residual", default=True)),
        residual_scale_init=float(
            _cfg_get(fcl_cfg, "residual_scale_init", default=0.0)
        ),
        eps=float(_cfg_get(fcl_cfg, "eps", default=1e-6)),
        amplitude_kernel_size=int(_cfg_get(amp_cfg, "kernel_size", default=3)),
        amplitude_depthwise=bool(_cfg_get(amp_cfg, "depthwise", default=True)),
        amplitude_pointwise=bool(_cfg_get(amp_cfg, "pointwise", default=True)),
        phase_kernel_size=int(_cfg_get(phase_cfg, "kernel_size", default=3)),
        phase_depthwise=bool(_cfg_get(phase_cfg, "depthwise", default=True)),
        phase_pointwise=bool(_cfg_get(phase_cfg, "pointwise", default=True)),
        bias=bool(_cfg_get(fcl_cfg, "bias", default=True)),
    )
