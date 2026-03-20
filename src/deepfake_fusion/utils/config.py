from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union

import yaml


PathLike = Union[str, Path]


class Config(dict):
    """
    dict 기반 설정 객체.
    cfg["train"]["epochs"] 와 cfg.train.epochs 둘 다 사용할 수 있게 해준다.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = _to_config(value)

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain_dict(self)


def _to_config(obj: Any) -> Any:
    """중첩 dict/list를 Config 객체로 재귀 변환."""
    if isinstance(obj, Config):
        return obj
    if isinstance(obj, dict):
        return Config({k: _to_config(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config(v) for v in obj]
    return obj


def _to_plain_dict(obj: Any) -> Any:
    """Config 객체를 일반 dict/list로 재귀 변환."""
    if isinstance(obj, Config):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_dict(v) for v in obj]
    return obj


def get_project_root() -> Path:
    """
    repo root 반환.
    현재 파일 위치:
    src/deepfake_fusion/utils/config.py
    """
    return Path(__file__).resolve().parents[3]


def resolve_path(path: PathLike, root: Optional[PathLike] = None) -> Path:
    """
    상대 경로면 root 기준으로 절대 경로로 변환.
    root가 없으면 repo root 기준으로 변환.
    """
    path = Path(path)
    if path.is_absolute():
        return path.resolve()

    base = Path(root).resolve() if root is not None else get_project_root()
    return (base / path).resolve()


def load_yaml(path: PathLike) -> Config:
    """YAML 파일 로드."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    if path.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"Unsupported config extension: {path.suffix}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML must be a mapping: {path}")

    return _to_config(data)


def save_yaml(config: Union[Config, Dict[str, Any]], path: PathLike) -> None:
    """설정을 YAML 파일로 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = _to_plain_dict(config)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def merge_dicts(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> Config:
    """
    재귀적으로 dict 병합.
    override 값이 base 값을 덮어쓴다.
    """
    merged = deepcopy(dict(base))

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return _to_config(merged)


def load_experiment_config(
    data_config_path: PathLike,
    model_config_path: PathLike,
    train_config_path: PathLike,
    project_root: Optional[PathLike] = None,
) -> Config:
    """
    data / model / train YAML을 읽어서 하나의 config로 묶는다.
    이후 train.py 에서 다음처럼 쓰면 된다.

    cfg = load_experiment_config(...)
    cfg.data.name
    cfg.model.backbone.name
    cfg.train.optimizer.lr
    """
    root = Path(project_root).resolve() if project_root is not None else get_project_root()

    data_cfg = load_yaml(resolve_path(data_config_path, root))
    model_cfg = load_yaml(resolve_path(model_config_path, root))
    train_cfg = load_yaml(resolve_path(train_config_path, root))

    cfg = Config(
        {
            "project_root": str(root),
            "data": data_cfg,
            "model": model_cfg,
            "train": train_cfg,
        }
    )

    return cfg


def apply_overrides(
    cfg: Union[Config, Dict[str, Any]],
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """
    코드에서 추가 override를 적용할 때 사용.
    예:
        overrides = {"train": {"train": {"epochs": 20}}}
    """
    cfg = _to_config(cfg)
    overrides = overrides or {}
    return merge_dicts(cfg, overrides)


def pretty_print_config(cfg: Union[Config, Dict[str, Any]]) -> str:
    """로그 출력용 YAML 문자열."""
    data = _to_plain_dict(cfg)
    return yaml.safe_dump(
        data,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )