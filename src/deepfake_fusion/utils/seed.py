from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """
    random / numpy / torch 전역 시드 고정.

    Args:
        seed: 사용할 시드 값
        deterministic: True면 가능한 범위에서 연산을 deterministic 하게 설정

    Returns:
        설정된 seed 값
    """
    if seed is None:
        raise ValueError("seed must not be None")

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

    return seed


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker별 시드 고정 함수.

    예:
        DataLoader(..., worker_init_fn=seed_worker, generator=get_torch_generator(seed))
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_torch_generator(seed: int = 42) -> torch.Generator:
    """
    DataLoader shuffle 재현성을 위한 torch.Generator 반환.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator