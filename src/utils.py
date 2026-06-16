"""공용 유틸리티: 시드 고정, 디바이스 선택, 선형 워밍업 스케줄러."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_warmup_cosine(optimizer, total_steps, warmup_ratio=0.05):
    """선형 워밍업 후 코사인 감쇠 LR 스케줄러."""
    warmup = max(1, int(total_steps * warmup_ratio))

    def fn(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)
