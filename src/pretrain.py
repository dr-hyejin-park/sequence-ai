"""
MLM(Masked Language Modeling) 사전학습.

라벨 없이 전체 사용자(정상+scam)의 행동 시퀀스로 인코더를 자기지도 학습한다.
이렇게 학습된 인코더 가중치를 이후 scam 분류 fine-tuning의 초기값으로 사용한다.
(요구사항: "Sequence를 사전학습한 모형을 참조하여 Fine-tuning")
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ModelConfig, PretrainConfig, DataConfig
from src import utils
from src.dataset import MLMDataset, load_arrays, IGNORE_INDEX
from src.model import MLMModel


def pretrain(npz_path, model_cfg: ModelConfig, pre_cfg: PretrainConfig,
             data_cfg: DataConfig, device=None):
    device = device or utils.get_device()
    arrays = load_arrays(npz_path)
    n = len(arrays["labels"])
    max_total_len = data_cfg.max_seq_len + 1  # +[CLS]

    idx = list(range(n))
    if pre_cfg.max_users:
        idx = idx[: pre_cfg.max_users]

    ds = MLMDataset(arrays, idx, max_total_len, mask_prob=pre_cfg.mask_prob)
    dl = DataLoader(ds, batch_size=pre_cfg.batch_size, shuffle=True,
                    num_workers=2, drop_last=True)

    model = MLMModel(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=pre_cfg.lr,
                            weight_decay=pre_cfg.weight_decay)
    total_steps = len(dl) * pre_cfg.epochs
    sched = utils.linear_warmup_cosine(opt, total_steps, pre_cfg.warmup_ratio)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    print(f"[pretrain] users={len(idx):,} steps/epoch={len(dl)} device={device}")
    model.train()
    for ep in range(pre_cfg.epochs):
        t0 = time.time()
        running, seen = 0.0, 0
        for batch in dl:
            ids = batch["input_ids"].to(device)
            tb = batch["time_buckets"].to(device)
            am = batch["attention_mask"].to(device)
            yl = batch["mlm_labels"].to(device)

            logits = model(ids, tb, am)
            loss = loss_fn(logits.view(-1, logits.size(-1)), yl.view(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item() * ids.size(0)
            seen += ids.size(0)
        print(f"[pretrain] epoch {ep+1}/{pre_cfg.epochs} "
              f"loss={running/seen:.4f} ({time.time()-t0:.1f}s)")

    os.makedirs(os.path.dirname(pre_cfg.ckpt_path), exist_ok=True)
    torch.save({"encoder": model.encoder.state_dict(),
                "model_cfg": vars(model_cfg)}, pre_cfg.ckpt_path)
    print(f"[pretrain] saved encoder -> {pre_cfg.ckpt_path}")
    return pre_cfg.ckpt_path
