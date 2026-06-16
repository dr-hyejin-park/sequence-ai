"""
Scam 이진 분류 Fine-tuning.

- 사전학습(MLM)된 인코더 가중치를 초기값으로 로드.
- 불균형(양성 1%) 대응:
    * pos_weight 적용 BCEWithLogitsLoss (또는 Focal Loss)
    * WeightedRandomSampler로 미니배치 내 양성 비율을 끌어올림
- Stratified train/valid 분할 후 두 세트의 성능을 모두 보고.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import ModelConfig, FinetuneConfig, DataConfig
from src import utils, evaluate
from src.dataset import ClassificationDataset, load_arrays
from src.model import ScamClassifier


def _focal_loss(logits, targets, gamma, pos_weight):
    p = torch.sigmoid(logits)
    ce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    pt = torch.where(targets == 1, p, 1 - p)
    return ((1 - pt) ** gamma * ce).mean()


@torch.no_grad()
def _predict(model, dl, device):
    model.eval()
    scores, ys = [], []
    for batch in dl:
        ids = batch["input_ids"].to(device)
        tb = batch["time_buckets"].to(device)
        am = batch["attention_mask"].to(device)
        logits = model(ids, tb, am)
        scores.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(batch["label"].numpy())
    return np.concatenate(ys), np.concatenate(scores)


def finetune(npz_path, model_cfg: ModelConfig, ft_cfg: FinetuneConfig,
             data_cfg: DataConfig, pretrained_ckpt=None, device=None):
    device = device or utils.get_device()
    arrays = load_arrays(npz_path)
    labels = arrays["labels"]
    n = len(labels)
    max_total_len = data_cfg.max_seq_len + 1

    # stratified split (불균형 유지)
    tr_idx, va_idx = train_test_split(
        np.arange(n), test_size=ft_cfg.valid_ratio,
        stratify=labels, random_state=data_cfg.seed,
    )

    train_ds = ClassificationDataset(arrays, tr_idx, max_total_len)
    valid_ds = ClassificationDataset(arrays, va_idx, max_total_len)

    # 미니배치 양성 비율 상향: 오버샘플링으로 목표 비율(TARGET_POS_FRAC)까지만 끌어올린다.
    # (오버샘플링과 pos_weight를 동시에 과하게 적용하면 모델이 전부 양성으로 붕괴하므로
    #  여기서는 오버샘플링으로 대부분 보정하고, 손실 pos_weight는 잔여 불균형만 보정한다.)
    TARGET_POS_FRAC = 0.25
    y_tr = labels[tr_idx]
    n_pos, n_neg = int(y_tr.sum()), int((y_tr == 0).sum())
    pos_up = (TARGET_POS_FRAC / (1 - TARGET_POS_FRAC)) * (n_neg / max(1, n_pos))
    sample_w = np.where(y_tr == 1, pos_up, 1.0)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(tr_idx), replacement=True,
    )

    train_dl = DataLoader(train_ds, batch_size=ft_cfg.batch_size,
                          sampler=sampler, num_workers=2, drop_last=True)
    eval_train_dl = DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=2)
    valid_dl = DataLoader(valid_ds, batch_size=512, shuffle=False, num_workers=2)

    model = ScamClassifier(model_cfg).to(device)
    used_pretrained = False
    if pretrained_ckpt and os.path.exists(pretrained_ckpt):
        missing, unexpected = model.load_pretrained_encoder(pretrained_ckpt, map_location=device)
        used_pretrained = True
        print(f"[finetune] loaded pretrained encoder (missing={len(missing)}, "
              f"unexpected={len(unexpected)})")
    else:
        print("[finetune] WARNING: no pretrained encoder -> random init")

    # pos_weight: 오버샘플링 후의 '잔여' 불균형만 보정 (목표 비율 기준)
    pw = ((1 - TARGET_POS_FRAC) / TARGET_POS_FRAC) if ft_cfg.use_class_weight else 1.0
    pw = min(ft_cfg.pos_weight_cap, pw)
    pos_weight = torch.tensor([pw], device=device)
    print(f"[finetune] train n={len(tr_idx):,} (pos={n_pos}), valid n={len(va_idx):,} "
          f"(pos={int(labels[va_idx].sum())}), pos_weight={pw:.1f}")

    opt = torch.optim.AdamW(model.parameters(), lr=ft_cfg.lr,
                            weight_decay=ft_cfg.weight_decay)
    total_steps = len(train_dl) * ft_cfg.epochs
    sched = utils.linear_warmup_cosine(opt, total_steps, ft_cfg.warmup_ratio)

    best_valid_ap, best_state = -1.0, None
    for ep in range(ft_cfg.epochs):
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        for batch in train_dl:
            ids = batch["input_ids"].to(device)
            tb = batch["time_buckets"].to(device)
            am = batch["attention_mask"].to(device)
            y = batch["label"].to(device)

            logits = model(ids, tb, am)
            if ft_cfg.use_focal_loss:
                loss = _focal_loss(logits, y, ft_cfg.focal_gamma, pos_weight)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item() * ids.size(0)
            seen += ids.size(0)

        yv, sv = _predict(model, valid_dl, device)
        from sklearn.metrics import average_precision_score, roc_auc_score
        ap = average_precision_score(yv, sv)
        auc = roc_auc_score(yv, sv)
        print(f"[finetune] epoch {ep+1}/{ft_cfg.epochs} loss={running/seen:.4f} "
              f"valid_AP={ap:.4f} valid_AUC={auc:.4f} ({time.time()-t0:.1f}s)")
        if ap > best_valid_ap:
            best_valid_ap = ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # 최종 평가 (Train / Valid)
    y_tr_all, s_tr = _predict(model, eval_train_dl, device)
    y_va_all, s_va = _predict(model, valid_dl, device)
    train_m = evaluate.compute_metrics(y_tr_all, s_tr, ft_cfg.threshold)
    valid_m = evaluate.compute_metrics(y_va_all, s_va, ft_cfg.threshold)

    os.makedirs(os.path.dirname(ft_cfg.ckpt_path), exist_ok=True)
    torch.save({"model": model.state_dict(), "model_cfg": vars(model_cfg)},
               ft_cfg.ckpt_path)

    extra = (f"- 사전학습 인코더 사용: **{'예' if used_pretrained else '아니오(랜덤 초기화)'}**\n"
             f"- 불균형 대응: pos_weight={pw:.1f}"
             f"{', focal_loss' if ft_cfg.use_focal_loss else ''}, "
             f"WeightedRandomSampler 오버샘플링\n"
             f"- Train/Valid 분할: stratified {1-ft_cfg.valid_ratio:.0%}/{ft_cfg.valid_ratio:.0%}")
    md = evaluate.save_report(ft_cfg.report_dir, train_m, valid_m, extra)
    print(f"[finetune] saved report -> {ft_cfg.report_dir}/performance_report.md")
    return train_m, valid_m, md
