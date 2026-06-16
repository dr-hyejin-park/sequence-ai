"""
PyTorch Dataset / DataLoader 유틸리티.

- 저장된 npz(token_ids, time_buckets, lengths, labels)를 읽어
  [CLS]를 맨 앞에 붙인 모델 입력 텐서를 만든다.
- MLMDataset       : 사전학습용. BERT 방식 마스킹(15%: 80/10/10) 적용.
- ClassificationDataset : fine-tuning용. [CLS] + 시퀀스 + 라벨.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src import vocab

IGNORE_INDEX = -100


def load_arrays(npz_path: str):
    d = np.load(npz_path)
    return {
        "token_ids": d["token_ids"].astype(np.int64),
        "time_buckets": d["time_buckets"].astype(np.int64),
        "lengths": d["lengths"].astype(np.int64),
        "labels": d["labels"].astype(np.int64),
        "scenarios": d["scenarios"].astype(np.int64),
    }


def _build_input(tok_row, time_row, length):
    """[CLS] 프리펜드 후 (input_ids, time_buckets, attention_mask) 구성."""
    L = int(length)
    n = L + 1  # +[CLS]
    input_ids = np.full(n, vocab.PAD_ID, dtype=np.int64)
    time_b = np.zeros(n, dtype=np.int64)
    input_ids[0] = vocab.CLS_ID
    input_ids[1:] = tok_row[:L]
    time_b[1:] = time_row[:L]
    attn = np.ones(n, dtype=np.int64)
    return input_ids, time_b, attn


class _BaseSeqDataset(Dataset):
    def __init__(self, arrays, indices, max_total_len):
        self.tok = arrays["token_ids"]
        self.time = arrays["time_buckets"]
        self.len = arrays["lengths"]
        self.labels = arrays["labels"]
        self.indices = indices
        self.max_total_len = max_total_len  # [CLS] 포함 최대 길이

    def __len__(self):
        return len(self.indices)

    def _padded_input(self, idx):
        u = self.indices[idx]
        ids, time_b, attn = _build_input(self.tok[u], self.time[u], self.len[u])
        T = self.max_total_len
        out_ids = np.full(T, vocab.PAD_ID, dtype=np.int64)
        out_time = np.zeros(T, dtype=np.int64)
        out_attn = np.zeros(T, dtype=np.int64)
        n = min(len(ids), T)
        out_ids[:n] = ids[:n]
        out_time[:n] = time_b[:n]
        out_attn[:n] = attn[:n]
        return out_ids, out_time, out_attn, u


class ClassificationDataset(_BaseSeqDataset):
    def __getitem__(self, idx):
        ids, time_b, attn, u = self._padded_input(idx)
        return {
            "input_ids": torch.from_numpy(ids),
            "time_buckets": torch.from_numpy(time_b),
            "attention_mask": torch.from_numpy(attn),
            "label": torch.tensor(float(self.labels[u])),
        }


class MLMDataset(_BaseSeqDataset):
    def __init__(self, arrays, indices, max_total_len, mask_prob=0.15, seed=0):
        super().__init__(arrays, indices, max_total_len)
        self.mask_prob = mask_prob
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, idx):
        ids, time_b, attn, _ = self._padded_input(idx)
        labels = np.full_like(ids, IGNORE_INDEX)

        # 마스킹 후보: 유효 토큰 중 특수토큰([CLS],[PAD]) 제외
        cand = (attn == 1) & (ids != vocab.CLS_ID) & (ids != vocab.PAD_ID)
        cand_pos = np.where(cand)[0]
        if cand_pos.size > 0:
            n_mask = max(1, int(round(cand_pos.size * self.mask_prob)))
            chosen = self.rng.choice(cand_pos, size=n_mask, replace=False)
            for p in chosen:
                labels[p] = ids[p]
                r = self.rng.random()
                if r < 0.8:
                    ids[p] = vocab.MASK_ID                       # 80%: [MASK]
                elif r < 0.9:
                    ids[p] = self.rng.integers(len(vocab.SPECIAL_TOKENS), vocab.VOCAB_SIZE)  # 10%: 랜덤
                # 10%: 원본 유지
        return {
            "input_ids": torch.from_numpy(ids),
            "time_buckets": torch.from_numpy(time_b),
            "attention_mask": torch.from_numpy(attn),
            "mlm_labels": torch.from_numpy(labels),
        }
