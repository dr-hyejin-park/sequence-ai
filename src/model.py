"""
BERT 스타일 Transformer 인코더와 두 개의 헤드.

  - SequenceBertEncoder : 토큰/위치/시간버킷 임베딩 + Transformer Encoder
  - MLMHead             : 사전학습(Masked Language Modeling)용 토큰 복원 헤드
  - ScamClassifier      : fine-tuning용 [CLS] 풀링 + 이진 분류 헤드

행동 시퀀스를 자연어 문장처럼 다루어, 먼저 자기지도(MLM)로 행동 표현을 사전학습한 뒤
같은 인코더 가중치를 분류 과제로 fine-tuning 한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src import vocab
from config import ModelConfig


class SequenceBertEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(vocab.VOCAB_SIZE, cfg.hidden_size, padding_idx=vocab.PAD_ID)
        self.pos_emb = nn.Embedding(cfg.max_position, cfg.hidden_size)
        self.time_emb = nn.Embedding(cfg.num_time_buckets, cfg.hidden_size)
        self.ln = nn.LayerNorm(cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_size,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.intermediate_size,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)

    def embed_tokens(self, token_ids):
        """토큰 id -> 토큰 임베딩 (Integrated Gradients의 적분 대상)."""
        return self.token_emb(token_ids)

    def forward_from_embeds(self, token_embeds, time_buckets, attention_mask):
        """
        토큰 임베딩을 직접 입력으로 받아 인코딩한다(IG용).
        token_embeds: (B, L, H)  — 위치/시간 임베딩은 내부에서 더한다.
        """
        B, L, _ = token_embeds.shape
        pos = torch.arange(L, device=token_embeds.device).unsqueeze(0).expand(B, L)
        x = token_embeds + self.pos_emb(pos) + self.time_emb(time_buckets)
        x = self.dropout(self.ln(x))
        key_padding_mask = attention_mask == 0
        return self.encoder(x, src_key_padding_mask=key_padding_mask)

    def forward(self, token_ids, time_buckets, attention_mask):
        """
        token_ids:      (B, L) long
        time_buckets:   (B, L) long
        attention_mask: (B, L) 1=유효, 0=패딩
        반환: (B, L, H) 토큰별 표현
        """
        return self.forward_from_embeds(
            self.embed_tokens(token_ids), time_buckets, attention_mask
        )


class MLMHead(nn.Module):
    """마스킹된 토큰을 어휘 전체에 대해 예측."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(cfg.hidden_size)
        self.decoder = nn.Linear(cfg.hidden_size, vocab.VOCAB_SIZE)

    def forward(self, hidden):
        return self.decoder(self.ln(self.act(self.dense(hidden))))


class MLMModel(nn.Module):
    """사전학습용: 인코더 + MLM 헤드."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.encoder = SequenceBertEncoder(cfg)
        self.mlm_head = MLMHead(cfg)

    def forward(self, token_ids, time_buckets, attention_mask):
        h = self.encoder(token_ids, time_buckets, attention_mask)
        return self.mlm_head(h)


class ScamClassifier(nn.Module):
    """fine-tuning용: (사전학습된) 인코더 + [CLS] 풀링 + 이진 분류."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.encoder = SequenceBertEncoder(cfg)
        self.pooler = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.Tanh(),
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.hidden_size, 1)

    def _head(self, h):
        cls = h[:, 0]                       # [CLS] 위치 표현
        pooled = self.dropout(self.pooler(cls))
        return self.classifier(pooled).squeeze(-1)   # (B,) logit

    def forward(self, token_ids, time_buckets, attention_mask):
        h = self.encoder(token_ids, time_buckets, attention_mask)
        return self._head(h)

    def forward_from_embeds(self, token_embeds, time_buckets, attention_mask):
        """토큰 임베딩을 직접 받아 로짓을 반환(IG용)."""
        h = self.encoder.forward_from_embeds(token_embeds, time_buckets, attention_mask)
        return self._head(h)

    def load_pretrained_encoder(self, ckpt_path, map_location="cpu"):
        """MLM 사전학습 체크포인트에서 인코더 가중치만 로드."""
        state = torch.load(ckpt_path, map_location=map_location)
        enc_state = state["encoder"] if "encoder" in state else state
        missing, unexpected = self.encoder.load_state_dict(enc_state, strict=False)
        return missing, unexpected
