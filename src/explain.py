"""
Integrated Gradients(IG) 기반 설명가능성.

모델이 scam(Y=1)으로 예측한 사례에 대해, 시퀀스 내 각 토큰(행동 이벤트)이
예측 점수에 기여한 정도를 Integrated Gradients로 정량화하고, 사례별 상위 기여
토큰을 리포트로 정리한다.

방법:
  - 이산 토큰을 직접 적분할 수 없으므로 '토큰 임베딩' 공간에서 IG를 계산한다.
  - 기준선(baseline): 내용 토큰을 모두 [PAD]로 치환한 임베딩
    (= "행동이 없었을 때"의 참조점). [CLS]/[PAD] 위치는 입력과 동일하므로 기여도 0.
  - IG_pos = (E_input - E_base) ⊙ ∫_0^1 ∂f/∂E (E_base + α(E_input-E_base)) dα
    를 임베딩 차원으로 합산하여 토큰별 기여도를 얻는다(f=scam 로짓).
  - 적분은 midpoint Riemann 합(steps 구간)으로 근사하며, completeness
    (Σ기여도 ≈ f(input) - f(baseline))로 근사 품질을 함께 보고한다.
"""

from __future__ import annotations

import numpy as np
import torch

from src import vocab

# 시간 간격 버킷(데이터 생성과 동일한 경계)에 대한 사람이 읽기 쉬운 라벨
TIME_BUCKET_LABELS = ["<1h", "1-3h", "3-6h", "6-12h", "12-24h", "1-2d", "2-4d", ">4d"]


def time_bucket_label(b: int) -> str:
    return TIME_BUCKET_LABELS[b] if 0 <= b < len(TIME_BUCKET_LABELS) else str(b)


@torch.no_grad()
def _baseline_embeds(encoder, ids):
    """내용 토큰을 [PAD]로 치환한 기준선 임베딩."""
    base_ids = ids.clone()
    content = (ids != vocab.CLS_ID) & (ids != vocab.PAD_ID)
    base_ids[content] = vocab.PAD_ID
    return encoder.embed_tokens(base_ids), base_ids


def integrated_gradients(model, input_ids, time_buckets, attn, steps=64, device="cpu"):
    """
    단일 사례에 대한 IG.
    반환: (attributions(L,), f_input, f_baseline, completeness_error)
    """
    model.eval()
    enc = model.encoder
    ids = input_ids.unsqueeze(0).to(device)          # (1, L)
    tb = time_buckets.unsqueeze(0).to(device)
    am = attn.unsqueeze(0).to(device)

    with torch.no_grad():
        input_embeds = enc.embed_tokens(ids)         # (1, L, H)
        base_embeds, _ = _baseline_embeds(enc, ids)  # (1, L, H)
    delta = input_embeds - base_embeds               # (1, L, H)

    # midpoint Riemann: α = (k+0.5)/steps
    alphas = (torch.arange(steps, device=device) + 0.5) / steps
    interp = base_embeds + alphas.view(-1, 1, 1) * delta   # (S, L, H)
    interp.requires_grad_(True)

    tb_s = tb.expand(steps, -1)
    am_s = am.expand(steps, -1)
    logits = model.forward_from_embeds(interp, tb_s, am_s)  # (S,)
    grads = torch.autograd.grad(logits.sum(), interp)[0]    # (S, L, H)
    avg_grad = grads.mean(dim=0, keepdim=True)              # (1, L, H)

    ig = (delta * avg_grad).sum(dim=-1).squeeze(0)          # (L,)
    ig_np = ig.detach().cpu().numpy()

    with torch.no_grad():
        f_in = float(model.forward_from_embeds(input_embeds, tb, am).item())
        f_base = float(model.forward_from_embeds(base_embeds, tb, am).item())
    completeness_err = float(abs(ig_np.sum() - (f_in - f_base)))
    return ig_np, f_in, f_base, completeness_err


def top_contributors(input_ids, time_buckets, attributions, top_k=10):
    """
    상위 기여 토큰(예측을 scam 쪽으로 민 토큰)을 정리.
    [CLS]/[PAD] 위치는 제외한다.
    """
    L = len(input_ids)
    rows = []
    total_pos = float(sum(float(a) for a in attributions if a > 0)) + 1e-12
    for p in range(L):
        tid = int(input_ids[p])
        if tid in (vocab.CLS_ID, vocab.PAD_ID):
            continue
        rows.append({
            "position": int(p),
            "token": vocab.id2token[tid],
            "is_risk": bool(vocab.is_risk_event(vocab.id2token[tid])),
            "time_gap": time_bucket_label(int(time_buckets[p])),
            "attribution": float(attributions[p]),
            "share": float(attributions[p]) / total_pos,  # 양의 기여 합 대비 비중
        })
    rows.sort(key=lambda r: r["attribution"], reverse=True)
    return rows[:top_k]


def sequence_render(input_ids, attributions, tail=18):
    """가장 최근 행동 tail개(패딩 제외)를 시간순으로 기여도와 함께 렌더링."""
    L = len(input_ids)
    content = [p for p in range(L)
               if int(input_ids[p]) not in (vocab.CLS_ID, vocab.PAD_ID)]
    recent = content[-tail:]  # 좌측 정렬이므로 뒤쪽이 최근 행동
    parts = []
    for p in recent:
        tok = vocab.id2token[int(input_ids[p])]
        a = attributions[p]
        mark = "▲" if a > 0 else "▽"
        parts.append(f"{tok}({mark}{a:+.2f})")
    return " · ".join(parts)
