"""
설명가능성 리포트 생성기 (Integrated Gradients).

학습된 scam 분류기를 불러와, **검증셋에서 모델이 scam으로 예측한 사례**들에 대해
각 토큰의 기여도를 Integrated Gradients로 측정하고, 사례별 상위 기여 토큰과
전체 집계를 리포트(reports/explainability_report.md, .json, 그림)로 저장한다.

사용:
  python run_explain.py                 # 기본(검증셋, 상위 20개 사례 상세)
  python run_explain.py --n-detail 30 --ig-steps 96
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import PipelineConfig
from src import utils, explain, vocab
from src.dataset import ClassificationDataset, load_arrays
from src.model import ScamClassifier
from config import ModelConfig


def _load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    mcfg = ModelConfig(**ckpt["model_cfg"]) if "model_cfg" in ckpt else ModelConfig()
    model = ScamClassifier(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, mcfg


@torch.no_grad()
def _predict(model, dl, device):
    scores = []
    for b in dl:
        logits = model(b["input_ids"].to(device), b["time_buckets"].to(device),
                       b["attention_mask"].to(device))
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-detail", type=int, default=20, help="상세 분석할 사례 수")
    ap.add_argument("--ig-steps", type=int, default=64, help="IG 적분 스텝 수")
    ap.add_argument("--max-global", type=int, default=300,
                    help="전체 집계에 사용할 예측-scam 최대 사례 수")
    ap.add_argument("--data", default="data/sequences.npz")
    ap.add_argument("--ckpt", default="artifacts/scam_classifier.pt")
    args = ap.parse_args()

    cfg = PipelineConfig()
    utils.set_seed(cfg.data.seed)
    device = utils.get_device()
    print(f"=== Explainability (Integrated Gradients), device={device} ===")

    arrays = load_arrays(args.data)
    labels = arrays["labels"]
    n = len(labels)
    max_total_len = cfg.data.max_seq_len + 1

    # 학습과 동일한 stratified 분할 → 검증셋 확보
    _, va_idx = train_test_split(
        np.arange(n), test_size=cfg.finetune.valid_ratio,
        stratify=labels, random_state=cfg.data.seed,
    )
    valid_ds = ClassificationDataset(arrays, va_idx, max_total_len)
    dl = DataLoader(valid_ds, batch_size=512, shuffle=False)

    model, _ = _load_model(args.ckpt, device)
    scores = _predict(model, dl, device)

    thr = cfg.finetune.threshold
    pred_pos = np.where(scores >= thr)[0]           # valid_ds 내 지역 인덱스
    order = pred_pos[np.argsort(-scores[pred_pos])]  # 점수 내림차순
    print(f"[explain] valid n={len(va_idx):,}, predicted scam(>= {thr})={len(pred_pos)}")

    # ---- 전체 집계용 IG (상위 max_global개) ----
    global_idx = order[: args.max_global]
    tok_freq = defaultdict(int)         # 상위5 기여 토큰 등장 횟수
    tok_attr_sum = defaultdict(float)   # 기여도 누적
    comp_errs = []
    n_tp = n_fp = 0
    for li in global_idx:
        s = valid_ds[int(li)]
        ig, f_in, f_base, cerr = explain.integrated_gradients(
            model, s["input_ids"], s["time_buckets"], s["attention_mask"],
            steps=args.ig_steps, device=device)
        comp_errs.append(cerr)
        true_y = int(s["label"].item())
        n_tp += int(true_y == 1)
        n_fp += int(true_y == 0)
        for r in explain.top_contributors(s["input_ids"], s["time_buckets"], ig, top_k=5):
            if r["attribution"] > 0:
                tok_freq[r["token"]] += 1
                tok_attr_sum[r["token"]] += r["attribution"]

    def _explain_case(li):
        s = valid_ds[int(li)]
        ig, f_in, f_base, cerr = explain.integrated_gradients(
            model, s["input_ids"], s["time_buckets"], s["attention_mask"],
            steps=args.ig_steps, device=device)
        return {
            "user_index": int(va_idx[int(li)]),
            "true_label": int(s["label"].item()),
            "pred_score": float(scores[int(li)]),
            "logit_input": f_in, "logit_baseline": f_base,
            "completeness_error": cerr,
            "top_contributors": explain.top_contributors(
                s["input_ids"], s["time_buckets"], ig, top_k=10),
            "sequence_tail": explain.sequence_render(s["input_ids"], ig),
        }

    # ---- 상세 사례 (상위 점수 n_detail개) ----
    detail = [_explain_case(li) for li in order[: args.n_detail]]

    # ---- 오탐(FP) 사례: 예측은 scam이나 실제 정상 ----
    fp_local = [int(li) for li in order
                if int(valid_ds[int(li)]["label"].item()) == 0]
    detail_fp = [_explain_case(li) for li in fp_local]

    # ---- 그림: 전체 상위 기여 행동 ----
    os.makedirs("reports/figures", exist_ok=True)
    fig_path = "reports/figures/global_top_tokens.png"
    _plot_global(tok_attr_sum, fig_path)

    # ---- 리포트 작성 ----
    summary = {
        "valid_n": int(len(va_idx)),
        "predicted_scam": int(len(pred_pos)),
        "threshold": thr,
        "global_cases_used": int(len(global_idx)),
        "global_tp": n_tp, "global_fp": n_fp,
        "precision_in_predicted": (n_tp / len(global_idx)) if len(global_idx) else float("nan"),
        "mean_completeness_error": float(np.mean(comp_errs)) if comp_errs else float("nan"),
        "ig_steps": args.ig_steps,
    }
    global_rank = sorted(tok_attr_sum.items(), key=lambda kv: kv[1], reverse=True)
    global_table = [
        {"token": t, "is_risk": vocab.is_risk_event(t),
         "freq_in_top5": tok_freq[t], "total_attribution": a}
        for t, a in global_rank
    ]

    os.makedirs("reports", exist_ok=True)
    with open("reports/explainability.json", "w") as f:
        json.dump({"summary": summary, "global": global_table,
                   "cases": detail, "false_positives": detail_fp},
                  f, indent=2, ensure_ascii=False)
    md = _format_md(summary, global_table, detail, detail_fp, fig_path)
    with open("reports/explainability_report.md", "w") as f:
        f.write(md)

    print(f"[explain] saved reports/explainability_report.md "
          f"(predicted scam={len(pred_pos)}, detailed={len(detail)})")
    print("\n" + md[:1500])


def _plot_global(tok_attr_sum, path, top=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(tok_attr_sum.items(), key=lambda kv: kv[1], reverse=True)[:top]
    if not items:
        return
    toks = [t for t, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    colors = ["#c0392b" if vocab.is_risk_event(t) else "#7f8c8d" for t in toks]
    plt.figure(figsize=(9, max(4, 0.35 * len(toks))))
    plt.barh(toks, vals, color=colors)
    plt.xlabel("Total Integrated-Gradients attribution (toward scam)")
    plt.title("Top contributing behavior tokens across predicted-scam cases\n"
              "(red = risk leading-indicator, gray = everyday)")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def _case_block(c, k):
    tag = "정탐(TP)" if c["true_label"] == 1 else "오탐(FP)"
    out = [
        f"### 사례 {k}: user #{c['user_index']} — 예측확률 {c['pred_score']:.4f} [{tag}]",
        "",
        f"- 로짓: input={c['logit_input']:.3f}, baseline={c['logit_baseline']:.3f}, "
        f"completeness 오차={c['completeness_error']:.4f}",
        f"- 최근 행동 흐름(기여도): {c['sequence_tail']}",
        "",
        "| 순위 | 토큰(행동) | 위치 | 직전간격 | 위험지표 | 기여도 | 비중 |",
        "|---:|---|---:|:---:|:---:|---:|---:|",
    ]
    for j, r in enumerate(c["top_contributors"], 1):
        out.append(f"| {j} | `{r['token']}` | {r['position']} | {r['time_gap']} | "
                   f"{'✔' if r['is_risk'] else ''} | {r['attribution']:+.3f} | "
                   f"{r['share']*100:.1f}% |")
    out.append("")
    return out


def _format_md(summary, global_table, detail, detail_fp, fig_path):
    L = ["# Scam 예측 설명가능성 리포트 (Integrated Gradients)", ""]
    L += [
        "모델이 **scam(Y=1)으로 예측한 검증셋 사례**들에 대해, 시퀀스 내 각 토큰(행동 "
        "이벤트)이 예측 점수를 scam 쪽으로 민 정도를 Integrated Gradients로 측정했다. "
        "기준선(baseline)은 모든 행동 토큰을 `[PAD]`로 치환한 '행동 부재' 상태이며, "
        "기여도 합은 completeness 성질에 의해 `f(input) − f(baseline)`(로짓 차이)와 같아야 한다.",
        "",
        "## 요약",
        "",
        f"- 검증셋 표본수: {summary['valid_n']:,}",
        f"- 모델이 scam으로 예측(score ≥ {summary['threshold']:.2f}): "
        f"**{summary['predicted_scam']}건**",
        f"- 설명 대상(상위 점수 {summary['global_cases_used']}건) 중 실제 scam(TP) "
        f"{summary['global_tp']}건 / 오탐(FP) {summary['global_fp']}건 "
        f"(정밀도 {summary['precision_in_predicted']:.3f})",
        f"- IG 적분 스텝: {summary['ig_steps']}, 평균 completeness 오차: "
        f"{summary['mean_completeness_error']:.4f} (작을수록 근사 양호)",
        "",
        "## 전체 집계: scam 예측을 가장 많이 견인한 행동",
        "",
        f"![global top tokens]({os.path.relpath(fig_path, 'reports')})",
        "",
        "| 순위 | 토큰(행동) | 위험지표 | 상위5 등장수 | 누적 기여도 |",
        "|---:|---|:---:|---:|---:|",
    ]
    for i, r in enumerate(global_table[:20], 1):
        L.append(f"| {i} | `{r['token']}` | {'✔' if r['is_risk'] else ''} | "
                 f"{r['freq_in_top5']} | {r['total_attribution']:.3f} |")
    L += ["", "## 정탐 사례 상세 (예측확률 상위 순)", ""]
    for k, c in enumerate(detail, 1):
        L += _case_block(c, k)

    L += ["", "## 오탐(FP) 사례 상세 — 모델이 잘못 scam으로 본 이유", ""]
    if detail_fp:
        L += [f"검증셋에서 예측 scam이지만 실제 정상인 사례 {len(detail_fp)}건. "
              "정상 사용자가 위험 행동을 우연히 군집해 수행하면 오탐이 발생할 수 있다.", ""]
        for k, c in enumerate(detail_fp, 1):
            L += _case_block(c, k)
    else:
        L += ["검증셋에 오탐(FP) 사례가 없습니다.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
