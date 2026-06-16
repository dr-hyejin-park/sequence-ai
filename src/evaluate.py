"""
성능 평가 및 리포트 생성.

불균형 이진 분류에 적합한 지표를 계산한다:
  - ROC-AUC, PR-AUC(Average Precision)
  - 임계값별 Precision/Recall/F1, Confusion Matrix
  - PR 곡선 기반 Best-F1 임계값
  - Recall@TopK% (운영 관점: 상위 위험군 K%를 검수했을 때 잡아내는 scam 비율)
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def recall_at_topk(y_true, scores, k_frac):
    """점수 상위 k_frac 비율을 양성으로 봤을 때의 recall과 precision."""
    n = len(scores)
    k = max(1, int(round(n * k_frac)))
    order = np.argsort(-scores)
    top = order[:k]
    tp = int(y_true[top].sum())
    total_pos = int(y_true.sum())
    recall = tp / total_pos if total_pos > 0 else float("nan")
    precision = tp / k
    return {"k_frac": k_frac, "k": k, "recall": recall, "precision": precision}


def best_f1_threshold(y_true, scores):
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    # 마지막 점(thr 없음) 제외
    i = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
    t = float(thr[i]) if len(thr) > 0 else 0.5
    return t, float(f1[i])


def compute_metrics(y_true, scores, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    roc = _safe_auc(y_true, scores)
    pr_auc = float(average_precision_score(y_true, scores)) if y_true.sum() > 0 else float("nan")

    y_pred = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    bt, bf1 = best_f1_threshold(y_true, scores)
    yb = (scores >= bt).astype(int)
    pb, rb, fb, _ = precision_recall_fscore_support(
        y_true, yb, average="binary", zero_division=0
    )

    return {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "pos_rate": float(y_true.mean()),
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "threshold": float(threshold),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
        "best_f1_threshold": bt,
        "best_f1": bf1,
        "best_f1_precision": float(pb),
        "best_f1_recall": float(rb),
        "recall_at_topk": [
            recall_at_topk(y_true, scores, kf) for kf in (0.01, 0.02, 0.05, 0.10)
        ],
    }


def format_report(train_m, valid_m, extra=None):
    def block(name, m):
        cm = m["confusion_matrix"]
        lines = [
            f"## {name} 성능",
            "",
            f"- 표본수 n = {m['n']:,} (양성 {m['n_pos']:,}, 양성비율 {m['pos_rate']*100:.3f}%)",
            f"- **ROC-AUC** : {m['roc_auc']:.4f}",
            f"- **PR-AUC (Average Precision)** : {m['pr_auc']:.4f}",
            "",
            f"고정 임계값(threshold={m['threshold']:.2f}) 기준:",
            f"- Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  F1={m['f1']:.4f}",
            f"- Confusion Matrix [[TN,FP],[FN,TP]] = {cm}",
            "",
            f"Best-F1 임계값({m['best_f1_threshold']:.4f}) 기준:",
            f"- Precision={m['best_f1_precision']:.4f}  Recall={m['best_f1_recall']:.4f}  "
            f"F1={m['best_f1']:.4f}",
            "",
            "운영 관점 Recall@TopK% (상위 위험군 검수 시 적발률):",
        ]
        for rk in m["recall_at_topk"]:
            lines.append(
                f"- 상위 {rk['k_frac']*100:.0f}% (n={rk['k']:,}): "
                f"Recall={rk['recall']:.4f}  Precision={rk['precision']:.4f}"
            )
        lines.append("")
        return "\n".join(lines)

    out = ["# Scam 예측 Sequence-AI 성능 리포트", ""]
    if extra:
        out.append(extra)
        out.append("")
    out.append(block("Train set", train_m))
    out.append(block("Valid set", valid_m))
    return "\n".join(out)


def save_report(report_dir, train_m, valid_m, extra=None):
    os.makedirs(report_dir, exist_ok=True)
    md = format_report(train_m, valid_m, extra)
    with open(os.path.join(report_dir, "performance_report.md"), "w") as f:
        f.write(md)
    with open(os.path.join(report_dir, "metrics.json"), "w") as f:
        json.dump({"train": train_m, "valid": valid_m}, f, indent=2, ensure_ascii=False)
    return md
