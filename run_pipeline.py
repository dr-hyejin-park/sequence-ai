"""
엔드투엔드 파이프라인 실행기.

단계:
  1) 합성 행동 시퀀스 데이터 생성 (10만명, scam 1%)
  2) MLM 사전학습 (라벨 없이 전체 시퀀스)
  3) scam 분류 fine-tuning (사전학습 인코더 로드)
  4) Train/Valid 성능 리포트 저장

사용:
  python run_pipeline.py                 # 기본(10만명) 전체 실행
  python run_pipeline.py --quick         # 축소 설정으로 빠른 검증
  python run_pipeline.py --skip-data     # 기존 data/sequences.npz 재사용
  python run_pipeline.py --no-pretrain   # 사전학습 생략(랜덤 초기화 fine-tuning, 비교용)
"""

from __future__ import annotations

import argparse
import os

from config import PipelineConfig, quick_config
from src import utils, data_generation, pretrain as pretrain_mod, finetune as finetune_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="축소 설정으로 빠르게 실행")
    ap.add_argument("--skip-data", action="store_true", help="기존 데이터 재사용")
    ap.add_argument("--no-pretrain", action="store_true", help="사전학습 생략(비교용)")
    ap.add_argument("--n-users", type=int, default=None)
    ap.add_argument("--n-scam", type=int, default=None)
    ap.add_argument("--pretrain-epochs", type=int, default=None)
    ap.add_argument("--finetune-epochs", type=int, default=None)
    args = ap.parse_args()

    cfg: PipelineConfig = quick_config() if args.quick else PipelineConfig()
    if args.n_users is not None:
        cfg.data.n_users = args.n_users
    if args.n_scam is not None:
        cfg.data.n_scam = args.n_scam
    if args.pretrain_epochs is not None:
        cfg.pretrain.epochs = args.pretrain_epochs
    if args.finetune_epochs is not None:
        cfg.finetune.epochs = args.finetune_epochs

    utils.set_seed(cfg.data.seed)
    device = utils.get_device()
    print(f"=== Scam Sequence-AI pipeline (device={device}) ===")

    npz_path = os.path.join(cfg.data.out_dir, "sequences.npz")
    if args.skip_data and os.path.exists(npz_path):
        print(f"[1/4] reuse existing data: {npz_path}")
    else:
        print("[1/4] generating synthetic behavior sequences ...")
        npz_path = data_generation.generate(cfg.data)

    ckpt = None
    if not args.no_pretrain:
        print("[2/4] MLM pretraining ...")
        ckpt = pretrain_mod.pretrain(npz_path, cfg.model, cfg.pretrain, cfg.data, device)
    else:
        print("[2/4] skip pretraining (random init)")

    print("[3/4] fine-tuning scam classifier ...")
    train_m, valid_m, md = finetune_mod.finetune(
        npz_path, cfg.model, cfg.finetune, cfg.data, pretrained_ckpt=ckpt, device=device
    )

    print("[4/4] done. report saved under reports/")
    print("\n" + md)


if __name__ == "__main__":
    main()
