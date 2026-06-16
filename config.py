"""
중앙 설정(Config) 모듈.

Scam 예측 Sequence-AI 파이프라인 전반에서 공유하는 하이퍼파라미터를 모아둔다.
- DataConfig    : 합성 행동 시퀀스 데이터 생성 설정
- ModelConfig   : BERT 스타일 Transformer 인코더 구조 설정
- PretrainConfig: MLM 사전학습 설정
- FinetuneConfig: scam 이진 분류 fine-tuning 설정

CPU 환경에서도 현실적인 시간 안에 학습이 끝나도록 기본값을 "작은 모델"로 두었으며,
모든 값은 run_pipeline.py 의 CLI 인자나 환경에 맞춰 조정할 수 있다.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    # ----- 모집단 / 불균형 설정 -----
    n_users: int = 100_000          # 전체 사용자 수
    n_scam: int = 1_000             # scam(Y=1) 사용자 수 (불균형: 1%)
    seed: int = 42

    # ----- 시퀀스(관찰 윈도우) 설정 -----
    observation_days: int = 21      # 관찰 기간(일): cutoff 이전 N일의 행동을 입력으로 사용
    label_horizon_days: int = 3     # cutoff 이후 N일 내 scam 발생 시 Y=1 (요구사항: 3일)
    max_seq_len: int = 64           # 모델 입력 최대 토큰 수(이벤트 수). 초과 시 최근 이벤트만 사용
    min_seq_len: int = 5            # 최소 이벤트 수(너무 짧은 시퀀스 방지)

    # ----- 행동 강도 -----
    base_events_per_day_mean: float = 4.0   # 1일 평균 이벤트 수(개인별로 변동)

    # 산출물 경로
    out_dir: str = "data"


@dataclass
class ModelConfig:
    hidden_size: int = 128
    num_layers: int = 3
    num_heads: int = 4
    intermediate_size: int = 256
    dropout: float = 0.1
    max_position: int = 128         # DataConfig.max_seq_len(+[CLS]) 보다 충분히 크게
    num_time_buckets: int = 8       # 이벤트 간 시간 간격 버킷 임베딩 수


@dataclass
class PretrainConfig:
    epochs: int = 3
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 0.01
    mask_prob: float = 0.15         # BERT MLM 마스킹 비율
    warmup_ratio: float = 0.05
    max_users: Optional[int] = None # None이면 전체 사용자로 사전학습
    ckpt_path: str = "artifacts/pretrained_encoder.pt"


@dataclass
class FinetuneConfig:
    epochs: int = 5
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    valid_ratio: float = 0.2
    # 불균형 대응
    use_class_weight: bool = True   # 양성 클래스에 손실 가중
    pos_weight_cap: float = 50.0    # pos_weight 상한(과도한 가중 방지)
    use_focal_loss: bool = False    # True면 focal loss 사용
    focal_gamma: float = 2.0
    # 결정 임계값(리포트용). PR 곡선 기반 best-F1 임계값도 함께 보고
    threshold: float = 0.5
    ckpt_path: str = "artifacts/scam_classifier.pt"
    report_dir: str = "reports"


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)


def quick_config() -> PipelineConfig:
    """CPU에서 빠르게 전체 파이프라인을 검증하기 위한 축소 설정."""
    cfg = PipelineConfig()
    cfg.data.n_users = 20_000
    cfg.data.n_scam = 200
    cfg.pretrain.epochs = 2
    cfg.finetune.epochs = 4
    return cfg
