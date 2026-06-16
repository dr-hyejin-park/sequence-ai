"""
합성 행동 시퀀스 데이터 생성.

목표:
  - 10만 명 중 1,000명(1%)을 scam(Y=1)으로 두는 불균형 상황을 모사.
  - 각 사용자마다 '관찰 윈도우(cutoff 이전 observation_days일)' 동안의 행동 이벤트를
    시간순으로 나열한 시퀀스를 만든다.
  - 라벨 정의: cutoff 이후 label_horizon_days(=3일) 이내에 scam 사건이 발생하면 Y=1.
    → scam 사건 자체는 입력 시퀀스에 포함하지 않는다(미래 정보 누수 방지).
       모델은 cutoff '이전'의 선행 행동만으로 미래 scam을 예측한다.

설계 포인트:
  - 정상 사용자도 위험 이벤트(신규기기 로그인, OTP, 신규 수취인 등록 등)를 드물게 수행한다.
    따라서 단일 토큰으로는 분리되지 않고, '결합·순서·밀도' 패턴을 학습해야 한다.
  - Y=1 사용자는 cutoff 직전 1~3일 구간에 시나리오 기반의 위험 선행 지표가 군집한다.

산출물:
  data/sequences.npz  - token_ids, time_buckets, lengths, labels, scenarios
  data/meta.json      - 생성 통계
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np

from src import vocab
from config import DataConfig


# ---------------------------------------------------------------------------
# 정상(배경) 행동 분포: 흔한 저위험 이벤트에 높은 가중치, 위험 이벤트엔 매우 낮은 가중치
# ---------------------------------------------------------------------------
_BACKGROUND_WEIGHTS = {
    # 인증/세션 (자주)
    "open_app": 10.0, "login_success": 6.0, "check_notification": 8.0,
    "biometric_auth": 5.0, "logout": 4.0, "login_fail": 0.6,
    # 조회 (매우 자주)
    "view_balance": 9.0, "view_transactions": 7.0, "view_statement": 2.0,
    "view_card_benefit": 2.5, "view_point": 2.5, "search_branch_atm": 1.0,
    # 결제/소비
    "card_payment_small": 8.0, "card_payment_medium": 3.5, "payment_qr": 3.0,
    "payment_bill": 1.5, "pay_subscription": 1.2, "transit_payment": 6.0,
    # 이체
    "transfer_known_payee_small": 4.0, "transfer_known_payee_medium": 1.8,
    "transfer_self": 1.5, "deposit": 1.5, "withdrawal_atm": 1.5,
    # 저축/투자
    "view_savings": 1.5, "deposit_savings": 0.8, "view_fund": 1.0,
    "view_stock": 2.0, "trade_stock_small": 1.2,
    # 계정 관리/기타
    "update_profile": 0.4, "view_security_center": 0.5, "set_alarm": 0.6,
    "open_chatbot": 1.0, "view_event_promo": 2.0, "idle": 3.0,
    # 위험 이벤트의 '정상' 기저 발생률 (작게)
    "open_link_sms": 0.20, "open_link_messenger": 0.20, "install_external_app": 0.08,
    "grant_accessibility_perm": 0.02, "incoming_call_long": 0.25, "login_new_device": 0.10,
    "login_unusual_location": 0.10, "receive_otp": 0.40, "input_otp": 0.35,
    "change_password": 0.05, "reissue_cert": 0.04, "change_phone_number": 0.02,
    "disable_security_alert": 0.03, "increase_transfer_limit": 0.04, "add_new_payee": 0.30,
    "night_activity": 0.30, "apply_quick_loan": 0.05, "view_crypto": 0.40,
}


def _background_distribution():
    """배경 이벤트 토큰 id 배열과 정규화된 확률 벡터를 반환."""
    toks = list(_BACKGROUND_WEIGHTS.keys())
    ids = np.array([vocab.token2id[t] for t in toks], dtype=np.int64)
    w = np.array([_BACKGROUND_WEIGHTS[t] for t in toks], dtype=np.float64)
    w = w / w.sum()
    return ids, w


# ---------------------------------------------------------------------------
# scam 시나리오: cutoff 직전 윈도우에 군집되는 위험 선행 지표 패턴
# 각 시나리오는 (이벤트 토큰, 발생 가중치) 의 순서 있는 리스트.
# ---------------------------------------------------------------------------
_SCENARIOS = {
    # A. 보이스피싱 → 원격제어 → 계좌 탈취
    "voice_phishing_takeover": [
        "incoming_call_long", "open_link_sms", "install_external_app",
        "grant_accessibility_perm", "login_new_device", "reissue_cert",
        "change_password", "disable_security_alert", "increase_transfer_limit",
        "add_new_payee", "night_activity",
    ],
    # B. 스미싱/피싱 → 인증정보 탈취 → 다발 이체
    "smishing_credential_theft": [
        "open_link_sms", "login_unusual_location", "receive_otp", "input_otp",
        "receive_otp", "input_otp", "change_phone_number", "add_new_payee",
        "increase_transfer_limit",
    ],
    # C. 투자리딩방 → 가상자산/대출 송금
    "investment_fraud": [
        "open_link_messenger", "view_crypto", "view_crypto", "view_crypto",
        "increase_transfer_limit", "apply_quick_loan", "add_new_payee",
    ],
    # D. 메신저 지인사칭
    "messenger_impersonation": [
        "open_link_messenger", "incoming_call_long", "add_new_payee",
        "transfer_known_payee_medium", "login_new_device",
    ],
}
_SCENARIO_NAMES = list(_SCENARIOS.keys())

# 시계열 시간 간격(시간 단위) 버킷 경계 → 8개 버킷(0..7)
_TIME_BUCKET_EDGES = np.array([1, 3, 6, 12, 24, 48, 96], dtype=np.float64)  # hours


def _bucketize_gaps(gaps_hours: np.ndarray) -> np.ndarray:
    """이벤트 간 간격(시간)을 0~7 버킷으로 변환."""
    return np.digitize(gaps_hours, _TIME_BUCKET_EDGES).astype(np.int64)


def _gen_background_times(rng, rate_per_day, n_days):
    """관찰 윈도우 [0, n_days*24h) 동안의 배경 이벤트 발생 '시각(시간단위)' 생성."""
    n = rng.poisson(rate_per_day * n_days)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    times = rng.uniform(0.0, n_days * 24.0, size=n)
    # 낮 시간대 편향(사람은 주로 깨어있을 때 활동) — 약한 편향만 적용
    hours_in_day = times % 24.0
    keep = rng.uniform(size=n) < (0.35 + 0.65 * (np.sin((hours_in_day - 3) / 24 * np.pi) ** 2))
    return np.sort(times[keep])


def generate(cfg: DataConfig):
    """전체 사용자 시퀀스를 생성하여 npz/meta로 저장하고 배열을 반환."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    bg_ids, bg_probs = _background_distribution()
    obs_hours = cfg.observation_days * 24.0

    # scam 사용자 인덱스 무작위 선정
    scam_mask = np.zeros(cfg.n_users, dtype=bool)
    scam_idx = rng.choice(cfg.n_users, size=cfg.n_scam, replace=False)
    scam_mask[scam_idx] = True

    # 개인별 활동량(이벤트/일): 로그정규로 이질성 부여
    rates = rng.lognormal(mean=np.log(cfg.base_events_per_day_mean), sigma=0.5, size=cfg.n_users)
    rates = np.clip(rates, 0.5, 20.0)

    max_len = cfg.max_seq_len
    token_ids = np.full((cfg.n_users, max_len), vocab.PAD_ID, dtype=np.int16)
    time_buckets = np.zeros((cfg.n_users, max_len), dtype=np.int8)
    lengths = np.zeros(cfg.n_users, dtype=np.int16)
    labels = np.zeros(cfg.n_users, dtype=np.int8)
    scenarios = np.full(cfg.n_users, -1, dtype=np.int8)  # -1: 정상

    for u in range(cfg.n_users):
        # 1) 배경(일상) 이벤트
        times = _gen_background_times(rng, rates[u], cfg.observation_days)
        if times.size > 0:
            ev_ids = rng.choice(bg_ids, size=times.size, p=bg_probs)
        else:
            ev_ids = np.empty(0, dtype=np.int64)

        # 2) scam 사용자: cutoff 직전 1~3일에 시나리오 위험 군집 주입
        if scam_mask[u]:
            sc_name = _SCENARIO_NAMES[rng.integers(len(_SCENARIO_NAMES))]
            scenarios[u] = _SCENARIO_NAMES.index(sc_name)
            pattern = _SCENARIOS[sc_name]
            esc_window_h = float(rng.uniform(12.0, 72.0))      # 12~72시간 전부터 군집
            esc_start = obs_hours - esc_window_h
            # 시나리오 이벤트를 순서대로, 약간의 지터를 주어 cutoff 직전에 배치
            k = len(pattern)
            offs = np.sort(rng.uniform(0, esc_window_h, size=k))
            inj_times = esc_start + offs
            inj_ids = np.array([vocab.token2id[t] for t in pattern], dtype=np.int64)
            # 군집을 강화하기 위해 일부 위험 이벤트를 반복(밀도↑)
            rep = rng.integers(0, 3)
            if rep > 0:
                extra_t = esc_start + np.sort(rng.uniform(0, esc_window_h, size=rep))
                extra_e = rng.choice(inj_ids, size=rep)
                inj_times = np.concatenate([inj_times, extra_t])
                inj_ids = np.concatenate([inj_ids, extra_e])
            times = np.concatenate([times, inj_times])
            ev_ids = np.concatenate([ev_ids, inj_ids])
            order = np.argsort(times)
            times, ev_ids = times[order], ev_ids[order]

        # 3) 너무 짧은 시퀀스 보정(최소 길이 확보)
        if ev_ids.size < cfg.min_seq_len:
            need = cfg.min_seq_len - ev_ids.size
            pad_t = np.sort(rng.uniform(0, obs_hours, size=need))
            pad_e = rng.choice(bg_ids, size=need, p=bg_probs)
            times = np.concatenate([times, pad_t])
            ev_ids = np.concatenate([ev_ids, pad_e])
            order = np.argsort(times)
            times, ev_ids = times[order], ev_ids[order]

        # 4) 최근 max_len개만 사용(시간 절단)
        if ev_ids.size > max_len:
            times = times[-max_len:]
            ev_ids = ev_ids[-max_len:]

        # 5) 시간 간격 버킷 계산
        gaps = np.diff(times, prepend=times[0])
        buckets = _bucketize_gaps(gaps)

        L = ev_ids.size
        token_ids[u, :L] = ev_ids.astype(np.int16)
        time_buckets[u, :L] = buckets.astype(np.int8)
        lengths[u] = L
        labels[u] = 1 if scam_mask[u] else 0

        if (u + 1) % 20000 == 0:
            print(f"  generated {u + 1:,}/{cfg.n_users:,} users")

    out_path = os.path.join(cfg.out_dir, "sequences.npz")
    np.savez_compressed(
        out_path,
        token_ids=token_ids,
        time_buckets=time_buckets,
        lengths=lengths,
        labels=labels,
        scenarios=scenarios,
    )

    meta = {
        "config": asdict(cfg),
        "n_users": int(cfg.n_users),
        "n_pos": int(labels.sum()),
        "pos_rate": float(labels.mean()),
        "avg_len": float(lengths.mean()),
        "max_len": int(lengths.max()),
        "min_len": int(lengths.min()),
        "vocab_size": vocab.VOCAB_SIZE,
        "scenario_names": _SCENARIO_NAMES,
    }
    with open(os.path.join(cfg.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[data] saved {out_path}")
    print(f"[data] users={meta['n_users']:,} pos={meta['n_pos']:,} "
          f"pos_rate={meta['pos_rate']:.4f} avg_len={meta['avg_len']:.1f}")
    return out_path


if __name__ == "__main__":
    generate(DataConfig())
