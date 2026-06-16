"""
행동 이벤트 어휘(Vocabulary).

각 사용자의 행동을 '토큰의 시퀀스'로 표현하기 위해, 금융/모바일 앱에서
일상적으로 발생할 수 있는 행동 이벤트를 정의한다. 일부 이벤트는 평상시에도
드물게 발생하지만, scam(특히 보이스피싱/메신저피싱에 의한 계좌 탈취·송금) 직전에
빈도와 결합 패턴이 급격히 달라지는 '선행 지표(leading indicator)' 역할을 한다.

BERT 입력을 위해 특수 토큰([PAD], [CLS], [SEP], [MASK])을 앞쪽에 배치한다.
"""

from __future__ import annotations

# 특수 토큰 (BERT 관례를 따른다)
PAD = "[PAD]"
CLS = "[CLS]"
SEP = "[SEP]"
MASK = "[MASK]"
SPECIAL_TOKENS = [PAD, CLS, SEP, MASK]

# ---------------------------------------------------------------------------
# 일상 행동 이벤트 (정상/비정상 공통으로 등장 가능)
# 카테고리별로 묶되, 모델에는 토큰 문자열만 전달된다.
# ---------------------------------------------------------------------------
EVENT_TOKENS = [
    # --- 인증 / 세션 ---
    "login_success",
    "login_fail",
    "logout",
    "open_app",
    "check_notification",
    "biometric_auth",

    # --- 조회 (저위험, 매우 흔함) ---
    "view_balance",
    "view_transactions",
    "view_statement",
    "view_card_benefit",
    "view_point",
    "search_branch_atm",

    # --- 결제 / 소비 (일상) ---
    "card_payment_small",        # 소액 카드결제(편의점/카페 등)
    "card_payment_medium",
    "payment_qr",
    "payment_bill",              # 공과금/관리비
    "pay_subscription",          # 정기구독 결제
    "transit_payment",           # 교통카드

    # --- 이체 (일상) ---
    "transfer_known_payee_small",   # 기존 수취인 소액 이체
    "transfer_known_payee_medium",
    "transfer_self",                # 본인계좌 이체
    "deposit",
    "withdrawal_atm",

    # --- 저축 / 투자 (일상) ---
    "view_savings",
    "deposit_savings",
    "view_fund",
    "view_stock",
    "trade_stock_small",

    # --- 계정 관리 (저빈도, 일상에서도 가끔) ---
    "update_profile",
    "view_security_center",
    "set_alarm",

    # --- 고객 상호작용 ---
    "open_chatbot",
    "view_event_promo",
    "idle",                      # 비활동 구간 표시(시간 흐름 반영)

    # ---------------------------------------------------------------------
    # 위험 선행 지표 이벤트 (정상에서도 드물게 발생 가능하나
    # scam 직전 윈도우에서 빈도/결합이 급증)
    # ---------------------------------------------------------------------
    "open_link_sms",             # 스미싱 문자 링크 열람
    "open_link_messenger",       # 메신저(사칭) 링크 열람
    "install_external_app",      # 외부 출처 앱 설치(원격제어 등)
    "grant_accessibility_perm",  # 접근성/원격제어 권한 부여
    "incoming_call_long",        # 장시간 통화(사칭 상담)
    "login_new_device",          # 신규 기기 로그인
    "login_unusual_location",    # 비정상 지역/IP 로그인
    "receive_otp",               # OTP 수신
    "input_otp",                 # OTP 입력
    "change_password",           # 비밀번호 변경
    "reissue_cert",              # 공동인증서/금융인증서 재발급
    "change_phone_number",       # 등록 전화번호 변경
    "disable_security_alert",    # 보안 알림 해제
    "increase_transfer_limit",   # 이체 한도 상향
    "add_new_payee",             # 신규 수취인 등록
    "night_activity",            # 심야 시간대 활동(02~05시)
    "apply_quick_loan",          # 비대면 즉시대출 신청
    "view_crypto",               # 가상자산 시세 조회
]

# 사전학습/분류 모두에서 사용하는 최종 토큰 목록(특수토큰이 항상 앞)
ALL_TOKENS = SPECIAL_TOKENS + EVENT_TOKENS

# 토큰 <-> id 매핑
token2id = {tok: i for i, tok in enumerate(ALL_TOKENS)}
id2token = {i: tok for tok, i in token2id.items()}

PAD_ID = token2id[PAD]
CLS_ID = token2id[CLS]
SEP_ID = token2id[SEP]
MASK_ID = token2id[MASK]
VOCAB_SIZE = len(ALL_TOKENS)

# scam 발생 자체를 나타내는 사건(라벨 정의용; 입력 시퀀스에는 포함되지 않는다)
# cutoff 이후 horizon 내 이 사건이 있으면 Y=1.
SCAM_OUTCOME_EVENTS = [
    "transfer_new_payee_large",   # 신규 수취인 대상 고액 이체(대포통장 등)
    "crypto_purchase_large",      # 고액 가상자산 구매/송금
    "loan_then_transfer",         # 대출 직후 전액 이체
    "rapid_multi_transfer",       # 단시간 다발 이체
]


def is_risk_event(tok: str) -> bool:
    """위험 선행 지표 이벤트 여부(분석/검증용 헬퍼)."""
    return tok in {
        "open_link_sms", "open_link_messenger", "install_external_app",
        "grant_accessibility_perm", "incoming_call_long", "login_new_device",
        "login_unusual_location", "receive_otp", "input_otp", "change_password",
        "reissue_cert", "change_phone_number", "disable_security_alert",
        "increase_transfer_limit", "add_new_payee", "night_activity",
        "apply_quick_loan", "view_crypto",
    }


if __name__ == "__main__":
    print(f"VOCAB_SIZE = {VOCAB_SIZE}")
    print(f"special    = {SPECIAL_TOKENS}")
    print(f"#events    = {len(EVENT_TOKENS)}")
    print(f"#risk      = {sum(is_risk_event(t) for t in EVENT_TOKENS)}")
