# Scam 예측 Sequence-AI (BERT 기반)

행동 로그를 **시계열 시퀀스**로 표현하고, BERT 스타일 Transformer 인코더를
**MLM으로 사전학습**한 뒤 **scam 이진 분류로 Fine-tuning**하여
"향후 3일 내 scam 발생(Y=1)"을 사전에 예측하는 파이프라인입니다.

> 실제 고객 데이터가 없으므로, 요구사항(10만 명 중 1,000명 scam)에 맞춘
> **합성 행동 이벤트 데이터**를 생성합니다. 정상 사용자도 위험 이벤트를 드물게
>수행하도록 설계해, 단일 행동이 아니라 **행동의 결합·순서·밀도 패턴**을 학습해야
> 풀리는 비자명한 문제로 만들었습니다.

---

## 1. 문제 정의

- **단위**: 사용자 1명 = 시퀀스 1개
- **입력**: cutoff 시점 이전 `observation_days(=21일)` 동안의 행동 이벤트 토큰 시퀀스
  (시간순, 최근 `max_seq_len(=64)`개)
- **라벨**: cutoff 이후 `label_horizon_days(=3일)` 이내 scam 사건 발생 시 `Y=1`, 아니면 `Y=0`
- **누수 방지**: scam 사건(고액 이체·가상자산 송금 등)은 **cutoff 이후**에 발생하며
  입력 시퀀스에 포함하지 않습니다. 모델은 **선행 행동만으로 미래를 예측**합니다.
- **불균형**: 100,000명 중 1,000명(1%)이 Y=1

## 2. 행동 이벤트 설계 (`src/vocab.py`)

크게 두 부류의 이벤트를 정의합니다.

- **일상 행동(저위험, 매우 흔함)**: 앱 실행, 잔액·거래 조회, 소액 카드결제, 교통결제,
  기존 수취인 이체, 저축/주식 조회 등
- **위험 선행 지표(정상에서도 드물게 발생)**: 스미싱/메신저 링크 열람, 외부 앱 설치,
  접근성(원격제어) 권한 부여, 장시간 사칭 통화, 신규 기기/비정상 지역 로그인,
  OTP 다발 수신·입력, 비밀번호 변경·인증서 재발급, 전화번호 변경, 보안알림 해제,
  이체한도 상향, 신규 수취인 등록, 심야 활동, 비대면 즉시대출, 가상자산 조회

scam 사건 자체(라벨 정의용, 입력 미포함): 신규 수취인 고액 이체, 가상자산 고액 구매,
대출 직후 전액 이체, 단시간 다발 이체.

## 3. 데이터 생성 로직 (`src/data_generation.py`)

1. 개인별 활동량(이벤트/일)을 로그정규분포로 샘플링 → 이질적 사용자.
2. 관찰 윈도우 전체에 **배경(일상) 이벤트**를 분포 기반으로 생성(주간 시간대 편향).
3. **Y=1 사용자**는 cutoff 직전 12~72시간에 **시나리오 기반 위험 군집**을 주입:
   - `voice_phishing_takeover` 보이스피싱 → 원격제어 → 계좌 탈취
   - `smishing_credential_theft` 스미싱/피싱 → 인증정보 탈취 → 다발 이체
   - `investment_fraud` 투자리딩방 → 가상자산/대출 송금
   - `messenger_impersonation` 메신저 지인사칭
4. 이벤트 간 시간 간격을 8개 버킷으로 변환해 **시간 임베딩**으로 사용.
5. 최근 `max_seq_len`개로 절단 후 `data/sequences.npz`로 저장.

## 4. 모델 (`src/model.py`)

BERT 스타일 인코더: `토큰 임베딩 + 위치 임베딩 + 시간버킷 임베딩` →
`TransformerEncoder(pre-LN, GELU)`. 기본값은 CPU에서도 학습 가능한 소형 모델
(hidden 128, 3 layers, 4 heads).

- **사전학습 헤드(MLM)**: 15% 토큰 마스킹(80/10/10) 후 원 토큰 복원 (`src/pretrain.py`)
- **분류 헤드**: `[CLS]` 풀링 → 이진 로짓 (`src/finetune.py`)

## 5. 불균형 대응 (`src/finetune.py`)

- `WeightedRandomSampler`로 미니배치 양성 비율을 목표치(25%)까지 상향
- 잔여 불균형만 `pos_weight`로 보정 (오버샘플링과 과도 가중의 **이중 적용 방지**)
- 선택적 Focal Loss 지원
- 평가는 임계값에 둔감한 **ROC-AUC / PR-AUC / Recall@TopK**를 중심으로 보고

## 6. 평가 리포트 (`src/evaluate.py`)

Train/Valid 각각에 대해:
- ROC-AUC, PR-AUC(Average Precision)
- 고정 임계값 및 **Best-F1 임계값**에서의 Precision/Recall/F1, Confusion Matrix
- **Recall@TopK%** (상위 위험군 K%를 검수했을 때의 적발률) — 운영 관점 핵심 지표

산출물: `reports/performance_report.md`, `reports/metrics.json`

---

## 실행 방법

```bash
pip install -r requirements.txt

# 전체(10만 명) 파이프라인: 데이터 생성 → MLM 사전학습 → fine-tuning → 리포트
python run_pipeline.py

# CPU 빠른 검증(2만 명 축소)
python run_pipeline.py --quick

# 데이터 재사용 / 사전학습 효과 비교(랜덤 초기화)
python run_pipeline.py --skip-data
python run_pipeline.py --skip-data --no-pretrain
```

주요 산출물:

| 경로 | 내용 |
|------|------|
| `data/sequences.npz` | 사용자별 토큰 시퀀스/시간버킷/라벨 |
| `artifacts/pretrained_encoder.pt` | MLM 사전학습 인코더 |
| `artifacts/scam_classifier.pt` | fine-tuning된 분류기 |
| `reports/performance_report.md` | Train/Valid 성능 리포트 |

## 프로젝트 구조

```
config.py                 # 모든 하이퍼파라미터
run_pipeline.py           # 엔드투엔드 실행기
src/vocab.py              # 행동 이벤트 어휘
src/data_generation.py    # 합성 시퀀스 생성
src/dataset.py            # MLM / 분류 Dataset (+BERT 마스킹)
src/model.py              # BERT 인코더 + MLM/분류 헤드
src/pretrain.py           # MLM 사전학습
src/finetune.py           # scam 분류 fine-tuning
src/evaluate.py           # 지표 계산 및 리포트
```

## 한계와 확장

- 데이터는 합성이며 시나리오 가정에 의존합니다. 실데이터 적용 시 어휘·분포를 교체하세요.
- 더 큰 모델/긴 시퀀스/추가 epoch로 성능을 끌어올릴 수 있습니다(`config.py`).
- 현실에서는 시점별(rolling) 라벨링, 캘리브레이션, 비용민감 임계값 설정,
  설명가능성(SHAP/attention) 보강을 권장합니다.
