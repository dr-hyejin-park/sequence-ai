# Scam 예측 Sequence-AI 성능 리포트

- 사전학습 인코더 사용: **예**
- 불균형 대응: pos_weight=3.0, WeightedRandomSampler 오버샘플링
- Train/Valid 분할: stratified 80%/20%

## Train set 성능

- 표본수 n = 80,000 (양성 800, 양성비율 1.000%)
- **ROC-AUC** : 1.0000
- **PR-AUC (Average Precision)** : 1.0000

고정 임계값(threshold=0.50) 기준:
- Precision=0.9604  Recall=1.0000  F1=0.9798
- Confusion Matrix [[TN,FP],[FN,TP]] = [[79167, 33], [0, 800]]

Best-F1 임계값(0.9996) 기준:
- Precision=1.0000  Recall=0.9988  F1=0.9994

운영 관점 Recall@TopK% (상위 위험군 검수 시 적발률):
- 상위 1% (n=800): Recall=0.9988  Precision=0.9988
- 상위 2% (n=1,600): Recall=1.0000  Precision=0.5000
- 상위 5% (n=4,000): Recall=1.0000  Precision=0.2000
- 상위 10% (n=8,000): Recall=1.0000  Precision=0.1000

## Valid set 성능

- 표본수 n = 20,000 (양성 200, 양성비율 1.000%)
- **ROC-AUC** : 1.0000
- **PR-AUC (Average Precision)** : 1.0000

고정 임계값(threshold=0.50) 기준:
- Precision=0.9615  Recall=1.0000  F1=0.9804
- Confusion Matrix [[TN,FP],[FN,TP]] = [[19792, 8], [0, 200]]

Best-F1 임계값(0.9995) 기준:
- Precision=0.9950  Recall=1.0000  F1=0.9975

운영 관점 Recall@TopK% (상위 위험군 검수 시 적발률):
- 상위 1% (n=200): Recall=0.9950  Precision=0.9950
- 상위 2% (n=400): Recall=1.0000  Precision=0.5000
- 상위 5% (n=1,000): Recall=1.0000  Precision=0.2000
- 상위 10% (n=2,000): Recall=1.0000  Precision=0.1000
