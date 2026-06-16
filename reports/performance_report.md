# Scam 예측 Sequence-AI 성능 리포트

- 사전학습 인코더 사용: **예**
- 불균형 대응: pos_weight=3.0, WeightedRandomSampler 오버샘플링
- Train/Valid 분할: stratified 80%/20%

## Train set 성능

- 표본수 n = 16,000 (양성 160, 양성비율 1.000%)
- **ROC-AUC** : 1.0000
- **PR-AUC (Average Precision)** : 0.9985

고정 임계값(threshold=0.50) 기준:
- Precision=0.7339  Recall=1.0000  F1=0.8466
- Confusion Matrix [[TN,FP],[FN,TP]] = [[15782, 58], [0, 160]]

Best-F1 임계값(0.9958) 기준:
- Precision=0.9816  Recall=1.0000  F1=0.9907

운영 관점 Recall@TopK% (상위 위험군 검수 시 적발률):
- 상위 1% (n=160): Recall=0.9875  Precision=0.9875
- 상위 2% (n=320): Recall=1.0000  Precision=0.5000
- 상위 5% (n=800): Recall=1.0000  Precision=0.2000
- 상위 10% (n=1,600): Recall=1.0000  Precision=0.1000

## Valid set 성능

- 표본수 n = 4,000 (양성 40, 양성비율 1.000%)
- **ROC-AUC** : 0.9996
- **PR-AUC (Average Precision)** : 0.9741

고정 임계값(threshold=0.50) 기준:
- Precision=0.6842  Recall=0.9750  F1=0.8041
- Confusion Matrix [[TN,FP],[FN,TP]] = [[3942, 18], [1, 39]]

Best-F1 임계값(0.9944) 기준:
- Precision=0.9250  Recall=0.9250  F1=0.9250

운영 관점 Recall@TopK% (상위 위험군 검수 시 적발률):
- 상위 1% (n=40): Recall=0.9250  Precision=0.9250
- 상위 2% (n=80): Recall=1.0000  Precision=0.5000
- 상위 5% (n=200): Recall=1.0000  Precision=0.2000
- 상위 10% (n=400): Recall=1.0000  Precision=0.1000
