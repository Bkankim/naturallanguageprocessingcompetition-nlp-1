# 🧪 실험 기록 (Experiment Log)

> 한국어 대화 요약 모델 실험 결과 추적

## 📊 Baseline 점수

| 지표 | 점수 | 비고 |
|------|------|------|
| **ROUGE-1** | **0.5660** | Baseline |
| **ROUGE-2** | **0.3675** | Baseline |
| **ROUGE-L** | **0.4719** | Baseline |
| **Final Score** | **46.8487** | Baseline |

---

## 📈 실험 결과

### Experiment #1: Baseline (Modular Structure)

**날짜**: 2025-10-01
**모델**: `digit82/kobart-summarization`
**체크포인트**: `checkpoint-1750` (Epoch 7, Best Model)

#### 설정
```yaml
model: digit82/kobart-summarization
num_train_epochs: 20
per_device_train_batch_size: 50
per_device_eval_batch_size: 32
learning_rate: 1.0e-5
warmup_ratio: 0.1
lr_scheduler_type: cosine
max_input_length: 512
max_output_length: 100
```

#### 학습 결과 (Dev Set)
- ROUGE-1: 32.28
- ROUGE-2: 13.46
- ROUGE-L: 30.03
- ROUGE Sum: **75.77**

#### 제출 결과 (Test Set)
| 지표 | 점수 | 변화 |
|------|------|------|
| ROUGE-1 | 0.5660 | **Baseline** |
| ROUGE-2 | 0.3675 | **Baseline** |
| ROUGE-L | 0.4719 | **Baseline** |
| **Final Score** | **46.8487** | **Baseline** |

#### 특징
- ✅ Java/konlpy 의존성 제거 (rouge 라이브러리 사용)
- ✅ 모듈화된 코드 구조
- ✅ Early stopping 적용 (Epoch 7에서 best model)
- ✅ Type hints & Docstrings 완비

#### 개선점
- [ ] Hyperparameter tuning (learning rate, batch size)
- [ ] Longer training (더 많은 epoch)
- [ ] Advanced models (mBART, KoGPT 등)
- [ ] Data augmentation
- [ ] Ensemble 전략

---

## 🎯 다음 실험 계획

### Experiment #2: Learning Rate Tuning (예정)
- learning_rate: 5e-6, 3e-5, 5e-5 시도
- 목표: Final Score > 47.0 (+0.15)

### Experiment #3: Longer Training (예정)
- num_train_epochs: 30
- early_stopping_patience: 5
- 목표: Overfitting 방지하면서 성능 향상

### Experiment #4: Model Upgrade (예정)
- 모델: `gogamza/kobart-base-v2` or `facebook/mbart-large-50`
- 목표: 더 강력한 사전학습 모델 활용

---

## 📝 실험 템플릿

### Experiment #N: [실험명]

**날짜**: YYYY-MM-DD
**모델**: [모델명]
**체크포인트**: [체크포인트 경로]

#### 설정
```yaml
[주요 하이퍼파라미터]
```

#### 학습 결과 (Dev Set)
- ROUGE-1: XX.XX
- ROUGE-2: XX.XX
- ROUGE-L: XX.XX
- ROUGE Sum: XX.XX

#### 제출 결과 (Test Set)
| 지표 | 점수 | 변화 |
|------|------|------|
| ROUGE-1 | X.XXXX | (+/-X.XXXX) |
| ROUGE-2 | X.XXXX | (+/-X.XXXX) |
| ROUGE-L | X.XXXX | (+/-X.XXXX) |
| **Final Score** | **XX.XXXX** | **(+/-X.XXXX)** |

#### 특징
- [변경사항 1]
- [변경사항 2]

#### 분석
- [성능 변화 분석]
- [개선/악화 원인 분석]

---

## 📌 메모

- **최고 점수**: 46.8487 (Experiment #1)
- **최고 ROUGE-1**: 0.5660 (Experiment #1)
- **최고 ROUGE-2**: 0.3675 (Experiment #1)
- **최고 ROUGE-L**: 0.4719 (Experiment #1)
