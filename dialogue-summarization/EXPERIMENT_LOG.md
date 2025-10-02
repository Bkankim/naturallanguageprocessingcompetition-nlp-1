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

### Experiment #2: Model Screening (Zero-shot Evaluation)

**날짜**: 2025-10-02
**목적**: 여러 한국어 요약 모델의 Zero-shot 성능 비교
**방법**: Fine-tuning 없이 사전학습된 모델로 Dev Set 평가

#### 테스트 모델 (4개)
1. `gogamza/kobart-base-v2` - SKT KoBART base model
2. `ainize/kobart-news` - News summarization specialized
3. `psyche/KoT5-summarization` - T5 Korean summarization
4. `digit82/kobart-summarization` - Current baseline model

#### 스크리닝 결과 (Dev Set, Zero-shot)

| 모델 | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE Sum | 상태 |
|------|---------|---------|---------|-----------|------|
| **KoT5-sum** | **10.04** | **3.18** | **9.51** | **22.74** | ✅ Success |
| KoBART-sum | 8.34 | 2.19 | 7.75 | 18.27 | ✅ Success |
| KoBART-base | 3.74 | 0.61 | 3.62 | 7.97 | ✅ Success |
| KoBART-news | 0.00 | 0.00 | 0.00 | 0.00 | ❌ Failed (tokenizer error) |

#### 설정
```yaml
inference:
  batch_size: 16
  num_beams: 4
  no_repeat_ngram_size: 2
  early_stopping: true
  generate_max_length: 100

disk_management:
  auto_cleanup_cache: true  # 모델별 순차 실행 & 캐시 삭제
  max_disk_usage_gb: 80
```

#### 분석
- ✅ **최고 모델**: `psyche/KoT5-summarization` (ROUGE Sum 22.74)
  - T5 기반 모델이 BART 기반보다 Zero-shot 성능 우수
  - Fine-tuning 시 더 큰 성능 향상 기대
- ✅ **Fine-tuning 효과 확인**:
  - KoBART-sum Zero-shot: 18.27 → Fine-tuned: 75.77 (4.1배 향상)
  - Fine-tuning이 필수적임을 재확인
- ⚠️ **모델 선택 전략**:
  - Zero-shot 성능이 높은 모델 = Fine-tuning 성능도 높을 가능성
  - KoT5-sum을 다음 Fine-tuning 후보로 선정

#### 개선점
- [ ] KoT5-sum 모델 Fine-tuning 실험
- [ ] 더 큰 모델 (mBART, KoGPT 등) 스크리닝 (디스크 제약 고려)
- [ ] Ensemble 전략 (KoT5 + KoBART)

---

## 🎯 다음 실험 계획

### Experiment #3: KoT5 Fine-tuning (우선순위 높음)
- 모델: `psyche/KoT5-summarization`
- Zero-shot에서 가장 높은 성능 (ROUGE Sum 22.74)
- 목표: Final Score > 48.0 (+1.15)

### Experiment #4: Learning Rate Tuning
- learning_rate: 5e-6, 3e-5, 5e-5 시도
- 목표: Fine-tuning 최적화

### Experiment #5: Ensemble Strategy
- KoT5-sum + KoBART-sum 앙상블
- 목표: 다양성 확보 및 성능 향상

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
