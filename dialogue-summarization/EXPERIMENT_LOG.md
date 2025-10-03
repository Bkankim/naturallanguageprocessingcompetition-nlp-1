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

### Experiment #3: Large LLM Screening (4bit QLoRA + TF32)

**날짜**: 2025-10-03
**목적**: 대형 한국어 LLM의 Zero-shot 요약 성능 비교 (Decoder-only models)
**방법**: 4bit 양자화 + TF32 최적화로 Zero-shot 평가

#### 테스트 모델 (4개)
1. `Qwen/Qwen2.5-7B-Instruct` - Qwen 7B 모델
2. `MLP-KTLim/llama-3-Korean-Bllossom-8B` - Llama 3 Korean 8B
3. `Bllossom/llama-3.2-Korean-Bllossom-3B` - Llama 3.2 Korean 3B
4. `upstage/SOLAR-10.7B-Instruct-v1.0` - SOLAR 10.7B

#### 스크리닝 결과 (Dev Set, Zero-shot)

| 모델 | 파라미터 | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE Sum | 추론 시간 | 상태 |
|------|---------|---------|---------|---------|-----------|----------|------|
| **Llama-3.2-Korean-3B** 🥇 | **3.21B** | **1.73** | **0.11** | **1.71** | **3.56** | **10분** | ✅ Success |
| Llama-3-Korean-8B | 8.03B | 1.16 | 0.03 | 1.14 | 2.33 | 25분 | ✅ Success |
| Qwen2.5-7B | 7.61B | 0.61 | 0.05 | 0.61 | 1.27 | 33분 | ✅ Success |
| SOLAR-10.7B | 10.73B | 0.00 | 0.00 | 0.00 | 0.00 | 11시간 | ❌ Failed (빈 요약 생성) |

#### 설정
```yaml
qlora:
  load_in_4bit: true
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true

inference:
  batch_size: 16
  num_beams: 4
  no_repeat_ngram_size: 2
  early_stopping: true
  generate_max_length: 100

optimization:
  tf32: true  # RTX 3090 Ampere 최적화
  flash_attention_2: false  # 짧은 시퀀스(~500토큰)에서 효과 미미
```

#### 분석
- ✅ **압도적 승자**: `Llama-3.2-Korean-3B`
  - **가장 작은 모델**인데 **가장 높은 성능** (ROUGE Sum 3.56)
  - 2위(Llama-8B) 대비 **53% 높은 성능**
  - **가장 빠른 추론** (10분, 배치당 20초)
  - 모델 크기 ≠ 성능 (3B > 8B > 7B > 10.7B)

- ⚠️ **SOLAR-10.7B 완전 실패**:
  - 11시간 22분 소요 (배치당 22분!)
  - 빈 요약 생성 → ROUGE 0.00
  - 에러: "Hypothesis is empty"
  - 원인 추정: 프롬프트 형식 불일치 또는 긴 추론 시간으로 인한 문제

- 📊 **Encoder-Decoder vs Decoder-only 비교**:
  - KoT5-sum (Exp #2): ROUGE Sum 22.74 (Zero-shot)
  - Llama-3.2-3B (Exp #3): ROUGE Sum 3.56 (Zero-shot)
  - **Encoder-Decoder 모델이 요약 태스크에 더 적합**
  - Decoder-only는 Fine-tuning 필수

- 💡 **최적화 인사이트**:
  - FlashAttention2: 짧은 시퀀스(512+100토큰)에서 효과 거의 없음 (1.1~1.2배)
  - TF32: RTX 3090에서 안정적인 속도 향상 (1.5~2배 추정)
  - 4bit 양자화: 메모리 효율 극대화, 속도도 개선

#### 다음 단계
- [ ] Llama-3.2-Korean-3B QLoRA Fine-tuning
- [ ] KoT5-sum Fine-tuning (이전 실험에서 Zero-shot 최고)
- [ ] Llama vs KoT5 성능 비교

#### W&B 링크
- https://wandb.ai/bkan-ai/dialogue-summarization-screening/runs/xyhzikrq

#### 결과 파일
- `screening_results/screening_results_20251003_075909.csv`

---

### Experiment #3.1: Fixed LLM Screening (Mecab + Foreign Language Blocking)

**날짜**: 2025-10-03
**목적**: Experiment #3의 문제점 해결 및 재평가
**방법**: Mecab 형태소 토큰화 + bad_words_ids 외국어 차단 적용

#### 문제점 발견 (Experiment #3)
1. ❌ **토큰화 방식 오류**:
   - 문자 기반 토큰화 사용 → ROUGE 점수 **과대평가** (79.21 vs 51.13)
   - 대회 공식 평가: **Mecab 형태소 기반** 토큰화
   - 실제 점수는 약 **65% 낮음**

2. ❌ **외국어 생성 문제**:
   - Zero-shot 요약에 영어/일본어/중국어 혼입
   - 예시: "especialmente", "答答", "农구", "đặc biệt"
   - 한국어 전용 요약 실패

#### 해결 방안 테스트 결과

**테스트 모델**: `Bllossom/llama-3.2-Korean-Bllossom-3B` (Dev Set, 499 samples)

| 방식 | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE Sum | 외국어 차단 | 추론 시간 | 상태 |
|------|---------|---------|---------|-----------|------------|----------|------|
| 기본 (Mecab) | 25.19 | 4.21 | 21.51 | **51.13** | ❌ 다수 포함 | 10분 | ✅ Baseline |
| Enhanced Prompt | 22.42 | 3.17 | 19.51 | 45.09 | ❌ 여전히 포함 | 10분 | ❌ 성능 저하 |
| bad_words_ids (라틴+일본어) | 24.75 | 4.00 | 21.37 | 50.12 | ⚠️ 일부 포함 | 10분 | ⚠️ 미흡 |
| **bad_words_ids + CJK 한자** | **25.19** | **4.21** | **21.51** | **50.92** | **✅ 대부분 차단** | **10분** | **✅ 채택** |
| LogitsProcessor | 24.54 | 3.77 | 20.98 | 49.28 | ❌ 한자 누락 | **58분** | ❌ 너무 느림 |

#### 최종 선택: bad_words_ids + CJK 방식

**구현 내용**:
```python
def generate_bad_words_ids(tokenizer):
    bad_ids = []
    for token_id in range(len(tokenizer)):
        token_str = tokenizer.decode([token_id])
        for ch in token_str:
            code = ord(ch)
            # 라틴 (A-Z, a-z, 확장)
            if 0x41 <= code <= 0x5A or 0x61 <= code <= 0x7A or 0x00C0 <= code <= 0x024F:
                bad_ids.append([token_id])
            # 히라가나/가타카나
            elif 0x3040 <= code <= 0x30FF:
                bad_ids.append([token_id])
            # CJK 통합 한자
            elif 0x4E00 <= code <= 0x9FFF:
                bad_ids.append([token_id])
    return bad_ids  # 약 2644개 토큰 차단
```

#### 분석
- ✅ **Mecab 토큰화 적용**: 대회 공식 평가 방식 준수
- ✅ **외국어 차단 성공**: bad_words_ids + CJK가 가장 효과적
- ✅ **성능 유지**: ROUGE Sum 50.92 (baseline 대비 -0.21, 0.4% 차이)
- ✅ **추론 속도**: 배치당 20초로 빠름 (LogitsProcessor 대비 5.5배)
- ⚠️ **프롬프트 개선 실패**: Enhanced prompt가 오히려 성능 저하 (-6.04)

#### 다음 단계
- [ ] 5개 모델 풀 스크리닝 (Mecab + bad_words_ids + CJK 적용)
  - Qwen/Qwen2.5-7B-Instruct
  - MLP-KTLim/llama-3-Korean-Bllossom-8B
  - Bllossom/llama-3.2-Korean-Bllossom-3B
  - Bllossom/llama-3.2-Korean-Bllossom-AICA-5B
  - upstage/SOLAR-10.7B-Instruct-v1.0
- [ ] 최고 성능 모델 선정 및 QLoRA Fine-tuning

#### 기술적 개선사항
```yaml
# metrics.py - Mecab 토큰화 추가
tokenization_mode: 'mecab'  # 'char' 대신 'mecab' 사용

# model_screening.py - 외국어 차단
inference:
  bad_words_ids: generate_bad_words_ids(tokenizer)  # 2644개 토큰 차단

optimization:
  tf32: true
  qlora_4bit: true
```

---

### Experiment #3.2: Final LLM Screening (5 Models + Qwen3-4B)

**날짜**: 2025-10-04
**모델**: 5개 한국어 LLM (QLoRA 4bit + TF32 최적화)

#### 설정
- Quantization: QLoRA 4bit (NF4)
- Batch Size: 16
- Max Length: 512 (input) / 100 (output)
- Beam Search: 4
- Foreign Token Blocking: bad_words_ids (121k+ tokens)
- Tokenizer: Mecab morphological tokenization for ROUGE

#### Zero-shot 평가 결과 (Dev Set, 499 samples)

| 순위 | 모델 | ROUGE-1 | ROUGE-2 | ROUGE-L | **ROUGE Sum** | 파라미터 |
|------|------|---------|---------|---------|---------------|----------|
| 🥇 | **Llama-3.2-Korean-3B** | 24.72 | 3.73 | 21.07 | **49.52** | 3.21B |
| 🥈 | Llama-3-Korean-8B | 23.95 | 4.01 | 20.65 | **48.61** | 8.03B |
| 🥉 | Qwen2.5-7B | 23.34 | 4.05 | 19.45 | **46.84** | 7.61B |
| 4 | Qwen3-4B-Instruct-2507 | 22.60 | 3.54 | 18.88 | **45.02** | 4.02B |
| 5 | Llama-3.2-Korean-AICA-5B | 21.22 | 2.91 | 17.86 | **41.99** | 4.31B |

**Notes**:
- SOLAR-10.7B aborted (40x slower due to depth upscaling architecture)
- Llama-3.2-3B achieved best zero-shot performance despite smallest size
- AICA-5B underperformed due to task misalignment (trained for conversation generation)

#### 핵심 발견
1. **모델 크기 ≠ 성능**: 3.21B 모델이 8B 모델을 능가
2. **Task Alignment 중요**: Instruction-tuned > Conversation-specialized
3. **Fine-tuning 후보**: Llama-3.2-3B (1순위), Llama-3-8B (2순위)

---

## 🎯 다음 실험 계획

### Experiment #4: KoT5 Fine-tuning (우선순위 1)
- 모델: `psyche/KoT5-summarization`
- Zero-shot 최고 성능 (ROUGE Sum 22.74) - Encoder-Decoder 아키텍처
- 목표: Final Score > 48.0 (+1.15)

### Experiment #5: Llama-3.2-Korean-3B QLoRA Fine-tuning (우선순위 2)
- 모델: `Bllossom/llama-3.2-Korean-Bllossom-3B`
- LLM 중 Zero-shot 최고 (ROUGE Sum 3.56) - Decoder-only 아키텍처
- QLoRA 4bit Fine-tuning 적용
- 목표: Decoder-only 모델의 Fine-tuning 효과 검증

### Experiment #6: Learning Rate Tuning
- learning_rate: 5e-6, 3e-5, 5e-5 시도
- 목표: Fine-tuning 최적화

### Experiment #7: Ensemble Strategy
- KoT5-sum + Llama-3.2-3B 앙상블
- 목표: Encoder-Decoder와 Decoder-only 다양성 확보

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
