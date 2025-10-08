# 한국어 대화 요약 대회 진행 상황 보고서

**작성일시**: 2025-10-04 14:43 KST
**프로젝트**: Korean Dialogue Summarization Competition
**현재 단계**: LLM Fine-tuning with QLoRA

---

## 📊 Executive Summary

### 프로젝트 개요
- **과제**: 한국어 다자간 대화 데이터로부터 요약문 생성
- **평가 지표**: ROUGE-1, ROUGE-2, ROUGE-L F1 점수 평균 (한국어 형태소 토크나이저 사용)
- **현재 방법론**: QLoRA 4bit 양자화를 활용한 대형 언어 모델 파인튜닝

### 진행 현황
- ✅ **완료**: 1/5 모델 (koBART)
- 🔄 **진행 중**: 1/5 모델 (Llama-3.2-Korean-3B)
- ⏳ **대기 중**: 3/5 모델 (Qwen3-4B, Qwen2.5-7B, Llama-3-Korean-8B)

### 전체 진행률
**20%** (1/5 모델 완료)

---

## ✅ Completed Work

### 1. koBART Fine-tuning
**상태**: ✅ Complete
**완료 시간**: 2025-10-04 13:30 KST
**소요 시간**: ~30분

#### 성능 지표
| Metric | Score |
|--------|-------|
| **ROUGE-1 F1** | 56.20% |
| **ROUGE-2 F1** | 24.35% |
| **ROUGE-L F1** | 13.96% |
| **ROUGE SUM** | **94.51** |

#### 모델 구성
- **베이스 모델**: gogamza/kobart-base-v2
- **파라미터 수**: ~123M
- **양자화**: 없음 (Full precision)
- **학습 설정**:
  - Epochs: 20
  - Batch size: 8
  - Learning rate: 5e-5
  - Optimizer: AdamW

#### 산출물
- 모델 체크포인트: `/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/dialogue-summarization/training/kobart-baseline-finetuned/`
- 예측 결과: `baseline_kobart_predictions.csv`
- W&B 로그: [링크 확인 필요]

---

### 2. Critical Bug Fixes

#### 2.1 Metric Configuration Error 수정
**문제**: `metric_for_best_model` 설정 오류로 인한 학습 중단
```python
# Before (잘못된 설정)
metric_for_best_model="rouge-1"

# After (올바른 설정)
metric_for_best_model="rouge1"
```

#### 2.2 Chat Template Tokens 추가
**문제**: Llama/Qwen 모델의 chat template 토큰 누락
```python
# 추가된 special tokens
special_tokens = {
    'additional_special_tokens': [
        '<|begin_of_text|>',
        '<|end_of_text|>',
        '<|start_header_id|>',
        '<|end_header_id|>',
        '<|eot_id|>',
        # ... PII 및 Person 토큰들
    ]
}
```

#### 2.3 QLoRA Compute Dtype 정렬
**문제**: 모델 dtype(bfloat16)과 compute dtype(float16) 불일치
```python
# Before
bnb_config = BitsAndBytesConfig(
    bnb_4bit_compute_dtype=torch.float16
)

# After
bnb_config = BitsAndBytesConfig(
    bnb_4bit_compute_dtype=torch.bfloat16  # 모델 dtype과 일치
)
```

---

## 🔄 Current Activities

### Llama-3.2-Korean-3B Fine-tuning (진행 중)

**시작 시간**: 2025-10-04 14:22 KST (재시작)
**현재 진행률**: Step 127/390 (33%)
**처리 속도**: ~8.7초/step
**예상 완료 시간**: ~40분 남음 (약 15:02 KST)

#### 실시간 학습 메트릭
- **현재 Step**: 127/390
- **Loss**: 모니터링 중
- **학습률**: 2e-4 (constant scheduler)

#### W&B 모니터링
- **Run URL**: https://wandb.ai/bkan-ai/dialogue-summarization-finetuning/runs/hlpzuzzs
- **실시간 추적**: Loss, Learning Rate, GPU Utilization

#### 모델 구성
- **베이스 모델**: beomi/Llama-3.2-Korean-3B-Instruct
- **파라미터 수**: ~3.21B
- **양자화**: QLoRA 4bit (NF4)
- **LoRA 설정**:
  - Rank: 16
  - Alpha: 32
  - Target modules: 모든 Linear 레이어
  - Dropout: 0.05

#### 학습 설정
```yaml
Training Arguments:
  learning_rate: 2e-4
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
  effective_batch_size: 8
  num_train_epochs: 10
  warmup_ratio: 0.03
  lr_scheduler_type: cosine

Generation Config:
  max_new_tokens: 512
  do_sample: True
  temperature: 0.7
  top_p: 0.9
```

---

## ⏳ Pending Tasks

### 1. Qwen3-4B-Instruct Fine-tuning
**상태**: ⏳ Pending
**예상 소요 시간**: ~40-50분
**우선순위**: High

#### 계획된 구성
- **베이스 모델**: Qwen/Qwen3-4B-Instruct-2507
- **파라미터 수**: ~4.02B
- **LoRA Rank**: 16
- **Batch Size**: 4
- **Learning Rate**: 2e-4

---

### 2. Qwen2.5-7B-Instruct Fine-tuning
**상태**: ⏳ Pending
**예상 소요 시간**: ~60-80분
**우선순위**: Medium

#### 계획된 구성
- **베이스 모델**: Qwen/Qwen2.5-7B-Instruct
- **파라미터 수**: ~7.61B
- **LoRA Rank**: 16
- **Batch Size**: 2 (메모리 제약)
- **Learning Rate**: 1e-4

---

### 3. Llama-3-Korean-8B Fine-tuning
**상태**: ⏳ Pending
**예상 소요 시간**: ~80-100분
**우선순위**: High

#### 계획된 구성
- **베이스 모델**: beomi/Llama-3-Open-Ko-8B-Instruct-preview
- **파라미터 수**: ~8B
- **LoRA Rank**: 16
- **Batch Size**: 1 (메모리 제약)
- **Learning Rate**: 1e-4

---

## 🛠 Technical Configuration

### Hardware Setup
```yaml
GPU: NVIDIA RTX 3090
VRAM: 24GB
CUDA: 12.1
PyTorch: 2.5.1
Precision: TF32 (최적화 활성화)
```

### QLoRA Configuration
```python
BitsAndBytesConfig:
  load_in_4bit: True
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: torch.bfloat16
  bnb_4bit_use_double_quant: True

LoRAConfig:
  r: 16
  lora_alpha: 32
  target_modules: "all-linear"
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"
```

### Optimization Features
- ✅ Gradient Checkpointing (메모리 절약)
- ✅ TF32 Precision (RTX 3090 최적화)
- ✅ Mixed Precision Training (BF16)
- ✅ Flash Attention 2 (일부 모델)
- ✅ Gradient Accumulation (효과적 배치 크기 증가)

---

## 📈 Performance Tracking

### 모델 비교표

| Model | Status | Params | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE SUM | Training Time |
|-------|--------|--------|---------|---------|---------|-----------|---------------|
| **koBART** | ✅ Complete | 123M | 56.20% | 24.35% | 13.96% | **94.51** | ~30분 |
| **Llama-3.2-Korean-3B** | 🔄 Training | 3.2B | - | - | - | - | ~40분 (33% 완료) |
| **Qwen3-4B-Instruct** | ⏳ Pending | 4.0B | - | - | - | - | ~40분 (예상) |
| **Qwen2.5-7B-Instruct** | ⏳ Pending | 7.6B | - | - | - | - | ~75분 (예상) |
| **Llama-3-Korean-8B** | ⏳ Pending | 8.0B | - | - | - | - | ~90분 (예상) |

### 기대 성능
- **Baseline (koBART)**: 94.51 ROUGE SUM
- **Target (LLM)**: 100+ ROUGE SUM
- **Competition Benchmark**: Random reference selection ≈ 70점

---

## ⚠️ Risk Management

### 1. Disk Space Monitoring
```bash
현재 사용량: 110GB / 150GB
가용 공간: 40GB
안전 버퍼: 충분 ✅
```

**모니터링 전략**:
- 학습 전 디스크 사용량 체크
- 체크포인트 저장 시 공간 확인
- 불필요한 중간 체크포인트 삭제

### 2. Memory Management
**RTX 3090 24GB VRAM**:
- 3B 모델: Batch size 4 (안전)
- 7B 모델: Batch size 2 (주의)
- 8B 모델: Batch size 1 (한계)

**대응 전략**:
- Gradient accumulation 활용
- Gradient checkpointing 활성화
- 필요시 max_length 축소

### 3. Training Stability
**잠재적 이슈**:
- OOM (Out of Memory) 에러
- Gradient explosion
- Learning rate 부적합

**대응 방안**:
- Automatic mixed precision
- Gradient clipping (max_grad_norm=1.0)
- Warmup ratio 0.03 적용

### 4. GitHub Backup
**동기화 전략**:
- 각 모델 학습 완료 후 커밋
- 주요 버그 수정 시 즉시 푸시
- 실험 결과 CSV 파일 백업

---

## 📅 Timeline

### 완료된 작업
- ✅ **2025-10-03**: 프로젝트 초기 설정 및 데이터 준비
- ✅ **2025-10-04 10:00**: koBART baseline 학습 시작
- ✅ **2025-10-04 13:30**: koBART 학습 완료 (94.51 ROUGE SUM)
- ✅ **2025-10-04 14:00**: QLoRA 버그 수정 완료 (compute dtype alignment)
- ✅ **2025-10-04 14:22**: Llama-3.2-Korean-3B 학습 재시작

### 진행 중
- 🔄 **2025-10-04 14:22-15:02**: Llama-3.2-Korean-3B 학습 중 (33% 완료)

### 예정된 작업
- ⏳ **2025-10-04 15:05**: Qwen3-4B-Instruct 학습 시작
- ⏳ **2025-10-04 15:45**: Qwen2.5-7B-Instruct 학습 시작
- ⏳ **2025-10-04 17:00**: Llama-3-Korean-8B 학습 시작
- ⏳ **2025-10-04 18:30**: 전체 모델 평가 및 비교 분석
- ⏳ **2025-10-04 19:00**: 최종 제출 모델 선정

### 전체 타임라인
```
10:00 ━━━━━━━━━━ koBART (30분) ━━━━━━━━━━━━━► 13:30 ✅
14:22 ━━━━━━━━ Llama-3.2-3B (40분) ━━━━━━━━━► 15:02 🔄 33%
15:05 ━━━━━━━━ Qwen3-4B (40분) ━━━━━━━━━━━━► 15:45 ⏳
15:45 ━━━━━━━━ Qwen2.5-7B (75분) ━━━━━━━━━━► 17:00 ⏳
17:00 ━━━━━━━━ Llama-3-KO-8B (90분) ━━━━━━━► 18:30 ⏳
18:30 ━━━━━━━━ 평가 및 분석 (30분) ━━━━━━━━━► 19:00 ⏳
```

**예상 전체 완료 시간**: 2025-10-04 19:00 KST

---

## 🎯 Next Steps

### 즉시 수행 (0-1시간)
1. **Llama-3.2-Korean-3B 학습 모니터링**
   - W&B 로그 실시간 확인
   - Loss 수렴 여부 관찰
   - 완료 시 평가 수행

2. **Qwen2.5-3B 학습 준비**
   - 디스크 공간 재확인
   - 학습 스크립트 최종 점검
   - W&B run 설정

### 단기 목표 (1-4시간)
3. **남은 3개 모델 순차 학습**
   - Qwen2.5-3B → Qwen2.5-7B → Llama-3-KO-8B
   - 각 모델별 최적 하이퍼파라미터 적용
   - 학습 중 메트릭 모니터링

4. **모델 평가 및 비교**
   - 모든 모델의 dev set 성능 측정
   - ROUGE 점수 비교 분석
   - 생성 품질 정성 평가

### 중기 목표 (4-8시간)
5. **앙상블 전략 수립**
   - 최고 성능 2-3개 모델 선정
   - 앙상블 방법론 검토
   - 테스트 데이터 예측 생성

6. **제출 준비**
   - submission.csv 생성
   - 형식 검증
   - 최종 제출 전 체크리스트 확인

### 장기 목표 (1-2일)
7. **추가 최적화**
   - 하이퍼파라미터 미세 조정
   - Prompt engineering 개선
   - Inference 파라미터 튜닝 (temperature, top_p)

8. **문서화 및 백업**
   - 최종 실험 결과 정리
   - GitHub 저장소 동기화
   - 재현 가능성 확보

---

## 📝 Lessons Learned

### 기술적 교훈
1. **Chat Template 토큰 중요성**
   - Instruction-tuned 모델은 chat template 토큰이 필수
   - 누락 시 학습 불안정 및 성능 저하

2. **Dtype 일관성**
   - 모델 dtype과 compute dtype 불일치 시 성능 저하
   - BF16 지원 하드웨어에서는 BF16 통일 권장

3. **Metric Configuration**
   - Hugging Face Trainer의 metric 이름 규칙 준수 필요
   - `rouge-1` ❌ → `rouge1` ✅

### 프로세스 개선
1. **체계적인 버그 추적**
   - 에러 발생 시 즉시 문서화
   - 해결 방법 공유 및 재사용

2. **실험 로깅**
   - W&B 활용으로 모든 실험 추적
   - 재현 가능한 설정 관리

3. **리소스 모니터링**
   - 디스크/메모리 사용량 지속 확인
   - 사전 예방적 관리

---

## 🔗 Resources

### 코드 및 데이터
- **프로젝트 루트**: `/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/dialogue-summarization/`
- **학습 스크립트**: `scripts/llm_finetuning/`
- **모델 체크포인트**: `training/`
- **예측 결과**: `predictions/`

### 외부 링크
- **W&B Dashboard**: https://wandb.ai/bkan-ai/dialogue-summarization-finetuning
- **GitHub Repo**: https://github.com/Bkankim/naturallanguageprocessingcompetition-nlp-1.git
- **Competition Platform**: [대회 링크]

### 문서
- **CLAUDE.md**: 프로젝트 가이드라인
- **README.md**: 프로젝트 개요
- **docs/**: 대회 규칙 및 기술 문서

---

## 📞 Contact & Support

**프로젝트 담당**: Claude Code Agent
**작성일**: 2025-10-04 14:43 KST
**최종 업데이트**: 진행 중 (실시간 업데이트)

---

**Note**: 이 보고서는 Llama-3.2-Korean-3B 학습이 완료되면 업데이트될 예정입니다. 최신 정보는 W&B 대시보드를 참조하세요.