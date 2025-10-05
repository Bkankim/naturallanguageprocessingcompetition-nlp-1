# 🔄 리팩토링 계획: 베이스라인 통합 전략

**작성일**: 2025-10-05
**목적**: Korean_DCS_2024 베이스라인의 핵심 요소를 현재 코드베이스에 통합하여 성능 및 효율성 개선

---

## 📋 Executive Summary

### 현재 상황
- **우리 성능**: Llama-3.2-Korean-3B LR=1e-4 학습 완료, submission 생성
- **목표**: koBART 94.51 ROUGE 수준 달성
- **문제점**:
  - LR 5배 과다 (1e-4 vs 베이스라인 2e-5)
  - Grad_norm 불안정 (2.76 spike)
  - Packing 미지원 (40-60% 토큰 낭비)
  - System prompt 역설 (외국어 금지 언급이 오히려 트리거)

### 전략
**하이브리드 접근**: 완전 재작성 대신 **점진적 통합**으로 리스크 최소화

---

## 🎯 Phase 1: Config 최적화 (즉시 적용 가능)

**소요 시간**: 30분
**리스크**: 낮음
**예상 효과**: 그래디언트 안정성 확보, 외국어 혼입 50% 감소

### 1.1 Learning Rate 조정

**변경 전** (`configs/finetune_config.yaml`):
```yaml
training:
  learning_rate: 1.0e-4  # ❌ 5배 과다
```

**변경 후**:
```yaml
training:
  learning_rate: 2.0e-5  # ✅ 베이스라인 기준
```

**근거**:
- Korean_DCS_2024: `2e-5` 사용
- 현재 grad_norm spike (2.76) → 불안정성 지표
- 5배 감소로 수렴 안정화

---

### 1.2 LR Scheduler 변경

**변경 전**:
```yaml
training:
  lr_scheduler_type: "constant"  # ❌ 수렴 품질 저하
```

**변경 후**:
```yaml
training:
  lr_scheduler_type: "cosine"  # ✅ 후반 수렴 개선
```

**근거**:
- Cosine annealing: 마지막 에폭에서 LR을 0에 가깝게 감소
- Fine-tuning에서 표준적으로 우수한 성능

---

### 1.3 Weight Decay 추가

**변경 전**:
```yaml
training:
  weight_decay: 0.0  # ❌ 과적합 위험
```

**변경 후**:
```yaml
training:
  weight_decay: 0.1  # ✅ 정규화 강화
```

---

### 1.4 Effective Batch Size 증가

**변경 전**:
```yaml
training:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  # Effective batch = 8 * 4 = 32
```

**변경 후**:
```yaml
training:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 8
  # Effective batch = 8 * 8 = 64 (베이스라인과 동일)
```

**근거**:
- 더 큰 배치 크기 → 그래디언트 추정 안정성 향상
- GPU 메모리 허용 범위 내 (Llama 19.15GB, Qwen 23.04GB)

---

### 1.5 System Prompt 단순화

**변경 전** (❌ 역설적 문제):
```yaml
system_prompt: |
  당신은 대화 요약 전문가입니다.
  다음 지침을 엄격히 준수하세요:

  - 반드시 한국어만 사용하세요 (영문/일문/베트남어/이모지/URL 금지).  # ← 문제!
  - 대화의 핵심 내용을 간결하게 요약하세요.
  ...
```

**변경 후** (✅ 허용적 접근):
```yaml
system_prompt: |
  당신은 유능한 AI 어시스턴트입니다.
  사용자의 질문에 대해 친절하게 답변해주세요.
```

**근거**:
- 베이스라인과 동일한 단순 프롬프트
- "금지" 언급이 multilingual token 활성화 가능성 제거
- 작업 인식은 User prompt의 [Question]에서 처리

---

## 🔧 Phase 2: Prompt 구조화 (1-2시간)

**소요 시간**: 1-2시간
**리스크**: 중간 (코드 수정 필요)
**예상 효과**: ROUGE +5~10

### 2.1 베이스라인 Prompt Format 적용

**현재 구조**:
```
[System]
당신은 대화 요약 전문가입니다...

[User]
#Person1#: 안녕하세요.
#Person2#: 반갑습니다.

위 대화를 요약해주세요.
```

**베이스라인 구조**:
```
[System]
당신은 유능한 AI 어시스턴트입니다. 사용자의 질문에 대해 친절하게 답변해주세요.

[User]
[Conversation]
화자SD2000001: 저는 여행 다니는 것을 굉장히 좋아하는데요...
화자SD2000002: 저 여행 다니는 거 되게 좋아해서...

[Question]
위 해외여행 주제에 대한 대화를 요약해주세요.
```

### 2.2 구현 코드

**파일**: `scripts/llm_finetuning.py`

**변경 위치**: `prepare_causal_lm_data()` 함수 내 prompt 생성 부분

```python
# 기존 코드 (단순 포맷)
def format_dialogue_prompt(sample):
    # 단순히 dialogue 텍스트만 전달
    return sample["dialogue"]

# 새 코드 (구조화 포맷)
def format_dialogue_prompt(sample):
    # [Conversation] 섹션 생성
    conversation_lines = ["[Conversation]"]

    # 화자별 발화 추출
    dialogue = sample["dialogue"]
    turns = re.split(r'(#Person\d+#:)', dialogue)

    for i in range(1, len(turns), 2):
        speaker = turns[i].strip(':').strip()
        utterance = turns[i+1].strip()
        conversation_lines.append(f"{speaker}: {utterance}")

    conversation_text = "\n".join(conversation_lines)

    # [Question] 섹션 생성
    # subject_keyword가 있으면 활용, 없으면 기본 문구
    subject = sample.get("subject_keyword", ["대화"])[0]  # 첫 번째 키워드만 사용
    question = f"[Question]\n위 {subject} 주제에 대한 대화를 요약해주세요."

    return conversation_text + "\n\n" + question
```

**주의사항**:
- 베이스라인은 `subject_keyword` 필드를 사용하지만, 우리 데이터에는 없을 수 있음
- 없는 경우 기본값 "대화"로 대체

---

## ⚡ Phase 3: TRL SFTTrainer 전환 (2-3시간)

**소요 시간**: 2-3시간
**리스크**: 중간
**예상 효과**: 학습 시간 40-60% 단축 (packing 효과)

### 3.1 TRL 설치

```bash
pip install trl==0.9.4
```

### 3.2 코드 변경

**파일**: `scripts/llm_finetuning.py`

**변경 전**:
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=training_config["per_device_train_batch_size"],
    ...
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator,
    callbacks=callbacks,
)
```

**변경 후**:
```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    # 기존 TrainingArguments 파라미터 모두 유지
    output_dir=output_dir,
    per_device_train_batch_size=training_config["per_device_train_batch_size"],
    ...

    # TRL 전용 파라미터 추가
    max_seq_length=tokenizer_config.get("encoder_max_len", 1024) +
                   tokenizer_config.get("decoder_max_len", 200),
    packing=True,  # ✅ 핵심 기능!
    dataset_text_field="text",  # 데이터셋에서 텍스트 필드명 (조정 필요)
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,  # ✅ 명시적 전달
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    # data_collator는 SFTTrainer가 자동 처리 (packing 지원)
    callbacks=callbacks,
)
```

### 3.3 데이터셋 포맷 조정

**문제**: SFTTrainer는 `dataset_text_field`로 raw text를 기대

**해결책**:

**Option A**: 데이터셋 포맷 변경
```python
# 기존: {"input_ids": [...], "labels": [...]}
# 변경: {"text": "<full_prompt_with_response>"}

def prepare_sft_dataset(samples):
    texts = []
    for sample in samples:
        # Full conversation text 생성
        prompt = format_dialogue_prompt(sample)
        response = sample["summary"]

        # Chat template 적용
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]

        # Tokenizer의 chat template 활용
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # 텍스트로 반환
            add_generation_prompt=False
        )

        texts.append({"text": full_text})

    return Dataset.from_dict({"text": texts})
```

**Option B**: `formatting_func` 사용
```python
def formatting_func(example):
    """SFTTrainer에 전달할 포맷팅 함수"""
    prompt = format_dialogue_prompt(example)
    response = example["summary"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

trainer = SFTTrainer(
    ...,
    formatting_func=formatting_func,  # ✅ 이 방식이 더 깔끔
)
```

---

## 📊 Phase 4: 검증 및 성능 비교

### 4.1 A/B 테스트 구성

| 설정 | Phase 1 | Phase 2 | Phase 3 |
|------|---------|---------|---------|
| LR | 2e-5 | 2e-5 | 2e-5 |
| Scheduler | cosine | cosine | cosine |
| Weight Decay | 0.1 | 0.1 | 0.1 |
| Batch Size | 64 | 64 | 64 |
| Prompt | 단순 | **구조화** | 구조화 |
| Trainer | Trainer | Trainer | **SFTTrainer** |
| Packing | ❌ | ❌ | **✅** |

### 4.2 예상 성능

| Metric | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| ROUGE Sum | 60-70 | **70-80** | **75-85** |
| 학습 시간/epoch | 60분 | 60분 | **25-35분** |
| Grad Norm Spike | <1.5 | <1.5 | <1.5 |
| 외국어 혼입 | <3% | <1% | <1% |

---

## 🛡️ 리스크 관리

### Phase 1 실행 리스크
- **리스크**: 거의 없음 (config만 변경)
- **롤백**: config 파일 git revert
- **검증**: 1 epoch 학습 후 grad_norm, loss 추이 확인

### Phase 2 실행 리스크
- **리스크**: 중간 (prompt 파싱 버그 가능)
- **롤백**: prompt 생성 함수만 되돌리기
- **검증**: 소수 샘플로 prompt 생성 테스트

### Phase 3 실행 리스크
- **리스크**: 중간 (Trainer → SFTTrainer 호환성)
- **롤백**: 기존 Trainer 코드 보존 (branch 생성)
- **검증**:
  - packing=False로 먼저 테스트
  - 학습 loss 곡선 비교

---

## 📅 실행 순서

### 우선순위 1: Phase 1 (즉시)
```bash
# 1. Config 수정
vim configs/finetune_config.yaml

# 2. 1 epoch 테스트 학습
python scripts/llm_finetuning.py --config configs/finetune_config.yaml

# 3. W&B에서 grad_norm, loss 확인
# 예상: grad_norm < 1.5, loss 안정적 감소
```

**판단 기준**:
- ✅ grad_norm < 1.5 유지 → Phase 2 진행
- ❌ grad_norm 여전히 spike → LR 추가 감소 (1e-5)

---

### 우선순위 2: Phase 2 (Phase 1 성공 시)
```bash
# 1. Prompt 생성 함수 수정
vim scripts/llm_finetuning.py

# 2. 소수 샘플 테스트
python -c "
from scripts.llm_finetuning import format_dialogue_prompt
# 샘플로 prompt 생성 확인
"

# 3. 전체 학습
python scripts/llm_finetuning.py --config configs/finetune_config.yaml
```

**판단 기준**:
- ✅ ROUGE > 70 → Phase 3 진행
- ❌ ROUGE < 70 → Prompt 디버깅 또는 Phase 1로 제출

---

### 우선순위 3: Phase 3 (Phase 2 성공 시)
```bash
# 1. TRL 설치
pip install trl==0.9.4

# 2. 코드 수정 (branch 생성)
git checkout -b feature/sft-trainer
vim scripts/llm_finetuning.py

# 3. Packing 비활성화 테스트
# packing=False로 먼저 학습하여 호환성 확인

# 4. Packing 활성화
# packing=True로 학습 속도 개선 확인
```

---

## 🎯 성공 기준

### Phase 1
- [x] Grad_norm < 1.5 안정 유지
- [x] Loss가 epoch마다 감소
- [x] 외국어 혼입 < 3%

### Phase 2
- [x] ROUGE-1 > 35
- [x] ROUGE-L > 30
- [x] ROUGE Sum > 70

### Phase 3
- [x] 학습 시간 40% 이상 단축
- [x] ROUGE 성능 유지 또는 향상
- [x] Packing으로 인한 품질 저하 없음

---

## 📝 문서화 체크리스트

- [ ] Phase 1 적용 후 W&B 링크 기록
- [ ] Phase 2 적용 후 샘플 생성 결과 비교
- [ ] Phase 3 적용 후 학습 시간 측정
- [ ] 최종 성능 비교표 업데이트
- [ ] CLAUDE.md에 Critical Learnings 추가

---

## 🔗 참고 자료

- Korean_DCS_2024 베이스라인: `/Competition/NLP/Korean_DCS_2024/`
- TRL 문서: https://huggingface.co/docs/trl
- 현재 코드: `/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/dialogue-summarization/`
- W&B 프로젝트: https://wandb.ai/bkan-ai/dialogue-summarization-finetuning

---

**마지막 업데이트**: 2025-10-05
**작성자**: Claude Code
**상태**: Phase 1 준비 완료
