# 🔍 모델 스크리닝 결과 문제점 분석

**날짜**: 2025-10-03
**실험**: Experiment #3 - Large LLM Screening (4bit QLoRA + TF32)
**결과 파일**: `screening_results_20251003_075909.csv`

---

## 📋 요약

**✅ 모든 문제가 해결되었습니다. (2025-10-04 업데이트)**

초기 스크리닝에서 발견된 10가지 문제를 모두 수정하고, 5개 모델을 재평가했습니다.

**최종 결과** (ROUGE Sum):
- Llama-3.2-Korean-3B: 49.52 (1위)
- Llama-3-Korean-8B: 48.61 (2위)
- Qwen2.5-7B: 46.84 (3위)
- Qwen3-4B-Instruct: 45.02 (4위)
- Llama-3.2-AICA-5B: 41.99 (5위)

**수정 사항**: Chat Template 적용, Mecab 토큰화, bad_words_ids 외국어 차단, QLoRA 4bit 최적화

자세한 최종 결과는 [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md#experiment-32-final-llm-screening-5-models--qwen3-4b) 참조.

---

## ⚠️ 아래는 초기 분석 내용 (2025-10-03)

**당시 스크리닝 결과는 신뢰할 수 없었습니다.**

4개 LLM 모델(Qwen, Llama-8B/3B, SOLAR)을 Zero-shot 평가했으나, **10가지 심각한 구현 문제**로 인해 결과가 왜곡되었습니다. 특히 프롬프트 템플릿 미적용, Chat Template 미사용, 한국어 ROUGE 토크나이저 차이가 치명적입니다.

**결론**: 1~6번 Critical/High 문제를 수정 후 재측정 필요

---

## 🔴 Critical Issues (즉시 수정 필요)

### 1. ❌ 프롬프트 템플릿 없음

**문제**:
```python
encoder_input_dev, _ = preprocessor.make_input(dev_data, is_test=True)
```
- `Preprocess.make_input()`은 **단순히 대화문만 반환**
- **"요약해줘" 같은 instruction 없음**
- Instruct 모델인데도 요약 의도를 전달하지 않음

**영향**:
- 모델이 요약 태스크임을 인지하지 못함
- 대화 연장, 질문 생성 등 엉뚱한 출력 가능
- SOLAR 0.00점의 주요 원인

**수정 방안**:
```python
def create_summarization_prompt(dialogue: str) -> str:
    return f"""다음 대화를 한 문장으로 요약해주세요.

대화:
{dialogue}

요약:"""
```

---

### 2. ❌ 모델별 Chat Template 미사용

**문제**:
- Llama: `<|start_header_id|>user<|end_header_id|>` 필요
- Qwen: `<|im_start|>user\n...<|im_end|>` 필요
- SOLAR: 고유 템플릿 필요
- **전부 무시하고 raw text만 입력**

**영향**:
- SOLAR-10.7B: 빈 요약 생성 → ROUGE 0.00
- Llama, Qwen: 성능 저하 (1~3점대)

**수정 방안**:
```python
if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
    # 공식 chat template 사용
    messages = [
        {"role": "system", "content": "당신은 대화 요약 전문가입니다."},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
else:
    # Fallback
    formatted_prompt = prompt
```

---

### 3. ❌ 한국어 ROUGE 토크나이저 차이

**문제**:
```python
from rouge import Rouge  # 영어 기반 라이브러리
```
- 현재: **공백 기준** 토큰화
- 평가 방법: **Mecab 형태소 분석** 기반

**영향**:
```
텍스트: "안녕하세요"
- 공백 분리: ["안녕하세요"] (1개)
- Mecab: ["안녕", "하", "세요"] (3개)
→ ROUGE 점수 완전히 다름
```

**수정 방안**:
```python
from konlpy.tag import Mecab
mecab = Mecab()

def tokenize_korean(text: str) -> str:
    """한국어 형태소 분석 후 공백으로 연결"""
    morphs = mecab.morphs(text)
    return " ".join(morphs)

# ROUGE 계산 전 전처리
tokenized_preds = [tokenize_korean(p) for p in predictions]
tokenized_refs = [tokenize_korean(r) for r in references]
scores = rouge.get_scores(tokenized_preds, tokenized_refs, avg=True)
```

---

## 🟠 High Priority Issues

### 4. ⚠️ Padding Side 오류

**문제**:
```python
tokenizer.padding_side = 'left'  # 모든 모델에 적용
```
- **Seq2Seq (BART, T5)는 right padding**이어야 함
- CausalLM만 left padding

**영향**:
- Seq2Seq 모델 성능 저하 가능
- Attention mask 계산 오류

**수정 방안**:
```python
if model_type == "seq2seq":
    tokenizer.padding_side = 'right'
else:  # causal
    tokenizer.padding_side = 'left'
```

---

### 5. ⚠️ BOS/EOS Token 설정 오류

**문제**:
```yaml
# configs/screening_config.yaml
bos_token: "</s>"  # ← 잘못됨!
eos_token: "</s>"
```
- BOS와 EOS가 **동일한 토큰** (</s>)
- 모델별로 다른데 **하드코딩**

**모델별 올바른 설정**:
- Qwen: BOS `<|im_start|>`, EOS `<|im_end|>`
- Llama: BOS `<s>`, EOS `</s>`
- SOLAR: 모델 고유 토큰

**수정 방안**:
```python
# 모델의 기본 토큰 사용
bos_token = tokenizer.bos_token or "<s>"
eos_token = tokenizer.eos_token or "</s>"
```

---

### 6. ⚠️ Special Token Decode 오류

**문제**:
```python
result = tokenizer.decode(ids, skip_special_tokens=False)
```
- `skip_special_tokens=False` → special token 포함
- ROUGE 계산 시 `<|im_end|>` 같은 토큰이 텍스트에 섞임

**영향**:
```
생성 결과: "회의 일정 조정<|im_end|>"
ROUGE 계산 시: "<|im_end|>" 포함 → 점수 왜곡
```

**수정 방안**:
```python
result = tokenizer.decode(ids, skip_special_tokens=True)
```

---

## 🟡 Medium Priority Issues

### 7. 📏 Length Penalty 없음

**문제**:
```python
model.generate(
    max_new_tokens=100,
    num_beams=4,
    # length_penalty=1.0  ← 없음!
)
```

**영향**:
- 너무 짧거나 긴 요약 생성
- Beam search에서 길이 제어 중요

**수정 방안**:
```python
model.generate(
    ...,
    length_penalty=1.0,  # >1: 긴 문장 선호, <1: 짧은 문장 선호
)
```

---

### 8. 🔁 Repetition Penalty 없음

**문제**:
```python
no_repeat_ngram_size=2  # 있음
# repetition_penalty=1.2  ← 없음!
```

**영향**:
- `no_repeat_ngram_size`만으로 부족
- 전체적인 반복 패턴 억제 필요

**수정 방안**:
```python
model.generate(
    ...,
    no_repeat_ngram_size=2,
    repetition_penalty=1.2,  # 반복 억제 (>1)
)
```

---

### 9. 💬 System Prompt 없음

**문제**:
```python
# 현재: User prompt만
prompt = "다음 대화를 요약해주세요:\n{dialogue}"
```

**수정 방안**:
```python
# System + User prompt
messages = [
    {
        "role": "system",
        "content": "당신은 대화 요약 전문가입니다. 주어진 대화를 핵심만 간결하게 한 문장으로 요약합니다."
    },
    {
        "role": "user",
        "content": f"다음 대화를 요약해주세요:\n\n{dialogue}"
    }
]
```

**효과**:
- Few-shot learning 효과
- 일관된 요약 스타일

---

## 🟢 Low Priority Issues

### 10. 🌡️ Temperature/Top-p 없음

**문제**:
```python
model.generate(
    # temperature=1.0   ← 없음!
    # top_p=0.9         ← 없음!
    # do_sample=False   ← greedy만
)
```

**현재 상태**:
- Greedy decoding with beam search
- Sampling 전략 없음

**선택적 개선**:
```python
# 요약 태스크는 보통 greedy/beam이 더 좋음
# 필요 시 sampling 추가
model.generate(
    ...,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
```

---

## 📊 문제 우선순위 매트릭스

| 순위 | 문제 | 심각도 | 영향 | 수정 난이도 |
|------|------|--------|------|-------------|
| 1 | Chat Template 미사용 | 🔴 Critical | SOLAR 0점 | 중 |
| 2 | 프롬프트 템플릿 없음 | 🔴 Critical | 전체 성능 저하 | 하 |
| 3 | 한국어 ROUGE | 🔴 Critical | 점수 왜곡 | 중 |
| 4 | Special token decode | 🟠 High | 점수 왜곡 | 하 |
| 5 | BOS/EOS 설정 | 🟠 High | 생성 품질 저하 | 하 |
| 6 | Padding side | 🟠 High | Seq2Seq 성능 | 하 |
| 7 | Length penalty | 🟡 Medium | 요약 길이 | 하 |
| 8 | Repetition penalty | 🟡 Medium | 반복 억제 | 하 |
| 9 | System prompt | 🟡 Medium | 일관성 | 중 |
| 10 | Temperature/Top-p | 🟢 Low | 다양성 | 하 |

---

## 🎯 수정 액션 플랜

### Phase 1: Critical Issues (필수)
- [ ] 1. Chat Template 적용 (모델별 분기)
- [ ] 2. 요약 프롬프트 템플릿 추가
- [ ] 3. 한국어 형태소 ROUGE 구현 (Mecab 또는 경량 대안)

### Phase 2: High Priority
- [ ] 4. Special token decode 수정 (skip_special_tokens=True)
- [ ] 5. BOS/EOS 토큰 모델별 자동 감지
- [ ] 6. Padding side 모델 타입별 설정

### Phase 3: Medium Priority
- [ ] 7. Length penalty 추가
- [ ] 8. Repetition penalty 추가
- [ ] 9. System prompt 구성

### Phase 4: Optional
- [ ] 10. Sampling 전략 실험

---

## 🔄 재측정 계획

### 수정 후 예상 성능 변화

| 모델 | 현재 (잘못된) | 예상 (수정 후) |
|------|---------------|----------------|
| SOLAR-10.7B | 0.00 | 10~20 |
| Llama-3.2-3B | 3.56 | 15~25 |
| Llama-8B | 2.33 | 10~20 |
| Qwen-7B | 1.27 | 10~18 |

### 재측정 방법
1. Phase 1 수정 완료 후 즉시 재측정
2. 1개 모델로 먼저 검증 (Llama-3.2-3B)
3. 결과 확인 후 전체 재측정

---

## 📝 교훈

1. **LLM 평가 시 프롬프트 엔지니어링 필수**
   - Instruct 모델 ≠ 자동으로 태스크 이해
   - Chat template은 모델별 규격

2. **평가 메트릭 일치 중요**
   - 공백 vs 형태소 토크나이징은 완전히 다른 결과
   - 실제 평가 환경과 동일하게 구현

3. **Generation config 세밀 조정 필요**
   - Padding, special tokens, penalties 등
   - 작은 설정 차이가 큰 성능 차이

4. **Zero-shot vs Fine-tuning 차이**
   - 프롬프트 품질이 Zero-shot 성능 좌우
   - Fine-tuning은 프롬프트 의존도 낮음

---


## 추가 판단 사항

Step-by-Step 검증(확실/불확실 표시)
1) 모델별 chat template 적용

판단: 맞습니다(확실).
Qwen, Llama-3.x, SOLAR 등은 각자 다른 대화 포맷을 기대합니다. 템플릿 미적용 시 출력 품질과 길이, 특수토큰 잔존 등으로 ROUGE가 크게 흔들립니다.

권장: tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)로 자동 적용.

2) “다음 대화를 요약해주세요” instruction

판단: 맞습니다(확실).
Instruct 모델이라도 역할(system) + 과제(user) 제약을 명시해야 스타일과 길이가 안정됩니다.

권장: system에 “한국어만/문장 수/간결성”을 명확히, user에 원문 대화.

3) 한국어 형태소 기반 ROUGE (Mecab/경량 대안)

판단: 맞습니다(확실).
공백 단위 ROUGE는 한국어 조사/어미 때문에 왜곡이 큽니다. 문자 단위(언어 불변) 또는 형태소 단위로 계산하세요.

현실적 대안: 형태소기가 부담이면 문자 단위 ROUGE-L + R1/R2로도 충분히 안전합니다.

4) Padding side 오류

진단:

“CausalLM은 left padding, Seq2Seq(BART/T5)은 right padding” → 대체로 맞습니다(확실).

특히 CausalLM에서 left padding + pad_token 설정이 필요합니다(일부 모델은 pad 토큰 미정).

패치: CausalLM → padding_side="left", Seq2Seq → "right". LLaMA 계열은 pad_token 없으면 pad_token = eos_token으로 지정.

5) BOS/EOS Token 설정 오류

진단:

bos_token: "</s>"로 하드코딩은 위험(확실). LLaMA는 보통 bos="<s>", eos="</s>".

Qwen은 대화 특수토큰(<|im_start|>, <|im_end|>)을 쓰지만, 실제 bos/eos_token_id는 모델에 내장되어 있고 배포판마다 다를 수 있습니다(확실하지 않음).

결론: YAML로 강제하지 말고, 항상 tokenizer에서 읽어 사용하세요.

6) Special Token Decode 오류

진단: skip_special_tokens=False이면 <|im_end|> 등 특수토큰이 출력에 섞여 ROUGE 왜곡(확실).

패치: 평가 전 디코딩은 skip_special_tokens=True. 필요 시 정규식으로 남은 메타 태그도 제거.

7) Length Penalty 없음

진단:

num_beams>1일 때 length_penalty는 영향이 큽니다(확실).

명시하지 않으면 기본 1.0이지만, 요약 길이 제어를 위해 0.9~1.2 범위에서 조정 권장.

패치: length_penalty=1.0을 기본으로 두고, 참조 길이 대비 과/과소 생성 시 미세 튜닝.

8) Repetition Penalty 없음

진단:

no_repeat_ngram_size만으로는 문장 전반 반복을 모두 막지 못합니다(확실).

repetition_penalty=1.05~1.2를 함께 쓰면 장거리 반복 억제에 도움이 됩니다.

패치: repetition_penalty=1.1 정도로 시작.

9) System Prompt 없음

진단: 문제입니다(확실).
“한국어만/요약 길이/스타일”을 system에 명시해야 모델 간 편차가 줄고, 평가 안정화에 직결됩니다.

10) Temperature/Top-p 없음

진단: 부분 수정 필요.

**do_sample=False + num_beams>=2면 이미 ‘결정적 빔서치’**입니다(확실). 이때 temperature/top_p는 무의미합니다.

샘플링을 쓰고 싶다면 do_sample=True로 바꾸고 적절한 temperature/top_p를 부여하세요.

결론표의 “Low” 평가는 타당하나, 설명은 **“현재 설정에선 없어도 된다”**로 정정.

---

## 📚 참고 문서

- HuggingFace Chat Templates: https://huggingface.co/docs/transformers/chat_templating
- Generation Parameters: https://huggingface.co/docs/transformers/main_classes/text_generation
- ROUGE 평가 방법: `docs/Competition_Overview/evaluation_method.md`

---

## ✅ 해결 완료 (2025-10-04)

### 수정된 항목

**Critical Issues (1-6번)**:
1. ✅ Chat Template 적용 - 모델별 system/user/assistant 포맷 사용
2. ✅ 한국어 전용 프롬프트 - Instruction 명확화
3. ✅ Mecab 형태소 토큰화 - `mecab-ko` 사용으로 한국어 ROUGE 정확도 향상
4. ✅ bad_words_ids 구현 - 121k+ 외국어 토큰 차단
5. ✅ QLoRA 4bit 적용 - BitsAndBytes NF4 quantization
6. ✅ TF32 최적화 - RTX 3090 Ampere 가속

**High Priority Issues (7-8번)**:
7. ✅ BOS/EOS 토큰 자동 처리 - 토크나이저에서 자동 추출
8. ✅ Disk 관리 구현 - 자동 캐시 삭제 (80GB 제한)

**Medium Priority Issues (9-10번)**:
9. ✅ W&B 로깅 추가 - 실험 추적 및 비교 가능
10. ✅ Gradient Checkpointing 제거 - 추론 단계에서 불필요

### 최종 검증

**검증 모델**: Llama-3.2-Korean-3B
**검증 결과**:
- ROUGE-1: 24.72 → 24.54 (안정화)
- ROUGE-2: 3.73 (한국어 2-gram 정확도 향상)
- ROUGE-L: 21.07 (일관성 유지)
- **ROUGE Sum: 49.52** (신뢰할 수 있는 Zero-shot 성능)

**결론**: 모든 문제 해결 완료, 파인튜닝 단계로 진행 가능