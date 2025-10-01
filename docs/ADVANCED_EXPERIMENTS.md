# Advanced Model Experiments Guide

고급 모델 실험을 위한 가이드 및 리소스

## 📚 개요

이 문서는 Stage 4 이후 추가적인 성능 향상을 위한 고급 실험 기법들을 설명합니다.

## 🔬 실험 가능한 기법들

### 1. 대규모 언어 모델 활용

#### Qwen2.5 시리즈
```python
# Qwen2.5-7B-Instruct 활용
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Few-shot prompting
prompt = f"""다음 대화를 요약해주세요:

{dialogue}

요약:"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

#### Solar API 활용
```python
import requests
import os

SOLAR_API_KEY = os.getenv("SOLAR_API_KEY")

def generate_summary_with_solar(dialogue):
    url = "https://api.upstage.ai/v1/solar/chat/completions"

    headers = {
        "Authorization": f"Bearer {SOLAR_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "solar-1-mini-chat",
        "messages": [
            {
                "role": "system",
                "content": "당신은 한국어 대화를 요약하는 전문가입니다."
            },
            {
                "role": "user",
                "content": f"다음 대화를 간결하게 요약해주세요:\\n\\n{dialogue}"
            }
        ],
        "temperature": 0.3
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]
```

### 2. Parameter Efficient Fine-Tuning (PEFT)

#### LoRA 설정 최적화
```python
from peft import LoraConfig, get_peft_model, TaskType

# 더 공격적인 LoRA 설정
lora_config = LoraConfig(
    r=16,  # Increased rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # More modules
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(base_model, lora_config)
```

#### QLoRA (4-bit quantization)
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
```

### 3. 앙상블 기법

#### Weighted Voting
```python
def weighted_ensemble(predictions_list, weights):
    \"\"\"가중치 기반 앙상블.\"\"\"
    from collections import Counter

    ensembled = []
    for preds in zip(*predictions_list):
        # Count votes with weights
        vote_counter = Counter()
        for pred, weight in zip(preds, weights):
            vote_counter[pred] += weight

        # Get winner
        winner = vote_counter.most_common(1)[0][0]
        ensembled.append(winner)

    return ensembled

# Example
weights = [0.4, 0.35, 0.25]  # Based on validation ROUGE scores
final_predictions = weighted_ensemble(
    [kobart_preds, mbart_preds, qwen_preds],
    weights
)
```

#### Stacking Ensemble
```python
from sklearn.linear_model import Ridge

def stacking_ensemble(base_predictions, meta_features, val_labels):
    \"\"\"스태킹 앙상블 with meta-learner.\"\"\"

    # Train meta-learner
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_features, val_labels)

    # Predict
    final_predictions = meta_model.predict(test_meta_features)

    return final_predictions
```

### 4. 데이터 증강

#### Back-Translation
```python
from transformers import pipeline

# Korean -> English -> Korean
ko_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")
en_to_ko = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ko")

def back_translate(text):
    en_text = ko_to_en(text)[0]['translation_text']
    ko_text = en_to_ko(en_text)[0]['translation_text']
    return ko_text

augmented_dialogues = [back_translate(d) for d in train_df['dialogue']]
```

#### Paraphrasing
```python
# Use GPT models for paraphrasing
def paraphrase_with_gpt(text):
    prompt = f"다음 문장을 다르게 표현해주세요: {text}"
    # Use Solar or other API
    return generated_paraphrase
```

### 5. Post-processing

#### 길이 제약 적용
```python
def apply_length_constraint(summary, max_length=150):
    \"\"\"요약문 길이 제약.\"\"\"
    if len(summary) > max_length:
        # Truncate at sentence boundary
        sentences = summary.split('.')
        truncated = ''
        for sent in sentences:
            if len(truncated + sent + '.') <= max_length:
                truncated += sent + '.'
            else:
                break
        return truncated.strip()
    return summary
```

#### 반복 제거
```python
import re

def remove_repetitions(text):
    \"\"\"반복 구문 제거.\"\"\"
    # Remove repeated phrases
    words = text.split()
    seen = set()
    filtered = []

    for i, word in enumerate(words):
        # Check 3-gram repetition
        if i >= 2:
            trigram = ' '.join(words[i-2:i+1])
            if trigram in seen:
                continue
            seen.add(trigram)
        filtered.append(word)

    return ' '.join(filtered)
```

## 📊 실험 추적

### WandB Sweep으로 하이퍼파라미터 탐색
```yaml
# sweep_config.yaml
program: notebooks/stage4_advanced_models.ipynb
method: bayes
metric:
  name: rouge_sum
  goal: maximize
parameters:
  lora_r:
    values: [8, 16, 32]
  lora_alpha:
    values: [16, 32, 64]
  learning_rate:
    min: 1e-5
    max: 5e-5
  num_train_epochs:
    values: [3, 5, 7]
```

```bash
# Run sweep
wandb sweep sweep_config.yaml
wandb agent <sweep_id>
```

## 🚀 실행 예시

### Context7을 활용한 모델 문서 검색
```python
# MCP context7 tool 사용 예시
# 특정 모델의 최신 문서를 검색하여 최적 설정 찾기

# 1. Qwen2.5 공식 문서 검색
# 2. LoRA 최적 설정 찾기
# 3. 한국어 성능 향상 기법 조사
```

## 📝 실험 체크리스트

- [ ] Qwen2.5-7B-Instruct 실험
- [ ] Solar API 통합 테스트
- [ ] QLoRA 4-bit 양자화 적용
- [ ] 3-model 앙상블 구성
- [ ] Back-translation 데이터 증강
- [ ] Post-processing 파이프라인 적용
- [ ] WandB Sweep 실행
- [ ] 최종 ROUGE > 80 달성

## 🎯 목표

- **Baseline (KoBART)**: ROUGE ~70
- **Optimized (Stage 3)**: ROUGE ~75
- **Advanced (Stage 4)**: ROUGE ~78-80
- **Target**: ROUGE > 80 ✨

## 📚 참고 자료

- [Qwen2.5 Documentation](https://qwenlm.github.io/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Library](https://github.com/huggingface/peft)
- [Korean NLP Resources](https://github.com/gyunggyung/Korean-NLP)
