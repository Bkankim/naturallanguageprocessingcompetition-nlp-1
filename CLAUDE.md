# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## .md Update Guidelines
- After completing each task, always prioritize updating all documents to ensure they are up to date.

## Git Updates
- After completing each task, update the .gitignore file and always synchronize with the GitHub repository (as a backup practice).

## code agents Guidelines
- 각 명령을 수행할 때, 가능한 SubAgents를 적극 활용할 것 (각각의 임무를 정의해둔 서브 에이전트들이 존재).
**주의** : 단독 수행할 때 성능이 더 좋을 것 같다는 판단이 서면 서브에이전트 사용금지. 어디까지나 너의 판단 하에 병렬 수행을 할 것.

## ⚠️ 프로젝트 구조 및 Git 작업 디렉토리 (절대 헷갈리지 말 것!)

**프로젝트 루트**: `/Competition/NLP`
**Git 작업 디렉토리**: `/Competition/NLP/naturallanguageprocessingcompetition-nlp-1`

### 중요 규칙:
1. **모든 Git 명령은 반드시 `naturallanguageprocessingcompetition-nlp-1/` 디렉토리 안에서 실행**
   ```bash
   # 올바른 예시
   cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1
   git add .
   git commit -m "message"
   git push origin main

   # 잘못된 예시 - 절대 루트에서 실행하지 말 것
   cd /Competition/NLP
   git add .  # ❌ 잘못됨!
   ```

2. **파일 작업은 루트(`/Competition/NLP`)에서 하되, Git 작업은 서브디렉토리에서**
   - Epic/Task 파일: `/Competition/NLP/.claude/`
   - 데이터: `/Competition/NLP/data/`
   - 문서: `/Competition/NLP/docs/`
   - Git 저장소: `/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/.git`

3. **GitHub 저장소**: `https://github.com/Bkankim/naturallanguageprocessingcompetition-nlp-1.git`

4. **작업 흐름**:
   - 프로젝트 파일 편집: 루트 또는 하위 디렉토리 어디서나 가능
   - Git 명령 실행: **반드시** `cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1` 후 실행
   - 커밋할 파일을 Git 저장소 안으로 복사/이동 필요

## MCP Usage Criteria
- sequential-thinking: Use freely when tasks require deep reasoning or involve complex processes.
- context7: Must be used when dependency issues arise or when downloading new packages is necessary.

## Prompt Language
- Always respond in Korean, and represent all reasoning steps in Korean as well.

## Code Writing Guidelines
- Every Class and def definition must be preceded by a simple comment explaining its functionality.

## Jupyter Notebook Guidelines
- **작업 완료 시 재현 노트북 생성 필수**: 모듈화된 코드나 스크립트를 작성한 경우, 사용자가 쉽게 재현할 수 있도록 Jupyter Notebook을 반드시 함께 제공할 것.
- **노트북 구조**:
  1. 환경 설정 (imports, path 설정)
  2. Config 로드 및 주요 설정 확인
  3. 데이터 로드 및 전처리
  4. 모델 학습/추론
  5. 결과 평가 및 시각화
  6. 샘플 예측 확인
- **노트북 위치**: 프로젝트 루트 또는 관련 디렉토리 (예: `dialogue-summarization/train_demo.ipynb`)
- **마크다운 셀 활용**: 각 섹션을 명확히 구분하고 설명 추가
- **실행 가능성 보장**: 모든 셀이 순차적으로 실행 가능해야 함
- **예제**:
  - 학습 프로젝트: `train_demo.ipynb` 또는 `reproduction_notebook.ipynb`
  - 추론 프로젝트: `inference_demo.ipynb`
  - EDA 프로젝트: `eda_notebook.ipynb`

## Storage Management Guidelines
- Before saving any run results, check disk usage with:
```bash
du -sh / 2>/dev/null
```
- **⚠️ CRITICAL**: Ensure that the root (/) disk usage does not exceed the **150GB limit** - server resets if exceeded
- **Safe zone**: Keep usage below 110GB for safety margin
- Use `du -sh /` for accurate measurement (not `df -h`)
- Clean up old checkpoints and cache files regularly

## Project Overview

This is a **Dialogue Summarization Competition** project focused on building models that generate summaries from Korean conversational data. The competition uses the DialogSum dataset (translated to Korean) where models must generate summaries of multi-turn dialogues between 2-7 participants.

**Key Task**: Given dialogue text, generate a concise summary that captures the main points of the conversation.

**Evaluation**: Model performance is measured using the average of ROUGE-1-F1, ROUGE-2-F1, and ROUGE-L-F1 scores across 3 reference summaries per dialogue. Korean morphological tokenization is used for scoring.

## Data Structure

### Dataset Files (`data/`)
- `train.csv` - 12,457 training samples (dialogue + summary)
- `dev.csv` - 499 development samples
- `test.csv` - 250 test samples (dialogue + 3 reference summaries)
- `sample_submission.csv` - Submission format template

### Data Format
- **fname**: Unique dialogue identifier
- **dialogue**: Multi-turn conversation with speakers labeled as `#Person1#`, `#Person2#`, etc.
  - Turns separated by `\n` (note: some noise includes `\\n` or `<br>` tags)
  - Contains masked PII: `#PhoneNumber#`, `#Address#`, `#DateOfBirth#`, `#PassportNumber#`, `#SSN#`, `#CardNumber#`, `#CarNumber#`, `#Email#`
- **summary**: Target summary text (문어체/written style vs dialogue 구어체/spoken style)

### Data Noise
The dataset contains noise that must be handled:
- Escaped newlines: `\\n` instead of `\n`
- HTML tags: `<br>` instead of newlines
- Informal tokens: `ㅋㅋ`, `ㅇㅇ` (Korean onomatopoeia)

## Development Commands

### Package Management
Use the PM CLI for unified package management:
- `/pm:install` - Install all dependencies
- `/pm:add <package>` - Add a new package
- `/pm:run <script>` - Run package scripts
- `/pm:test` - Run tests

### Python Environment
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

### Working with Notebooks
Baseline implementations are in Jupyter notebooks:
- `naturallanguageprocessingcompetition-nlp-1/code/baseline.ipynb` - Main baseline model
- `naturallanguageprocessingcompetition-nlp-1/code/solar_api.ipynb` - Solar API integration

## Competition Rules & Constraints

### Prohibited
- **DialogSum dataset** (original or derived) - this is the source of test data
- Using test data for training (analysis is allowed)
- Pretrained weights trained on DialogSum
- Paid APIs (exception: **Solar model is allowed**)

### Allowed
- External datasets (except DialogSum)
- Free APIs
- Pretrained weights (not trained on DialogSum)
- Solar model API

### Submission
- Maximum 12 submissions per day per team (resets at midnight KST)
- Final submission: select up to 2 results before deadline
- Output format: CSV with 249 dialogue summaries

## Model Development Guidelines

### Preprocessing Considerations
1. **Text Cleaning**: Handle noise (`\\n`, `<br>`, informal tokens)
2. **Special Tokens**: Add PII masking tokens and person markers to tokenizer:
   ```python
   special_tokens = ['#Person1#', '#Person2#', ..., '#PhoneNumber#', '#Address#', ...]
   tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
   ```
3. **Length Analysis**: Check dialogue/summary length distributions to set appropriate `max_length` parameters
4. **Speaker Extraction**: Use regex `r"#\w+\d#"` to identify speakers (`#Person1#`, etc.)

### Key Hyperparameters (Seq2SeqTrainingArguments)
- `learning_rate`: Default 5e-5 (AdamW)
- `per_device_train_batch_size`: Default 8
- `num_train_epochs`: Default 20 (watch for overfitting)
- `gradient_accumulation_steps`: For effective larger batch sizes
- `warmup_ratio`: Learning rate warmup (default 0.0)
- `lr_scheduler_type`: `linear` or `cosine`
- `fp16`/`bf16`: Mixed precision training

### Hyperparameter Tuning
The project supports Optuna for automated hyperparameter search:
```python
def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("batch", [16, 32]),
        # ... other parameters
    }

best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=20
)
```

## Documentation

### Competition Overview (`docs/Competition_Overview/`)
- `introduction.md` - Competition background and objectives
- `data_overview.md` - Detailed data structure and format
- `evaluation_method.md` - ROUGE metric calculation details
- `competition_rule.md` - Full rules and restrictions

### Advanced Topics (`docs/Competition_Advanced/`)
- `textdata_eda.md` - Text EDA techniques: word clouds, TF-IDF, regex patterns, PII masking analysis
- `hyperparameter_tuning_tip.md` - Grid/Random/Bayesian optimization with Optuna

## Expected Baseline Performance

Random reference selection from 3 gold summaries achieves approximately **70 points** (sum of ROUGE-1, ROUGE-2, ROUGE-L F1 scores). This provides a baseline expectation for model performance.

## Architecture Notes

This is primarily a **Seq2Seq summarization task** using Korean language models. Common approaches:
- Fine-tuning encoder-decoder models (e.g., KoBART, mBART)
- Using Hugging Face Trainer with Seq2SeqTrainingArguments
- Implementing custom metrics using Korean morphological tokenizers for ROUGE scoring

## Critical Technical Learnings

### 1. Chat Template Tokens for LLM Models
LLM 모델 파인튜닝 시 반드시 Chat Template 토큰을 명시적으로 토크나이저에 추가해야 함:

**Llama 모델 (Llama-3, Llama-3.1, Llama-3.2 등):**
```python
chat_template_tokens = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
tokenizer.add_special_tokens({'additional_special_tokens': chat_template_tokens})
model.resize_token_embeddings(len(tokenizer))
```

**Qwen 모델 (Qwen2, Qwen2.5 등):**
```python
chat_template_tokens = ["<|im_start|>", "<|im_end|>"]
tokenizer.add_special_tokens({'additional_special_tokens': chat_template_tokens})
model.resize_token_embeddings(len(tokenizer))
```

**주의사항:**
- Chat template 토큰을 추가하지 않으면 학습 중 토큰 인코딩 오류 발생
- 반드시 `model.resize_token_embeddings()` 호출하여 임베딩 크기 조정

### 2. QLoRA Configuration: compute_dtype Matching
QLoRA 설정 시 `compute_dtype`은 반드시 모델의 dtype과 일치해야 함:

**Llama 모델 (bf16 사용):**
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16  # bf16 사용
)

training_args = TrainingArguments(
    bf16=True,  # bf16 활성화
    fp16=False,
    ...
)
```

**Qwen 모델 (fp16 사용):**
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # fp16 사용
)

training_args = TrainingArguments(
    fp16=True,  # fp16 활성화
    bf16=False,
    ...
)
```

**주의사항:**
- compute_dtype 불일치 시 학습 중 수치적 불안정성 또는 오류 발생
- 항상 모델 문서에서 권장 dtype 확인

### 3. metric_for_best_model: Model Type별 차이

**Encoder-Decoder 모델 (KoBART, KoT5 등):**
```python
training_args = Seq2SeqTrainingArguments(
    metric_for_best_model="rouge_sum",  # ROUGE 합계 사용
    greater_is_better=True,
    load_best_model_at_end=True,
    ...
)
```

**Causal LM 모델 (Llama, Qwen 등):**
```python
training_args = TrainingArguments(
    metric_for_best_model="eval_loss",  # Loss 사용
    greater_is_better=False,  # Loss는 낮을수록 좋음
    load_best_model_at_end=True,
    ...
)
```

**이유:**
- Encoder-Decoder: `generate()` 메서드가 기본 제공되므로 ROUGE 직접 계산 가능
- Causal LM: generation config 설정이 복잡하여 eval_loss 사용이 안정적

### 4. Model Architecture 선택 가이드

**Encoder-Decoder (Seq2Seq) 모델:**
- **장점**: 요약 태스크에 최적화, 생성 품질 우수
- **단점**: 모델 크기 제약 (대부분 1B 이하)
- **추천 모델**: KoBART, KoT5, mBART-50

**Causal LM (Decoder-only) 모델:**
- **장점**: 대규모 모델 사용 가능 (3B~8B), 최신 아키텍처
- **단점**: 학습이 까다로움, Chat template 처리 필요
- **추천 모델**: Llama-3.2-3B, Qwen2.5-3B, EXAONE-3.5-2.4B

### 5. Prompt Truncation 문제 (LLM 모델의 치명적 이슈)

Causal LM 파인튜닝 시 max_length 설정이 부적절하면 **assistant 헤더가 잘려나가** 모델이 생성 위치를 인식하지 못하는 치명적 문제 발생.

**문제 발생 메커니즘:**

Chat template 구조:
```
<|start_header_id|>system<|end_header_id|>
당신은 요약 전문가입니다.
<|eot_id|><|start_header_id|>user<|end_header_id|>
다음 대화를 요약하세요:
---
[긴 대화 내용...]
---
<|eot_id|><|start_header_id|>assistant<|end_header_id|>  ← 여기서부터 생성!
[요약...]
```

만약 max_length=512로 잘리면:
```
<|start_header_id|>system<|end_header_id|>
당신은 요약 전문가입니다.
<|eot_id|><|start_header_id|>user<|end_header_id|>
다음 대화를 요약하세요:
---
[긴 대화... (여기서 끊김)]
```

→ **assistant 헤더 없음** → 모델은 "user가 말하는 중"으로 인식 → 빈 문자열 또는 대화 이어가기

**실측 데이터 (본 프로젝트):**

기존 설정 (encoder_max_len=512, decoder_max_len=100):
- 학습 Prompt 잘림: **6.07%** (756/12,457개)
- 학습 Full text 잘림: **4.50%** (560/12,457개)
- Dev Prompt 잘림: **6.81%** (34/499개)

개선 설정 (encoder_max_len=1024, decoder_max_len=200):
- 학습 Prompt 잘림: **0.11%** (14/12,457개)
- 학습 Full text 잘림: **0.07%** (9/12,457개)
- Dev Prompt 잘림: **0.00%** (0/499개)

**해결책:**

```python
# 1. Config 설정 (finetune_config.yaml)
tokenizer:
  encoder_max_len: 1024  # 512 → 1024 (LLM용 증가)
  decoder_max_len: 200   # 100 → 200 (여유 확보)

# 2. 학습 시: Right truncation 유지 (Label 계산 안정성)
full_ids = tokenizer(
    full_text,
    max_length=1024 + 200,  # 1224
    truncation=True,  # right truncation (기본값)
    ...
)

# 3. 추론 시: Left truncation 사용 (Assistant 헤더 보존)
if template_type:
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"  # ← 핵심!

inputs = tokenizer(
    batch_prompts,
    max_length=1024,
    truncation=True,
    ...
)

# 4. 생성 여유 확보
outputs = model.generate(
    **inputs,
    max_new_tokens=150,  # 100 → 150
    ...
)
```

**주의사항:**
- **학습 시 left truncation 금지**: Label 계산 로직이 틀어짐 (prompt 길이 불일치)
- **추론 시 left truncation 필수**: Assistant 헤더 보존이 최우선
- **실측 데이터 기반 설정**: 토크나이저로 실제 길이 측정 후 max_length 결정

**예상 성능 영향:**
- Prompt truncation 6% 발생 시: **-20~30 ROUGE points**
- 해결 시 즉각적인 성능 회복

## Fine-tuning Progress & Results

### 완료된 모델 (Completed)
| 모델 | ROUGE 점수 | 비고 |
|------|------------|------|
| **KoBART** | **94.51** | 가장 높은 성능, Encoder-Decoder 최고 |

### 제외된 모델 (Excluded)
| 모델 | 상태 | 이유 |
|------|------|------|
| KoT5 | ❌ Excluded | Overflow error (numerical instability) |

### 진행 중 모델 (In Progress)
| 모델 | 상태 | 진행률 |
|------|------|--------|
| **Llama-3.2-Korean-3B** | 🔄 Training | 33% (Step 127/390) |

### 대기 중 모델 (Pending)
| 모델 | 우선순위 | 비고 |
|------|----------|------|
| Qwen3-4B-Instruct | High | 4th place zero-shot (45.02) |
| Qwen2.5-7B-Instruct | High | 3rd place zero-shot (46.84) |
| Llama-3-Korean-8B | Medium | 2nd place zero-shot (48.61) |

### 성능 비교 기준
- **Baseline**: Random reference selection ~ 70 points
- **Target**: 90+ points (KoBART 수준 이상)
- **평가 지표**: ROUGE-1-F1 + ROUGE-2-F1 + ROUGE-L-F1 평균

## 학습 안정성 체크리스트

파인튜닝 시작 전 반드시 확인할 사항:

1. **디스크 용량 확인**
   ```bash
   du -sh / 2>/dev/null
   # 110GB 이하 유지 (150GB 한계)
   ```

2. **Chat Template 토큰 추가** (LLM 모델만 해당)
   - Llama: `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`
   - Qwen: `<|im_start|>`, `<|im_end|>`

3. **QLoRA compute_dtype 확인**
   - Llama: `torch.bfloat16` + `bf16=True`
   - Qwen: `torch.float16` + `fp16=True`

4. **metric_for_best_model 설정**
   - Encoder-Decoder: `"rouge_sum"` + `greater_is_better=True`
   - Causal LM: `"eval_loss"` + `greater_is_better=False`

5. **특수 토큰 추가**
   ```python
   special_tokens = ['#Person1#', '#Person2#', ..., '#PhoneNumber#', '#Address#', ...]
   tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
   ```

6. **모델 임베딩 크기 조정**
   ```python
   model.resize_token_embeddings(len(tokenizer))
   ```

7. **Max Length 설정 검증** (LLM 모델 필수!)
   - 실제 토큰 길이 통계 수집 후 설정
   - encoder_max_len: 1024 이상 권장 (512는 6% 잘림!)
   - decoder_max_len: 200 이상 권장
   - Left truncation 추론 시 적용: `tokenizer.truncation_side = "left"`