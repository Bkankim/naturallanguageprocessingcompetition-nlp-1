# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## .md Update Guidelines
- After completing each task, always prioritize updating all documents to ensure they are up to date.

## Git Updates
- After completing each task, update the .gitignore file and always synchronize with the GitHub repository (as a backup practice).

## code agents Guidelines
- 각 명령을 수행할 때, 너의 판단을 통해서 SubAgents를 적극 활용할 것 (각각의 임무를 정의해둔 서브 에이전트들이 존재).
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
- Ensure that the root (/) disk usage does not exceed the 150GB limit.

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