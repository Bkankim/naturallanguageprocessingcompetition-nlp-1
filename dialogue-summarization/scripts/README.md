# Scripts 사용 가이드

이 디렉토리는 대화 요약 모델의 학습 및 추론을 위한 실행 가능한 스크립트를 포함합니다.

## 목차
1. [train_baseline.py](#train_baselinepy) - 학습 스크립트
2. [generate_predictions.py](#generate_predictionspy) - 추론 스크립트

---

## train_baseline.py

baseline.ipynb의 학습 코드(Cell 25-33)를 바탕으로 작성된 실행 가능한 학습 스크립트입니다.

### 주요 기능

- Config YAML 파일 로딩
- 데이터 로딩 및 전처리
- BART 모델 로딩 및 토크나이저 설정
- Seq2SeqTrainingArguments 설정
- Seq2SeqTrainer 생성 및 학습 실행
- 최종 모델 저장

### 사용법

#### 기본 사용법

```bash
python scripts/train_baseline.py --config configs/train_config.yaml
```

#### Help 확인

```bash
python scripts/train_baseline.py --help
```

### 명령줄 인자

| 인자 | 필수 여부 | 기본값 | 설명 |
|------|----------|--------|------|
| `--config` | 필수 | - | YAML 설정 파일 경로 |

### 처리 흐름

1. **Config 로딩**: YAML 파일에서 모델 및 학습 설정 로드
2. **모델 로딩**: HuggingFace Hub에서 사전학습된 BART 모델 로드
3. **데이터 준비**: train/dev 데이터 로드 및 토크나이징
4. **Trainer 생성**: Seq2SeqTrainer 설정 (DataCollator, Metrics, EarlyStopping)
5. **학습 실행**: 모델 학습 수행
6. **모델 저장**: 최종 모델을 지정된 경로에 저장

### 주요 특징

- **모듈 재사용**: 기존 작성된 모듈들을 import하여 사용
  - `src.data.preprocessor.Preprocess`
  - `src.data.dataset.DatasetForTrain`, `DatasetForVal`
  - `src.models.model_loader.load_tokenizer_and_model`
  - `src.evaluation.metrics.compute_metrics_for_trainer`
  - `src.utils.seed.set_seed`

- **Type hints**: 모든 함수에 타입 힌트 제공
- **Docstring**: 상세한 함수 설명 포함
- **진행 상황 표시**: 학습 단계별 진행 상황 출력
- **DataCollatorForSeq2Seq**: 동적 패딩 사용
- **WandB 제외**: 로컬 학습에 집중

### 설정 파일 예시

`configs/train_config.yaml` 참고:

```yaml
general:
  data_path: "/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/data/"
  model_name: "digit82/kobart-summarization"
  output_dir: "/Competition/NLP/dialogue-summarization/checkpoints/baseline_run"

training:
  num_train_epochs: 20
  learning_rate: 1.0e-05
  per_device_train_batch_size: 50
  per_device_eval_batch_size: 32
  early_stopping_patience: 3
  # ... 기타 설정
```

### 출력 결과

- **체크포인트**: `{output_dir}/checkpoint-{step}/`
- **최종 모델**: `{output_dir}/final_model/`
- **로그**: `{logging_dir}/`

### 트러블슈팅

#### CUDA Out of Memory 오류

config 파일에서 배치 크기를 줄여보세요:

```yaml
training:
  per_device_train_batch_size: 16  # 50 -> 16
  per_device_eval_batch_size: 16   # 32 -> 16
```

#### 데이터 파일을 찾을 수 없음

config 파일의 `general.data_path`를 수정하세요.

자세한 내용은 `README_train.md`를 참고하세요.

---

## generate_predictions.py

baseline.ipynb의 추론 코드(Cell 37-40)를 바탕으로 작성된 실행 가능한 추론 스크립트입니다.

### 주요 기능

- 학습된 모델 체크포인트 로드
- 테스트 데이터 전처리 및 토크나이징
- 배치 추론 (DataLoader 사용)
- 토큰 정리 및 후처리
- submission 형식의 CSV 파일 생성 (fname, summary)

### 사용법

#### 기본 사용법

```bash
python scripts/generate_predictions.py \
    --config configs/base_config.yaml \
    --checkpoint checkpoints/best_model
```

#### 모든 옵션 사용

```bash
python scripts/generate_predictions.py \
    --config configs/base_config.yaml \
    --checkpoint checkpoints/best_model \
    --output submissions/my_predictions.csv \
    --batch_size 64 \
    --data_path /path/to/data
```

### 명령줄 인자

| 인자 | 필수 여부 | 기본값 | 설명 |
|------|----------|--------|------|
| `--config` | 필수 | - | YAML 설정 파일 경로 |
| `--checkpoint` | 필수 | - | 학습된 모델 체크포인트 경로 |
| `--output` | 선택 | config의 `result_path/output.csv` | 출력 CSV 파일 경로 |
| `--batch_size` | 선택 | config의 `batch_size` | 추론 배치 크기 |
| `--data_path` | 선택 | config의 `data_path` | 데이터 디렉토리 경로 |

### 출력 형식

CSV 파일로 저장되며, 다음 두 개의 컬럼을 포함합니다:

- `fname`: 테스트 데이터 파일명 (식별자)
- `summary`: 생성된 요약문

```csv
fname,summary
dialogsum_test_1,Person1과 Person2가 만나서 식사 계획을 논의했습니다
dialogsum_test_2,회의에서 프로젝트 일정과 예산을 검토했습니다
...
```

### 사용 예시

#### 1. 기본 추론 실행

```bash
cd /Competition/NLP/dialogue-summarization

python scripts/generate_predictions.py \
    --config configs/base_config.yaml \
    --checkpoint ./output/checkpoint-1000
```

#### 2. 출력 경로 지정

```bash
python scripts/generate_predictions.py \
    --config configs/base_config.yaml \
    --checkpoint ./output/checkpoint-1000 \
    --output submissions/submission_v1.csv
```

#### 3. 배치 크기 조정 (GPU 메모리 부족 시)

```bash
python scripts/generate_predictions.py \
    --config configs/base_config.yaml \
    --checkpoint ./output/checkpoint-1000 \
    --batch_size 16
```

#### 4. 다른 데이터 경로 사용

```bash
python scripts/generate_predictions.py \
    --config configs/base_config.yaml \
    --checkpoint ./output/checkpoint-1000 \
    --data_path /Competition/NLP/data
```

### 처리 흐름

1. **설정 로드**: YAML 파일에서 모델 및 추론 설정 로드
2. **모델 로드**: 체크포인트에서 학습된 모델과 토크나이저 로드
3. **데이터 준비**: 테스트 데이터 로드 및 토크나이징
4. **배치 추론**: DataLoader를 사용한 배치 단위 요약문 생성
5. **후처리**: 특수 토큰 제거 및 텍스트 정리
6. **결과 저장**: CSV 파일로 예측 결과 저장

### 주요 특징

- **모듈 재사용**: 기존 작성된 모듈들을 import하여 사용
  - `src.data.preprocessor.Preprocess`
  - `src.data.dataset.DatasetForInference`
  - `src.models.model_loader.load_tokenizer_and_model`
  - `src.evaluation.metrics.clean_text`

- **Type hints**: 모든 함수에 타입 힌트 제공
- **Docstring**: 상세한 함수 설명 포함
- **진행 상황 표시**: tqdm을 사용한 진행률 출력
- **에러 처리**: 안전한 디렉토리 생성 및 파일 처리

### 설정 파일 예시

`configs/base_config.yaml`에서 추론 관련 설정을 조정할 수 있습니다:

```yaml
inference:
  ckt_path: "model ckt path"  # 이 값은 --checkpoint 인자로 오버라이드됨
  result_path: "./prediction/"  # 기본 출력 디렉토리
  no_repeat_ngram_size: 2  # n-gram 반복 방지
  early_stopping: true  # 조기 종료
  generate_max_length: 100  # 생성 최대 길이
  num_beams: 4  # Beam search beam 개수
  batch_size: 32  # 배치 크기
  remove_tokens:  # 후처리 시 제거할 토큰
    - "<usr>"
    - "</s>"
    - "<pad>"
```

### 트러블슈팅

#### CUDA Out of Memory 오류

배치 크기를 줄여보세요:

```bash
python scripts/generate_predictions.py \
    --config configs/base_config.yaml \
    --checkpoint ./output/checkpoint-1000 \
    --batch_size 8
```

#### 체크포인트 로드 오류

체크포인트 경로가 올바른지 확인하세요. HuggingFace Trainer가 저장한 디렉토리 전체를 지정해야 합니다:

```
output/
├── checkpoint-1000/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── ...
```

#### 데이터 파일을 찾을 수 없음

`--data_path` 인자로 올바른 데이터 경로를 지정하거나, config 파일의 `general.data_path`를 수정하세요.

### 성능 최적화 팁

1. **배치 크기 증가**: GPU 메모리가 충분하다면 배치 크기를 늘려 추론 속도 향상
2. **Beam search 조정**: `num_beams`를 줄이면 속도는 빨라지지만 품질이 떨어질 수 있음
3. **Mixed precision**: FP16을 사용하면 메모리 사용량과 속도가 개선됨 (자동으로 적용됨)

### 참고

- baseline.ipynb의 Cell 37-40 코드를 기반으로 작성
- submission 형식에 맞춰 fname과 summary 컬럼만 포함
- clean_text 함수를 사용하여 토큰 정리 및 후처리 수행
