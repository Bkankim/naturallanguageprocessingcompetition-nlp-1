# train_baseline.py 사용 가이드

## 개요
`train_baseline.py`는 baseline.ipynb의 학습 코드를 바탕으로 작성된 실행 가능한 학습 스크립트입니다.
이미 작성된 모듈들을 import하여 사용하며, Config YAML 파일을 로딩하여 BART 기반 대화 요약 모델을 학습합니다.

## 특징
- **모듈화된 코드**: 이미 작성된 src 모듈들을 활용
- **Config 기반**: YAML 파일로 하이퍼파라미터 관리
- **Type hints 및 Docstring**: 코드 가독성 및 유지보수성 향상
- **진행 상황 출력**: 학습 진행 상황을 명확하게 출력
- **WandB 제외**: 일단 WandB는 제외하고 로컬 학습에 집중

## 사용 방법

### 1. 기본 실행
```bash
cd /Competition/NLP/dialogue-summarization
python scripts/train_baseline.py --config configs/train_config.yaml
```

### 2. 커스텀 Config 파일 사용
```bash
python scripts/train_baseline.py --config configs/my_custom_config.yaml
```

### 3. Help 확인
```bash
python scripts/train_baseline.py --help
```

## 주요 기능

### 1. Config 파일 로딩
- YAML 형식의 설정 파일을 로딩
- 모델, 데이터, 학습 하이퍼파라미터 등 모든 설정 관리

### 2. 데이터 준비 (prepare_train_dataset)
- CSV 파일에서 train/validation 데이터 로딩
- BART 모델을 위한 encoder/decoder 입력 생성
- 토크나이징 및 PyTorch Dataset 생성

### 3. Trainer 생성 (create_trainer)
- Seq2SeqTrainingArguments 설정
- DataCollatorForSeq2Seq를 사용한 동적 패딩
- EarlyStoppingCallback 설정
- ROUGE 메트릭을 사용한 평가

### 4. 학습 및 저장
- Seq2SeqTrainer를 사용한 학습
- 최고 성능 모델 자동 저장
- 최종 모델을 지정된 경로에 저장

## Config 파일 구조

### general
- `data_path`: 데이터 디렉토리 경로
- `model_name`: HuggingFace 모델 이름
- `output_dir`: 체크포인트 저장 경로

### tokenizer
- `encoder_max_len`: 인코더 최대 길이 (기본: 512)
- `decoder_max_len`: 디코더 최대 길이 (기본: 100)
- `bos_token`: Beginning of Sequence 토큰
- `eos_token`: End of Sequence 토큰
- `special_tokens`: 추가 특수 토큰 리스트

### training
- `num_train_epochs`: 학습 에폭 수
- `learning_rate`: 학습률
- `per_device_train_batch_size`: 배치 크기
- `warmup_ratio`: Warmup 비율
- `early_stopping_patience`: Early stopping patience
- 기타 Seq2SeqTrainingArguments 파라미터

### inference
- `remove_tokens`: 평가 시 제거할 토큰 리스트

## 출력 결과

### 학습 중
- Config 정보
- 모델 및 토크나이저 정보
- 데이터셋 정보 및 샘플
- 학습 진행 상황 (에폭별 loss, ROUGE 점수)
- 평가 결과 (ROUGE-1, ROUGE-2, ROUGE-L F1 점수)

### 학습 완료 후
- 체크포인트: `{output_dir}/checkpoint-{step}/`
- 최종 모델: `{output_dir}/final_model/`
- 로그: `{logging_dir}/`

## 예시 출력
```
==================================================
Loading config from: configs/train_config.yaml
==================================================
Config loaded successfully!
==================================================
---------- Device: cuda ----------
PyTorch version: 2.0.1
================================================================================
Loading Model & Tokenizer
================================================================================
---------- Load tokenizer & model ----------
---------- Model Name: digit82/kobart-summarization ----------
---------- Device: cuda ----------
...
================================================================================
1. 데이터 로딩 시작
================================================================================
Train data size: 12457
Validation data size: 499
...
================================================================================
4. 학습 시작
================================================================================
Epoch 1/20: 100%|████████████| ...
...
================================================================================
5. 학습 완료
================================================================================
Saving final model to: /Competition/NLP/dialogue-summarization/checkpoints/baseline_run/final_model
================================================================================
학습이 모두 완료되었습니다!
최종 모델 저장 경로: /Competition/NLP/dialogue-summarization/checkpoints/baseline_run/final_model
================================================================================
```

## 주의사항
- GPU 메모리가 부족할 경우 `per_device_train_batch_size`를 줄이세요
- 학습 시간이 오래 걸리므로 `num_train_epochs`를 조정하세요
- Early stopping이 작동하므로 과적합 걱정 없이 에폭 수를 늘려도 됩니다
- 데이터 경로가 올바른지 확인하세요

## 관련 파일
- 학습 스크립트: `/Competition/NLP/dialogue-summarization/scripts/train_baseline.py`
- Config 파일: `/Competition/NLP/dialogue-summarization/configs/train_config.yaml`
- 전처리 모듈: `/Competition/NLP/dialogue-summarization/src/data/preprocessor.py`
- 데이터셋 모듈: `/Competition/NLP/dialogue-summarization/src/data/dataset.py`
- 모델 로더: `/Competition/NLP/dialogue-summarization/src/models/model_loader.py`
- 평가 메트릭: `/Competition/NLP/dialogue-summarization/src/evaluation/metrics.py`
- 시드 유틸: `/Competition/NLP/dialogue-summarization/src/utils/seed.py`

## 문제 해결

### ModuleNotFoundError
```bash
# 프로젝트 루트에서 실행하는지 확인
cd /Competition/NLP/dialogue-summarization
python scripts/train_baseline.py --config configs/train_config.yaml
```

### CUDA Out of Memory
```yaml
# config 파일에서 배치 크기 줄이기
training:
  per_device_train_batch_size: 16  # 50 -> 16으로 줄이기
  per_device_eval_batch_size: 16   # 32 -> 16으로 줄이기
```

### FileNotFoundError: data not found
```yaml
# config 파일에서 데이터 경로 확인
general:
  data_path: "/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/data/"
```
