"""
학습 스크립트 - BART 기반 대화 요약 모델 학습

이 스크립트는 baseline.ipynb의 학습 코드를 바탕으로 작성되었습니다.
이미 작성된 모듈들을 import하여 사용하며, Config YAML 파일을 로딩하여 학습을 수행합니다.

사용법:
    python train_baseline.py --config configs/base_config.yaml
"""

import os
import argparse
import yaml
from typing import Dict, Any, Tuple
import torch
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq
)

# 프로젝트 모듈 import
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessor import Preprocess
from src.data.dataset import DatasetForTrain, DatasetForVal
from src.models.model_loader import load_tokenizer_and_model
from src.evaluation.metrics import compute_metrics_for_trainer
from src.utils.seed import set_seed


def load_config(config_path: str) -> Dict[str, Any]:
    """
    YAML 설정 파일을 로딩합니다.

    Args:
        config_path (str): YAML 설정 파일 경로

    Returns:
        Dict[str, Any]: 설정 딕셔너리
    """
    print('-' * 50)
    print(f'Loading config from: {config_path}')
    print('-' * 50)

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    print('Config loaded successfully!')
    print('-' * 50)

    return config


def prepare_train_dataset(
    config: Dict[str, Any],
    preprocessor: Preprocess,
    data_path: str,
    tokenizer: Any
) -> Tuple[DatasetForTrain, DatasetForVal]:
    """
    학습 및 검증 데이터셋을 준비합니다.

    이 함수는 baseline.ipynb의 Cell 25 코드를 기반으로 작성되었습니다.

    Args:
        config (Dict[str, Any]): 설정 딕셔너리
        preprocessor (Preprocess): 전처리 객체
        data_path (str): 데이터 경로
        tokenizer: HuggingFace 토크나이저

    Returns:
        Tuple[DatasetForTrain, DatasetForVal]: 학습 및 검증 데이터셋
    """
    print('=' * 80)
    print('1. 데이터 로딩 시작')
    print('=' * 80)

    # 데이터 파일 경로 설정
    train_file_path = os.path.join(data_path, 'train.csv')
    val_file_path = os.path.join(data_path, 'dev.csv')

    # train, validation 데이터프레임 구축
    train_data = preprocessor.make_set_as_df(train_file_path)
    val_data = preprocessor.make_set_as_df(val_file_path)

    print(f'Train data size: {len(train_data)}')
    print(f'Validation data size: {len(val_data)}')
    print('-' * 150)
    print(f'[Sample] train_data:\n{train_data["dialogue"][0]}')
    print(f'[Sample] train_label:\n{train_data["summary"][0]}')
    print('-' * 150)
    print(f'[Sample] val_data:\n{val_data["dialogue"][0]}')
    print(f'[Sample] val_label:\n{val_data["summary"][0]}')
    print('-' * 150)

    # 인코더/디코더 입력 생성
    encoder_input_train, decoder_input_train, decoder_output_train = preprocessor.make_input(train_data)
    encoder_input_val, decoder_input_val, decoder_output_val = preprocessor.make_input(val_data)
    print('-' * 10, 'Load data complete', '-' * 10)

    print('=' * 80)
    print('2. 데이터 토크나이징 시작')
    print('=' * 80)

    # 학습 데이터 토크나이징
    tokenized_encoder_inputs = tokenizer(
        encoder_input_train,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['encoder_max_len'],
        return_token_type_ids=False
    )

    tokenized_decoder_inputs = tokenizer(
        decoder_input_train,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['decoder_max_len'],
        return_token_type_ids=False
    )

    tokenized_decoder_outputs = tokenizer(
        decoder_output_train,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['decoder_max_len'],
        return_token_type_ids=False
    )

    train_inputs_dataset = DatasetForTrain(
        tokenized_encoder_inputs,
        tokenized_decoder_inputs,
        tokenized_decoder_outputs,
        len(encoder_input_train)
    )

    # 검증 데이터 토크나이징
    val_tokenized_encoder_inputs = tokenizer(
        encoder_input_val,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['encoder_max_len'],
        return_token_type_ids=False
    )

    val_tokenized_decoder_inputs = tokenizer(
        decoder_input_val,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['decoder_max_len'],
        return_token_type_ids=False
    )

    val_tokenized_decoder_outputs = tokenizer(
        decoder_output_val,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['decoder_max_len'],
        return_token_type_ids=False
    )

    val_inputs_dataset = DatasetForVal(
        val_tokenized_encoder_inputs,
        val_tokenized_decoder_inputs,
        val_tokenized_decoder_outputs,
        len(encoder_input_val)
    )

    print('-' * 10, 'Make dataset complete', '-' * 10)
    print(f'Train dataset size: {len(train_inputs_dataset)}')
    print(f'Validation dataset size: {len(val_inputs_dataset)}')

    return train_inputs_dataset, val_inputs_dataset


def create_trainer(
    config: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    train_dataset: DatasetForTrain,
    val_dataset: DatasetForVal
) -> Seq2SeqTrainer:
    """
    Seq2SeqTrainer를 생성합니다.

    이 함수는 baseline.ipynb의 Cell 28 코드를 기반으로 작성되었습니다.

    Args:
        config (Dict[str, Any]): 설정 딕셔너리
        model: 학습할 모델
        tokenizer: 토크나이저
        train_dataset (DatasetForTrain): 학습 데이터셋
        val_dataset (DatasetForVal): 검증 데이터셋

    Returns:
        Seq2SeqTrainer: 생성된 Trainer 객체
    """
    print('=' * 80)
    print('3. Trainer 설정 시작')
    print('=' * 80)

    print('-' * 10, 'Make training arguments', '-' * 10)

    # Seq2SeqTrainingArguments 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['general']['output_dir'],
        overwrite_output_dir=config['training']['overwrite_output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        optim=config['training']['optim'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        eval_strategy=config['training']['eval_strategy'],
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        fp16=config['training']['fp16'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        seed=config['training']['seed'],
        logging_dir=config['training']['logging_dir'],
        logging_strategy=config['training']['logging_strategy'],
        predict_with_generate=config['training']['predict_with_generate'],
        generation_max_length=config['training']['generation_max_length'],
        do_train=config['training']['do_train'],
        do_eval=config['training']['do_eval'],
        # wandb 제외
        report_to="none"
    )

    print('-' * 10, 'Make training arguments complete', '-' * 10)
    print(f'Output directory: {training_args.output_dir}')
    print(f'Number of epochs: {training_args.num_train_epochs}')
    print(f'Learning rate: {training_args.learning_rate}')
    print(f'Train batch size: {training_args.per_device_train_batch_size}')
    print(f'Eval batch size: {training_args.per_device_eval_batch_size}')

    # EarlyStopping 콜백 설정
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_threshold=config['training']['early_stopping_threshold']
    )

    print('-' * 10, 'Early stopping configured', '-' * 10)
    print(f'Patience: {config["training"]["early_stopping_patience"]}')
    print(f'Threshold: {config["training"]["early_stopping_threshold"]}')

    # DataCollatorForSeq2Seq 사용 (동적 패딩)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    print('-' * 10, 'Make trainer', '-' * 10)

    # Seq2SeqTrainer 생성
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_for_trainer(
            tokenizer=tokenizer,
            remove_tokens=config['inference']['remove_tokens']
        ),
        callbacks=[early_stopping_callback]
    )

    print('-' * 10, 'Make trainer complete', '-' * 10)

    return trainer


def main(config_path: str) -> None:
    """
    메인 학습 함수

    이 함수는 baseline.ipynb의 Cell 32 코드를 기반으로 작성되었습니다.

    Args:
        config_path (str): 설정 파일 경로
    """
    # Config 로딩
    config = load_config(config_path)

    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('-' * 10, f'Device: {device}', '-' * 10)
    print(f'PyTorch version: {torch.__version__}')

    # Seed 설정
    if 'seed' in config['training']:
        set_seed(config['training']['seed'])
        print(f'Random seed set to: {config["training"]["seed"]}')

    # 모델 및 토크나이저 로딩
    print('=' * 80)
    print('Loading Model & Tokenizer')
    print('=' * 80)

    model, tokenizer = load_tokenizer_and_model(
        model_name=config['general']['model_name'],
        special_tokens=config['tokenizer']['special_tokens'],
        device=device
    )

    print('-' * 10, "Tokenizer special tokens:", tokenizer.special_tokens_map, '-' * 10)

    # 데이터셋 준비
    preprocessor = Preprocess(
        bos_token=config['tokenizer']['bos_token'],
        eos_token=config['tokenizer']['eos_token']
    )

    data_path = config['general']['data_path']
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(
        config, preprocessor, data_path, tokenizer
    )

    # Trainer 생성
    trainer = create_trainer(
        config, model, tokenizer, train_inputs_dataset, val_inputs_dataset
    )

    # 학습 시작
    print('=' * 80)
    print('4. 학습 시작')
    print('=' * 80)

    trainer.train()

    print('=' * 80)
    print('5. 학습 완료')
    print('=' * 80)

    # 최종 모델 저장
    final_model_path = os.path.join(config['general']['output_dir'], 'final_model')
    print(f'Saving final model to: {final_model_path}')
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print('=' * 80)
    print('학습이 모두 완료되었습니다!')
    print(f'최종 모델 저장 경로: {final_model_path}')
    print('=' * 80)


if __name__ == "__main__":
    # ArgumentParser 설정
    parser = argparse.ArgumentParser(
        description='BART 기반 대화 요약 모델 학습 스크립트'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='YAML 설정 파일 경로 (예: configs/base_config.yaml)'
    )

    args = parser.parse_args()

    # 설정 파일 존재 여부 확인
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # 메인 함수 실행
    main(args.config)
