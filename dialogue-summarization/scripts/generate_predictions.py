"""
추론(Inference) 스크립트

baseline.ipynb의 추론 코드를 바탕으로 작성된 실행 가능한 추론 스크립트입니다.
학습된 모델 체크포인트를 로드하여 테스트 데이터에 대한 요약문을 생성하고 CSV 파일로 저장합니다.

사용법:
    python generate_predictions.py --config configs/base_config.yaml --checkpoint checkpoints/best_model

    또는 개별 인자 지정:
    python generate_predictions.py \
        --config configs/base_config.yaml \
        --checkpoint checkpoints/best_model \
        --output submissions/predictions.csv \
        --batch_size 32
"""

import os
import argparse
import yaml
from typing import Dict, Any, List
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

# 프로젝트 모듈 import
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessor import Preprocess
from src.data.dataset import DatasetForInference
from src.models.model_loader import load_tokenizer_and_model
from src.evaluation.metrics import clean_text


def load_config(config_path: str) -> Dict[str, Any]:
    """
    YAML 설정 파일을 로드합니다.

    Args:
        config_path (str): YAML 설정 파일 경로

    Returns:
        Dict[str, Any]: 로드된 설정 딕셔너리
    """
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def prepare_test_dataset(
    config: Dict[str, Any],
    preprocessor: Preprocess,
    tokenizer: Any
) -> tuple:
    """
    테스트 데이터셋을 준비하고 토크나이징합니다.

    Args:
        config (Dict[str, Any]): 설정 딕셔너리
        preprocessor (Preprocess): 전처리기 인스턴스
        tokenizer: HuggingFace 토크나이저

    Returns:
        tuple: (test_data, test_encoder_inputs_dataset)
            - test_data: 원본 테스트 데이터프레임
            - test_encoder_inputs_dataset: 토크나이징된 테스트 데이터셋
    """
    test_file_path = os.path.join(config['general']['data_path'], 'test.csv')

    # 테스트 데이터 로드
    test_data = preprocessor.make_set_as_df(test_file_path, is_train=False)
    test_id = test_data['fname']

    print('-' * 150)
    print(f'테스트 데이터 샘플:\n{test_data["dialogue"][0]}')
    print('-' * 150)

    # 인코더 입력 생성
    encoder_input_test, decoder_input_test = preprocessor.make_input(test_data, is_test=True)
    print('-' * 10, '데이터 로드 완료', '-' * 10)

    # 토크나이징
    test_tokenized_encoder_inputs = tokenizer(
        encoder_input_test,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['encoder_max_len'],
        return_token_type_ids=False,
    )

    # 추론용 데이터셋 생성
    test_encoder_inputs_dataset = DatasetForInference(
        test_tokenized_encoder_inputs,
        test_id,
        len(encoder_input_test)
    )
    print('-' * 10, '데이터셋 생성 완료', '-' * 10)

    return test_data, test_encoder_inputs_dataset


def generate_summaries(
    config: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    dataloader: DataLoader,
    device: torch.device
) -> tuple:
    """
    모델을 사용하여 요약문을 생성합니다.

    Args:
        config (Dict[str, Any]): 설정 딕셔너리
        model: 학습된 BART 모델
        tokenizer: HuggingFace 토크나이저
        dataloader: 테스트 데이터 로더
        device: 연산에 사용할 디바이스

    Returns:
        tuple: (text_ids, summaries)
            - text_ids: 테스트 데이터 식별자 리스트
            - summaries: 생성된 요약문 리스트
    """
    model.eval()
    summaries = []
    text_ids = []

    print('-' * 10, '추론 시작', '-' * 10)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="생성 중"):
            # 배치의 ID 수집
            text_ids.extend(batch['ID'])

            # 모델 추론
            generated_ids = model.generate(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                early_stopping=config['inference']['early_stopping'],
                max_length=config['inference']['generate_max_length'],
                num_beams=config['inference']['num_beams'],
            )

            # 생성된 토큰 ID를 텍스트로 디코딩
            for ids in generated_ids:
                result = tokenizer.decode(ids, skip_special_tokens=False)
                summaries.append(result)

    print('-' * 10, f'추론 완료: {len(summaries)}개 요약문 생성', '-' * 10)

    return text_ids, summaries


def postprocess_summaries(
    summaries: List[str],
    remove_tokens: List[str]
) -> List[str]:
    """
    생성된 요약문에서 불필요한 토큰을 제거하고 정리합니다.

    Args:
        summaries (List[str]): 원본 요약문 리스트
        remove_tokens (List[str]): 제거할 토큰 리스트

    Returns:
        List[str]: 정리된 요약문 리스트
    """
    # clean_text 함수를 사용하여 토큰 제거
    cleaned_summaries = clean_text(summaries, remove_tokens)

    print('-' * 10, '후처리 완료', '-' * 10)
    print(f'샘플 요약문 (후처리 후):\n{cleaned_summaries[0]}')
    print('-' * 150)

    return cleaned_summaries


def save_predictions(
    fname_list: List[str],
    summaries: List[str],
    output_path: str
) -> None:
    """
    예측 결과를 CSV 파일로 저장합니다.

    Args:
        fname_list (List[str]): 파일명 리스트
        summaries (List[str]): 요약문 리스트
        output_path (str): 출력 파일 경로
    """
    # 데이터프레임 생성
    output_df = pd.DataFrame({
        "fname": fname_list,
        "summary": summaries,
    })

    # 출력 디렉토리 생성
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # CSV 파일 저장
    output_df.to_csv(output_path, index=False)
    print('-' * 10, f'예측 결과 저장 완료: {output_path}', '-' * 10)
    print(f'저장된 샘플 수: {len(output_df)}')


def inference(
    config: Dict[str, Any],
    checkpoint_path: str,
    output_path: str,
    batch_size: int = None
) -> pd.DataFrame:
    """
    학습된 모델로 테스트 데이터에 대한 추론을 수행합니다.

    Args:
        config (Dict[str, Any]): 설정 딕셔너리
        checkpoint_path (str): 모델 체크포인트 경로
        output_path (str): 출력 파일 경로
        batch_size (int, optional): 배치 크기. None이면 config의 값 사용

    Returns:
        pd.DataFrame: 예측 결과 데이터프레임 (fname, summary)
    """
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('-' * 10, f'디바이스: {device}', '-' * 10)
    print(f'PyTorch 버전: {torch.__version__}')

    # 배치 크기 설정
    if batch_size is None:
        batch_size = config['inference']['batch_size']

    # 모델 및 토크나이저 로드
    print('-' * 10, '모델 로딩 중...', '-' * 10)
    model, tokenizer = load_tokenizer_and_model(
        model_name=config['general']['model_name'],
        special_tokens=config['tokenizer']['special_tokens'],
        device=device,
        checkpoint_path=checkpoint_path
    )

    # 전처리기 초기화
    preprocessor = Preprocess(
        bos_token=config['tokenizer']['bos_token'],
        eos_token=config['tokenizer']['eos_token']
    )

    # 테스트 데이터 준비
    test_data, test_dataset = prepare_test_dataset(config, preprocessor, tokenizer)

    # 데이터 로더 생성
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 요약문 생성
    text_ids, summaries = generate_summaries(
        config=config,
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        device=device
    )

    # 후처리: 불필요한 토큰 제거
    remove_tokens = config['inference']['remove_tokens']
    cleaned_summaries = postprocess_summaries(summaries, remove_tokens)

    # 결과 저장
    save_predictions(text_ids, cleaned_summaries, output_path)

    # 결과 데이터프레임 생성 및 반환
    output_df = pd.DataFrame({
        "fname": text_ids,
        "summary": cleaned_summaries,
    })

    return output_df


def main():
    """
    메인 함수: 명령줄 인자를 파싱하고 추론을 실행합니다.
    """
    parser = argparse.ArgumentParser(
        description="대화 요약 모델 추론 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="설정 YAML 파일 경로 (예: configs/base_config.yaml)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="학습된 모델 체크포인트 경로 (예: checkpoints/best_model)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 CSV 파일 경로 (기본값: config의 result_path/output.csv)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="추론 배치 크기 (기본값: config의 batch_size)"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="데이터 경로 (기본값: config의 data_path)"
    )

    args = parser.parse_args()

    # 설정 파일 로드
    print(f"설정 파일 로드 중: {args.config}")
    config = load_config(args.config)

    # 데이터 경로 오버라이드
    if args.data_path:
        config['general']['data_path'] = args.data_path

    # 출력 경로 설정
    if args.output:
        output_path = args.output
    else:
        result_dir = config['inference']['result_path']
        output_path = os.path.join(result_dir, "output.csv")

    # 추론 실행
    print("=" * 150)
    print("추론 시작".center(150))
    print("=" * 150)

    output_df = inference(
        config=config,
        checkpoint_path=args.checkpoint,
        output_path=output_path,
        batch_size=args.batch_size
    )

    # 결과 미리보기
    print("\n" + "=" * 150)
    print("추론 완료".center(150))
    print("=" * 150)
    print("\n예측 결과 샘플 (처음 5개):")
    print(output_df.head())


if __name__ == "__main__":
    main()
