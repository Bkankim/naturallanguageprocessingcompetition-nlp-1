"""
모델 및 토크나이저 로딩 모듈

이 모듈은 BART 기반 대화 요약 모델과 토크나이저를 로딩하는 함수를 제공합니다.
"""

from typing import Tuple, List, Optional
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig


def load_tokenizer_and_model(
    model_name: str,
    special_tokens: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,
) -> Tuple[BartForConditionalGeneration, AutoTokenizer]:
    """
    BART 토크나이저와 모델을 로딩하고 초기화합니다.

    Args:
        model_name (str): HuggingFace 모델 이름 (예: "digit82/kobart-summarization")
        special_tokens (Optional[List[str]]): 추가할 특수 토큰 리스트.
            기본값은 대화 요약 태스크에 필요한 토큰들:
            ['#Person1#', '#Person2#', '#Person3#', '#PhoneNumber#', '#Address#', '#PassportNumber#']
        device (Optional[torch.device]): 모델을 로드할 디바이스. None이면 자동으로 CUDA/CPU 선택.
        checkpoint_path (Optional[str]): 사전 학습된 체크포인트 경로. None이면 HuggingFace Hub에서 로드.

    Returns:
        Tuple[BartForConditionalGeneration, AutoTokenizer]:
            초기화된 BART 모델과 토크나이저 튜플

    Examples:
        >>> # 기본 사용법
        >>> model, tokenizer = load_tokenizer_and_model("digit82/kobart-summarization")

        >>> # 커스텀 special tokens와 체크포인트 사용
        >>> custom_tokens = ['#Person1#', '#Person2#', '#Email#']
        >>> model, tokenizer = load_tokenizer_and_model(
        ...     model_name="digit82/kobart-summarization",
        ...     special_tokens=custom_tokens,
        ...     checkpoint_path="./checkpoints/best_model"
        ... )
    """
    # 기본 special tokens 설정
    if special_tokens is None:
        special_tokens = [
            '#Person1#', '#Person2#', '#Person3#',
            '#PhoneNumber#', '#Address#', '#PassportNumber#'
        ]

    # 디바이스 자동 선택
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('-' * 10, 'Load tokenizer & model', '-' * 10)
    print('-' * 10, f'Model Name: {model_name}', '-' * 10)
    print('-' * 10, f'Device: {device}', '-' * 10)

    # 토크나이저 로딩
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Special tokens 추가
    if special_tokens:
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        tokenizer.add_special_tokens(special_tokens_dict)
        print('-' * 10, f'Added {len(special_tokens)} special tokens', '-' * 10)

    # 모델 로딩
    if checkpoint_path:
        # 체크포인트에서 로딩
        print('-' * 10, f'Loading from checkpoint: {checkpoint_path}', '-' * 10)
        model = BartForConditionalGeneration.from_pretrained(checkpoint_path)
    else:
        # HuggingFace Hub에서 로딩
        bart_config = BartConfig.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name, config=bart_config)

    # 토큰 임베딩 크기 재조정 (special tokens 추가 후)
    model.resize_token_embeddings(len(tokenizer))

    # 모델을 디바이스로 이동
    model.to(device)

    print('-' * 10, 'Load tokenizer & model complete', '-' * 10)
    print(f'Tokenizer vocab size: {len(tokenizer)}')
    print(f'Model config:\n{model.config}')

    return model, tokenizer


def get_model_info(model: BartForConditionalGeneration, tokenizer: AutoTokenizer) -> dict:
    """
    모델과 토크나이저의 상세 정보를 반환합니다.

    Args:
        model (BartForConditionalGeneration): BART 모델
        tokenizer (AutoTokenizer): 토크나이저

    Returns:
        dict: 모델 및 토크나이저 정보 딕셔너리
    """
    info = {
        'vocab_size': len(tokenizer),
        'special_tokens': tokenizer.special_tokens_map,
        'additional_special_tokens': tokenizer.additional_special_tokens,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'encoder_layers': model.config.encoder_layers,
        'decoder_layers': model.config.decoder_layers,
        'd_model': model.config.d_model,
    }
    return info
