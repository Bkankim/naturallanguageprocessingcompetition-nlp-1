"""
평가 메트릭 모듈

이 모듈은 대화 요약 모델의 성능을 평가하기 위한 ROUGE 메트릭을 제공합니다.
konlpy 의존성 없이 rouge 라이브러리만 사용합니다.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from rouge import Rouge
from transformers import PreTrainedTokenizer, EvalPrediction


def compute_metrics_for_trainer(
    tokenizer: PreTrainedTokenizer,
    remove_tokens: Optional[List[str]] = None,
) -> callable:
    """
    HuggingFace Trainer에서 사용할 메트릭 계산 함수를 생성합니다.

    Args:
        tokenizer (PreTrainedTokenizer): 디코딩에 사용할 토크나이저
        remove_tokens (Optional[List[str]]): 평가 전 제거할 토큰 리스트.
            기본값은 ['<usr>', '<s>', '</s>', '<pad>']

    Returns:
        callable: EvalPrediction을 받아 메트릭 딕셔너리를 반환하는 함수

    Examples:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("digit82/kobart-summarization")
        >>> compute_metrics = compute_metrics_for_trainer(tokenizer)
        >>>
        >>> # Trainer에서 사용
        >>> trainer = Seq2SeqTrainer(
        ...     model=model,
        ...     args=training_args,
        ...     compute_metrics=compute_metrics,
        ... )
    """
    # 기본 remove_tokens 설정
    if remove_tokens is None:
        remove_tokens = ['<usr>', tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token]

    def compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
        """
        예측 결과를 받아 ROUGE 점수를 계산합니다.

        Args:
            pred (EvalPrediction): 예측 결과 객체
                - predictions: 모델 예측 토큰 ID 배열
                - label_ids: 정답 토큰 ID 배열

        Returns:
            Dict[str, float]: ROUGE-1, ROUGE-2, ROUGE-L F1 점수 딕셔너리
        """
        rouge = Rouge()
        predictions = pred.predictions
        labels = pred.label_ids

        # -100을 패딩 토큰으로 대체 (loss 계산 시 무시된 토큰)
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # 토큰 ID를 텍스트로 디코딩
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=False)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)

        # 불필요한 토큰 제거
        cleaned_preds = clean_text(decoded_preds, remove_tokens)
        cleaned_labels = clean_text(decoded_labels, remove_tokens)

        # 샘플 출력 (디버깅용)
        print('-' * 150)
        for i in range(min(3, len(cleaned_preds))):
            print(f"PRED [{i}]: {cleaned_preds[i]}")
            print(f"GOLD [{i}]: {cleaned_labels[i]}")
            print('-' * 150)

        # ROUGE 점수 계산
        try:
            results = rouge.get_scores(cleaned_preds, cleaned_labels, avg=True)
            # F1 점수만 추출
            result = {key: value["f"] for key, value in results.items()}
        except Exception as e:
            print(f"Error calculating ROUGE scores: {e}")
            # 에러 발생 시 기본값 반환
            result = {
                'rouge-1': 0.0,
                'rouge-2': 0.0,
                'rouge-l': 0.0,
            }

        return result

    return compute_metrics


def clean_text(texts: List[str], remove_tokens: List[str]) -> List[str]:
    """
    텍스트 리스트에서 불필요한 토큰을 제거하고 정리합니다.

    Args:
        texts (List[str]): 정리할 텍스트 리스트
        remove_tokens (List[str]): 제거할 토큰 리스트

    Returns:
        List[str]: 정리된 텍스트 리스트

    Examples:
        >>> texts = ["<s>안녕하세요</s>", "<s>감사합니다</s>"]
        >>> remove_tokens = ["<s>", "</s>"]
        >>> clean_text(texts, remove_tokens)
        ['안녕하세요', '감사합니다']
    """
    cleaned_texts = texts.copy()

    # 각 토큰을 공백으로 대체
    for token in remove_tokens:
        if token:  # None이나 빈 문자열 제외
            cleaned_texts = [text.replace(token, " ") for text in cleaned_texts]

    # 연속된 공백을 하나로 줄이고 양쪽 공백 제거
    cleaned_texts = [" ".join(text.split()) for text in cleaned_texts]

    # 빈 문자열 처리 (ROUGE 계산 시 에러 방지)
    cleaned_texts = [text if text.strip() else "empty" for text in cleaned_texts]

    return cleaned_texts


def calculate_rouge_scores(
    predictions: List[str],
    references: List[str],
    remove_tokens: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    예측과 정답 텍스트 간의 ROUGE 점수를 계산합니다.

    Args:
        predictions (List[str]): 예측 텍스트 리스트
        references (List[str]): 정답 텍스트 리스트
        remove_tokens (Optional[List[str]]): 제거할 토큰 리스트

    Returns:
        Dict[str, Dict[str, float]]: ROUGE-1, ROUGE-2, ROUGE-L 점수 (precision, recall, f1)

    Examples:
        >>> predictions = ["안녕하세요 반갑습니다", "오늘 날씨가 좋아요"]
        >>> references = ["안녕하세요", "날씨가 좋습니다"]
        >>> scores = calculate_rouge_scores(predictions, references)
        >>> print(scores['rouge-1']['f'])
    """
    rouge = Rouge()

    # 텍스트 정리
    if remove_tokens:
        predictions = clean_text(predictions, remove_tokens)
        references = clean_text(references, remove_tokens)

    # ROUGE 점수 계산
    try:
        scores = rouge.get_scores(predictions, references, avg=True)
    except Exception as e:
        print(f"Error calculating ROUGE scores: {e}")
        scores = {
            'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0},
        }

    return scores


def print_rouge_scores(scores: Dict[str, Dict[str, float]]) -> None:
    """
    ROUGE 점수를 보기 좋게 출력합니다.

    Args:
        scores (Dict[str, Dict[str, float]]): ROUGE 점수 딕셔너리

    Examples:
        >>> scores = calculate_rouge_scores(preds, refs)
        >>> print_rouge_scores(scores)
        ╔═══════════╦═══════════╦═══════════╦═══════════╗
        ║  Metric   ║ Precision ║  Recall   ║  F1-Score ║
        ╠═══════════╬═══════════╬═══════════╬═══════════╣
        ║  ROUGE-1  ║   0.8234  ║   0.7891  ║   0.8058  ║
        ║  ROUGE-2  ║   0.6543  ║   0.6210  ║   0.6372  ║
        ║  ROUGE-L  ║   0.7654  ║   0.7321  ║   0.7483  ║
        ╚═══════════╩═══════════╩═══════════╩═══════════╝
    """
    print("\n" + "="*60)
    print("ROUGE Scores")
    print("="*60)
    print(f"{'Metric':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*60)

    for metric_name, values in scores.items():
        metric_display = metric_name.upper()
        print(f"{metric_display:<12} "
              f"{values['p']:<12.4f} "
              f"{values['r']:<12.4f} "
              f"{values['f']:<12.4f}")

    print("="*60 + "\n")
