"""
KoBART Checkpoint 추론 테스트 스크립트

목적: koBART fine-tuned 모델로 추론 및 ROUGE 평가
"""
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from src.evaluation.metrics import calculate_rouge_scores

def load_model_and_tokenizer(checkpoint_path: str):
    """모델과 토크나이저 로드"""
    print(f"📥 모델 로딩: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

    # GPU 사용
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"✅ 모델 로딩 완료 (device: {device})")
    return model, tokenizer, device


def generate_summaries(
    model,
    tokenizer,
    dialogues: list,
    device: str,
    batch_size: int = 4,
    max_length: int = 512,
    max_new_tokens: int = 100,
    num_beams: int = 4
):
    """배치 추론으로 요약 생성"""
    summaries = []

    print(f"\n🔄 추론 시작 (batch_size={batch_size}, num_beams={num_beams})")

    for i in tqdm(range(0, len(dialogues), batch_size), desc="Generating"):
        batch = dialogues[i:i+batch_size]

        # Tokenize
        inputs = tokenizer(
            batch,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        # BART는 token_type_ids를 사용하지 않으므로 제거
        inputs = {k: v.to(device) for k, v in inputs.items() if k != 'token_type_ids'}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2
            )

        # Decode (skip_special_tokens=False로 모든 토큰 포함)
        batch_summaries = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=False  # Baseline과 동일하게 False
        )
        summaries.extend(batch_summaries)

    return summaries


def clean_predictions(predictions: list, tokenizer) -> list:
    """생성 결과에서 불필요한 모델 토큰 제거 (Baseline 방식)"""
    # 제거할 토큰: 모델 관련 토큰만 (데이터 토큰은 유지)
    remove_tokens = [
        '<usr>',  # KoBART specific
        tokenizer.bos_token,  # <s>
        tokenizer.eos_token,  # </s>
        tokenizer.pad_token,  # <pad>
    ]

    cleaned = predictions.copy()
    for token in remove_tokens:
        if token:  # None 체크
            cleaned = [s.replace(token, " ") for s in cleaned]

    return cleaned


def evaluate_rouge(predictions: list, references: list):
    """ROUGE 점수 계산 (Mecab)"""
    print("\n📊 ROUGE 평가 중 (Mecab tokenization)...")

    # Mecab 토크나이제이션으로 ROUGE 계산
    scores = calculate_rouge_scores(
        predictions=predictions,
        references=references,
        tokenization_mode='mecab'
    )

    return scores


def main():
    """메인 실행 함수"""
    print("="*80)
    print("KoBART 추론 테스트")
    print("="*80)

    # 경로 설정
    checkpoint_path = project_root / "checkpoints/llm_finetuning/koBART-summarization/final_model"
    data_path = project_root.parent / "data/dev.csv"

    # 체크포인트 존재 확인
    if not checkpoint_path.exists():
        print(f"❌ 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        return

    print(f"\n📂 체크포인트: {checkpoint_path}")
    print(f"📂 데이터: {data_path}")

    # 1. 모델 로드
    model, tokenizer, device = load_model_and_tokenizer(str(checkpoint_path))

    # 2. 데이터 로드
    print(f"\n📥 데이터 로딩: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ 데이터 로드 완료: {len(df)}개")

    # 작은 샘플로 먼저 테스트 (50개)
    n_samples = min(50, len(df))
    df_sample = df.head(n_samples)

    print(f"\n🧪 테스트 샘플: {n_samples}개")

    dialogues = df_sample['dialogue'].tolist()
    references = df_sample['summary'].tolist()

    # 3. 추론
    predictions = generate_summaries(
        model=model,
        tokenizer=tokenizer,
        dialogues=dialogues,
        device=device,
        batch_size=4,
        max_new_tokens=100,
        num_beams=4
    )

    # 3-1. 토큰 정제 (Baseline 방식)
    print("\n🧹 모델 토큰 제거 중...")
    predictions_cleaned = clean_predictions(predictions, tokenizer)

    # 4. ROUGE 평가
    scores = evaluate_rouge(predictions_cleaned, references)

    # 5. 결과 출력
    print("\n" + "="*80)
    print("📈 ROUGE 점수 (Mecab tokenization)")
    print("="*80)
    print(f"ROUGE-1 F1: {scores['rouge-1']['f']:.4f}")
    print(f"ROUGE-2 F1: {scores['rouge-2']['f']:.4f}")
    print(f"ROUGE-L F1: {scores['rouge-l']['f']:.4f}")
    print(f"ROUGE SUM:  {scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f']:.4f}")
    print("="*80)

    # 6. 샘플 출력
    print("\n" + "="*80)
    print("📝 샘플 예측 (처음 3개)")
    print("="*80)
    for i in range(min(3, len(predictions_cleaned))):
        print(f"\n[샘플 {i+1}]")
        print(f"대화: {dialogues[i][:100]}...")
        print(f"정답: {references[i]}")
        print(f"예측 (원본): {predictions[i]}")
        print(f"예측 (정제): {predictions_cleaned[i]}")

    print("\n✅ 추론 테스트 완료!")

    # 메모리 정리
    del model
    del tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()