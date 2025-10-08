"""
Llama-3.2-Korean-3B Left-padding 추론 재실행 스크립트

저장된 체크포인트를 로드하여 Dev set 추론만 실행합니다.
"""

import sys
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 프로젝트 모듈
sys.path.append(str(Path(__file__).parent.parent))
from src.evaluation.metrics import calculate_rouge_scores


def clean_predictions(predictions: List[str], tokenizer: Any) -> List[str]:
    """생성 결과에서 불필요한 모델 토큰 제거"""
    remove_tokens = [
        '<usr>',
        tokenizer.bos_token,
        tokenizer.eos_token,
        tokenizer.pad_token,
    ]

    cleaned = predictions.copy()
    for token in remove_tokens:
        if token:
            cleaned = [s.replace(token, " ") for s in cleaned]

    # 연속된 공백 제거
    cleaned = [" ".join(s.split()) for s in cleaned]
    return cleaned


def run_inference(
    model: Any,
    tokenizer: Any,
    dev_df: pd.DataFrame,
    device: str,
    batch_size: int = 4,
    max_new_tokens: int = 100,
    num_beams: int = 4
) -> Tuple[List[str], List[str]]:
    """Dev set 추론 실행"""
    model.eval()

    dialogues = dev_df['dialogue'].tolist()
    references = dev_df['summary'].tolist()
    predictions = []

    print(f"\n🔄 Dev set 추론 시작 (samples={len(dialogues)}, batch_size={batch_size})")
    print(f"   Padding side: {tokenizer.padding_side}")

    for i in tqdm(range(0, len(dialogues), batch_size), desc="Inference"):
        batch = dialogues[i:i+batch_size]

        # Tokenize
        inputs = tokenizer(
            batch,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

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

        # Decode
        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        predictions.extend(batch_preds)

    # Clean predictions
    predictions = clean_predictions(predictions, tokenizer)

    print(f"✅ 추론 완료: {len(predictions)}개 생성")
    return predictions, references


def main():
    # 경로
    base_model = "Bllossom/llama-3.2-Korean-Bllossom-3B"
    checkpoint_dir = Path("/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/dialogue-summarization/checkpoints/llm_finetuning/Llama-3.2-Korean-3B/final_model")
    data_path = Path("/Competition/NLP/data/dev.csv")

    print(f"{'='*80}")
    print(f"📥 모델 로딩 중...")
    print(f"   Base: {base_model}")
    print(f"   Checkpoint: {checkpoint_dir}")
    print(f"{'='*80}\n")

    # Tokenizer (Left-padding 설정!)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ✅ Left-padding 설정 (Causal LM 필수)
    tokenizer.padding_side = "left"
    print(f"✅ Tokenizer padding_side: {tokenizer.padding_side}")

    # Model (QLoRA PEFT)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Load PEFT adapter
    model = PeftModel.from_pretrained(model, str(checkpoint_dir))
    model = model.merge_and_unload()  # Merge LoRA weights
    print(f"✅ 모델 로딩 완료\n")

    # Load dev data
    dev_df = pd.read_csv(data_path)
    print(f"✅ Dev 데이터 로드: {len(dev_df)}개\n")

    # Inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions, references = run_inference(
        model, tokenizer, dev_df, device,
        batch_size=4,
        max_new_tokens=100,
        num_beams=4
    )

    # ROUGE 평가
    print("\n📈 ROUGE 평가 중 (Mecab tokenization)...")
    rouge_scores = calculate_rouge_scores(
        predictions, references,
        tokenization_mode='mecab'
    )

    # 결과 출력
    print(f"\n{'='*80}")
    print(f"📊 Llama-3.2-Korean-3B (Left-padding) ROUGE 점수")
    print(f"{'='*80}")
    print(f"ROUGE-1 F1: {rouge_scores['rouge-1']['f']:.4f}")
    print(f"ROUGE-2 F1: {rouge_scores['rouge-2']['f']:.4f}")
    print(f"ROUGE-L F1: {rouge_scores['rouge-l']['f']:.4f}")
    rouge_sum = (rouge_scores['rouge-1']['f'] +
                rouge_scores['rouge-2']['f'] +
                rouge_scores['rouge-l']['f'])
    print(f"ROUGE SUM:  {rouge_sum:.4f}")
    print(f"{'='*80}\n")

    # 결과 저장
    results_file = Path("/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/dialogue-summarization/checkpoints/llm_finetuning/Llama-3.2-Korean-3B_left_padding_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Llama-3.2-Korean-3B (Left-padding) ROUGE Results\n")
        f.write(f"{'='*80}\n")
        f.write(f"ROUGE-1 F1: {rouge_scores['rouge-1']['f']:.4f}\n")
        f.write(f"ROUGE-2 F1: {rouge_scores['rouge-2']['f']:.4f}\n")
        f.write(f"ROUGE-L F1: {rouge_scores['rouge-l']['f']:.4f}\n")
        f.write(f"ROUGE SUM:  {rouge_sum:.4f}\n")

    print(f"✅ 결과 저장: {results_file}")


if __name__ == "__main__":
    main()