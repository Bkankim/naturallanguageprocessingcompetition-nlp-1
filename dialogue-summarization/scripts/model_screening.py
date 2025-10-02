"""
모델 스크리닝 스크립트 (QLoRA 4bit + W&B)

여러 대형 언어 모델을 QLoRA 4bit 양자화로 로드하여
Dev set에서 zero-shot 성능을 평가합니다.

주요 기능:
- QLoRA 4bit 로딩 (BitsAndBytes)
- 순차 실행 + 자동 캐시 삭제
- 디스크 80GB 관리
- W&B 실시간 로깅
- ROUGE 메트릭 평가

사용법:
    python scripts/model_screening.py --config configs/screening_config.yaml
"""

import os
import gc
import sys
import yaml
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# 프로젝트 모듈 import
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessor import Preprocess
from src.data.dataset import DatasetForInference
from src.evaluation.metrics import calculate_rouge_scores

# W&B import (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Install with: pip install wandb")


def check_disk_usage() -> float:
    """
    현재 디스크 사용량을 체크합니다 (GB 단위).

    Returns:
        float: 현재 디스크 사용량 (GB)
    """
    # /Competition, /opt, /data, /root 디렉토리 크기 합산
    total_gb = 0.0

    for directory in ['/Competition', '/opt', '/data', '/root']:
        try:
            result = subprocess.run(
                ['du', '-sb', directory],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                size_bytes = int(result.stdout.split()[0])
                total_gb += size_bytes / (1024**3)
        except Exception as e:
            print(f"⚠️  {directory} 크기 측정 실패: {e}")

    return total_gb


def cleanup_hf_cache(model_name: Optional[str] = None):
    """
    HuggingFace 캐시를 정리합니다.

    Args:
        model_name: 특정 모델 캐시만 삭제 (None이면 전체 삭제하지 않음)
    """
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    if not cache_dir.exists():
        return

    if model_name:
        # 특정 모델 캐시만 삭제
        model_cache_pattern = model_name.replace("/", "--")
        for cache_path in cache_dir.glob(f"models--{model_cache_pattern}*"):
            try:
                shutil.rmtree(cache_path)
                print(f"✅ 캐시 삭제 완료: {cache_path.name}")
            except Exception as e:
                print(f"⚠️  캐시 삭제 실패: {e}")

    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    gc.collect()


def load_config(config_path: str) -> Dict[str, Any]:
    """YAML 설정 파일을 로드합니다."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def setup_bnb_config(config: Dict[str, Any]) -> BitsAndBytesConfig:
    """
    BitsAndBytes 4bit 양자화 설정을 생성합니다.

    Args:
        config: 설정 딕셔너리

    Returns:
        BitsAndBytesConfig: 4bit 양자화 설정
    """
    qlora_config = config['qlora']

    return BitsAndBytesConfig(
        load_in_4bit=qlora_config['load_in_4bit'],
        bnb_4bit_compute_dtype=getattr(torch, qlora_config['bnb_4bit_compute_dtype']),
        bnb_4bit_quant_type=qlora_config['bnb_4bit_quant_type'],
        bnb_4bit_use_double_quant=qlora_config['bnb_4bit_use_double_quant'],
    )


def load_model_and_tokenizer(
    model_name: str,
    bnb_config: Optional[BitsAndBytesConfig],
    special_tokens: List[str],
    device: torch.device
) -> Tuple[Any, Any]:
    """
    모델과 토크나이저를 로드합니다.

    Args:
        model_name: HuggingFace 모델명
        bnb_config: BitsAndBytes 설정 (None이면 일반 로딩)
        special_tokens: 추가할 special tokens
        device: 디바이스

    Returns:
        Tuple[model, tokenizer, model_type]
    """
    print(f"\n{'='*80}")
    print(f"📥 모델 로딩 중: {model_name}")
    print(f"{'='*80}")

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Special tokens 추가
    if special_tokens:
        tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens
        })

    # 모델 로딩 설정
    load_kwargs = {
        "trust_remote_code": True
    }

    if bnb_config:
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
        quant_info = "4bit 양자화"
    else:
        load_kwargs["torch_dtype"] = torch.float16
        quant_info = "FP16"

    # 모델 타입 감지 (Seq2Seq vs CausalLM)
    try:
        # Seq2Seq 모델 시도 (BART, T5 등)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        model_type = "seq2seq"
    except:
        # CausalLM 모델 (GPT, Llama, Qwen 등)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        model_type = "causal"

    # GPU로 이동 (4bit이 아닌 경우)
    if not bnb_config:
        model = model.to(device)

    # Tokenizer resize
    model.resize_token_embeddings(len(tokenizer))

    print(f"✅ 모델 로딩 완료 (타입: {model_type}, {quant_info})")
    print(f"   파라미터 수: {model.num_parameters() / 1e9:.2f}B")

    return model, tokenizer, model_type


def prepare_dev_dataset(
    config: Dict[str, Any],
    tokenizer: Any
) -> Tuple[pd.DataFrame, DataLoader]:
    """
    Dev 데이터셋을 준비합니다.

    Returns:
        Tuple[dev_data, dataloader]
    """
    # 전처리기 초기화
    preprocessor = Preprocess(
        bos_token=config['tokenizer']['bos_token'],
        eos_token=config['tokenizer']['eos_token']
    )

    # Dev 데이터 로드 (validation이므로 is_train=True로 summary 포함)
    dev_file_path = os.path.join(config['general']['data_path'], 'dev.csv')
    dev_data = preprocessor.make_set_as_df(dev_file_path, is_train=True)

    # 인코더 입력 생성
    encoder_input_dev, _ = preprocessor.make_input(dev_data, is_test=True)

    # 토크나이징
    tokenized_encoder_inputs = tokenizer(
        encoder_input_dev,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['encoder_max_len'],
        return_token_type_ids=False,
    )

    # 추론용 데이터셋 생성
    dev_dataset = DatasetForInference(
        tokenized_encoder_inputs,
        dev_data['fname'].tolist(),
        len(encoder_input_dev)
    )

    # 데이터로더 생성
    dataloader = DataLoader(
        dev_dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False
    )

    print(f"✅ Dev 데이터셋 준비 완료: {len(dev_data)} samples")

    return dev_data, dataloader


def generate_summaries(
    model: Any,
    model_type: str,
    tokenizer: Any,
    dataloader: DataLoader,
    config: Dict[str, Any],
    device: torch.device
) -> List[str]:
    """
    모델로 요약문을 생성합니다.

    Returns:
        List[str]: 생성된 요약문 리스트
    """
    model.eval()
    summaries = []

    print("\n🔮 요약 생성 중...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="생성"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            if model_type == "seq2seq":
                # Seq2Seq 모델 (BART, T5)
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                    early_stopping=config['inference']['early_stopping'],
                    max_length=config['inference']['generate_max_length'],
                    num_beams=config['inference']['num_beams'],
                )
            else:
                # CausalLM 모델 (Llama, Qwen, SOLAR)
                # Prompt 구성 필요
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=config['inference']['generate_max_length'],
                    num_beams=config['inference']['num_beams'],
                    no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                    early_stopping=config['inference']['early_stopping'],
                )
                # Input 부분 제거 (CausalLM은 input을 포함하여 생성)
                generated_ids = generated_ids[:, input_ids.shape[1]:]

            # 디코딩
            for ids in generated_ids:
                result = tokenizer.decode(ids, skip_special_tokens=False)
                summaries.append(result)

    print(f"✅ 요약 생성 완료: {len(summaries)}개")

    return summaries


def evaluate_model(
    model_name: str,
    nickname: str,
    config: Dict[str, Any],
    wandb_run: Optional[Any] = None
) -> Dict[str, Any]:
    """
    단일 모델을 평가합니다.

    Returns:
        Dict: 평가 결과 {model_name, rouge-1, rouge-2, rouge-l, rouge_sum}
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 디스크 체크
    if config['disk_management']['check_before_download']:
        disk_usage_before = check_disk_usage()
        print(f"\n💾 현재 디스크 사용량: {disk_usage_before:.2f} GB")

        if disk_usage_before > config['general']['max_disk_usage_gb']:
            raise RuntimeError(
                f"디스크 사용량 초과! "
                f"{disk_usage_before:.2f}GB > {config['general']['max_disk_usage_gb']}GB"
            )

    try:
        # BitsAndBytes 설정 (4bit이 활성화된 경우에만)
        bnb_config = None
        if config['qlora']['load_in_4bit']:
            bnb_config = setup_bnb_config(config)

        # 모델 및 토크나이저 로드
        model, tokenizer, model_type = load_model_and_tokenizer(
            model_name=model_name,
            bnb_config=bnb_config,
            special_tokens=config['tokenizer']['special_tokens'],
            device=device
        )

        # Dev 데이터셋 준비
        dev_data, dataloader = prepare_dev_dataset(config, tokenizer)

        # 요약 생성
        summaries = generate_summaries(
            model=model,
            model_type=model_type,
            tokenizer=tokenizer,
            dataloader=dataloader,
            config=config,
            device=device
        )

        # 후처리: 불필요한 토큰 제거
        from src.evaluation.metrics import clean_text
        remove_tokens = config['inference']['remove_tokens']
        cleaned_summaries = clean_text(summaries, remove_tokens)

        # ROUGE 평가
        print("\n📊 ROUGE 평가 중...")
        references = dev_data['summary'].tolist()

        rouge_scores = calculate_rouge_scores(
            predictions=cleaned_summaries,
            references=references,
            remove_tokens=remove_tokens
        )

        # 점수 추출
        rouge_1 = rouge_scores['rouge-1']['f'] * 100
        rouge_2 = rouge_scores['rouge-2']['f'] * 100
        rouge_l = rouge_scores['rouge-l']['f'] * 100
        rouge_sum = rouge_1 + rouge_2 + rouge_l

        print(f"\n{'='*80}")
        print(f"✅ {nickname} 평가 완료")
        print(f"{'='*80}")
        print(f"ROUGE-1: {rouge_1:.2f}")
        print(f"ROUGE-2: {rouge_2:.2f}")
        print(f"ROUGE-L: {rouge_l:.2f}")
        print(f"ROUGE Sum: {rouge_sum:.2f}")
        print(f"{'='*80}\n")

        # W&B 로깅
        if wandb_run:
            wandb_run.log({
                f"{nickname}/rouge-1": rouge_1,
                f"{nickname}/rouge-2": rouge_2,
                f"{nickname}/rouge-l": rouge_l,
                f"{nickname}/rouge_sum": rouge_sum,
            })

        result = {
            'model_name': model_name,
            'nickname': nickname,
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'rouge_sum': rouge_sum,
            'status': 'success'
        }

        # 메모리 정리
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        return result

    except Exception as e:
        print(f"\n❌ {nickname} 평가 실패: {e}")

        result = {
            'model_name': model_name,
            'nickname': nickname,
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
            'rouge_sum': 0.0,
            'status': f'failed: {str(e)}'
        }

        return result

    finally:
        # 캐시 정리
        if config['disk_management']['auto_cleanup_cache']:
            cleanup_hf_cache(model_name)

        # 디스크 체크
        if config['disk_management']['check_after_inference']:
            disk_usage_after = check_disk_usage()
            print(f"💾 정리 후 디스크 사용량: {disk_usage_after:.2f} GB\n")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="모델 스크리닝 스크립트")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="설정 YAML 파일 경로"
    )
    args = parser.parse_args()

    # 설정 로드
    config = load_config(args.config)
    print(f"✅ 설정 파일 로드: {args.config}")

    # 결과 디렉토리 생성
    result_path = Path(config['general']['result_path'])
    result_path.mkdir(parents=True, exist_ok=True)

    # W&B 초기화
    wandb_run = None
    if config['wandb']['enabled'] and WANDB_AVAILABLE:
        run_name = f"{config['wandb']['name_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb_run = wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            name=run_name,
            tags=config['wandb']['tags'],
            config=config
        )
        print("✅ W&B 초기화 완료")

    # 모델 스크리닝
    results = []
    models = config['models']

    print(f"\n{'='*80}")
    print(f"🚀 모델 스크리닝 시작: {len(models)}개 모델")
    print(f"{'='*80}\n")

    for i, model_info in enumerate(models, 1):
        print(f"\n{'#'*80}")
        print(f"# [{i}/{len(models)}] {model_info['nickname']}")
        print(f"# Model: {model_info['model_name']}")
        print(f"# Description: {model_info['description']}")
        print(f"{'#'*80}\n")

        result = evaluate_model(
            model_name=model_info['model_name'],
            nickname=model_info['nickname'],
            config=config,
            wandb_run=wandb_run
        )

        results.append(result)

    # 결과 저장
    results_df = pd.DataFrame(results)
    csv_path = result_path / f"screening_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(csv_path, index=False)

    print(f"\n{'='*80}")
    print(f"🎉 모델 스크리닝 완료!")
    print(f"{'='*80}\n")
    print(results_df.to_string(index=False))
    print(f"\n💾 결과 저장: {csv_path}")

    # W&B에 테이블 로깅
    if wandb_run:
        wandb_run.log({"screening_results": wandb.Table(dataframe=results_df)})
        wandb_run.finish()

    # 최고 성능 모델 출력
    best_model = results_df.loc[results_df['rouge_sum'].idxmax()]
    print(f"\n🏆 최고 성능 모델: {best_model['nickname']}")
    print(f"   ROUGE Sum: {best_model['rouge_sum']:.2f}")
    print(f"   (R1: {best_model['rouge-1']:.2f}, R2: {best_model['rouge-2']:.2f}, RL: {best_model['rouge-l']:.2f})")


if __name__ == "__main__":
    main()
