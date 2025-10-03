"""
LLM 파인튜닝 스크립트 (QLoRA + LoRA + W&B)

6개 모델 (koBART, koT5, Llama-3.2-3B, Qwen3-4B, Qwen2.5-7B, Llama-3-8B)을
대화 요약 태스크로 파인튜닝합니다.

주요 기능:
- Encoder-Decoder (koBART, koT5) 전체 파인튜닝
- Decoder-only LLM (Llama, Qwen) QLoRA 4bit + LoRA 파인튜닝
- 모델별 최적화 설정 (LR, dropout, float type)
- W&B 실시간 로깅
- Mecab ROUGE 평가
- 디스크 관리 (100GB)

사용법:
    python scripts/llm_finetuning.py --config configs/finetune_config.yaml
"""

import os
import gc
import sys
import yaml
import shutil
import random
import argparse
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset

# 프로젝트 모듈 import
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.metrics import calculate_rouge_scores
from src.utils.wandb_logger import WandBLogger

# RTX 3090 최적화: TF32 활성화
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("✅ TF32 활성화 (RTX 3090 최적화)")

# W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed")


def set_seed(seed: int = 42):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"✅ Seed 설정: {seed}")


def check_disk_usage() -> float:
    """현재 디스크 사용량 체크 (GB)"""
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
        except Exception:
            pass
    return total_gb


def cleanup_hf_cache(model_name: Optional[str] = None):
    """HuggingFace 캐시 정리"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_dir.exists():
        return

    if model_name:
        # 모델명을 캐시 디렉토리명으로 변환
        cache_model_name = model_name.replace("/", "--")
        cache_model_dir = cache_dir / f"models--{cache_model_name}"

        if cache_model_dir.exists():
            shutil.rmtree(cache_model_dir)
            print(f"✅ 캐시 삭제 완료: {cache_model_name}")


def load_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """학습/검증 데이터 로드"""
    train_df = pd.read_csv(Path(data_path) / "train.csv")
    dev_df = pd.read_csv(Path(data_path) / "dev.csv")
    print(f"✅ 데이터 로드 완료: Train {len(train_df)}, Dev {len(dev_df)}")
    return train_df, dev_df


def apply_chat_template(
    dialogue: str,
    summary: str,
    template_type: str,
    system_prompt: str,
    tokenizer: Any
) -> Dict[str, str]:
    """
    Chat template 적용 (Llama/Qwen)

    Returns:
        dict: {"input": prompt, "target": summary}
    """
    templates = {
        "llama": {
            "system": "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
            "user": "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>",
            "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n"
        },
        "qwen": {
            "system": "<|im_start|>system\n{}<|im_end|>\n",
            "user": "<|im_start|>user\n{}<|im_end|>\n",
            "assistant": "<|im_start|>assistant\n"
        }
    }

    if template_type not in templates:
        # Encoder-Decoder는 chat template 불필요
        return {"input": dialogue, "target": summary}

    tmpl = templates[template_type]
    prompt = (
        tmpl["system"].format(system_prompt) +
        tmpl["user"].format(f"다음 대화를 요약하세요:\n---\n{dialogue}\n---") +
        tmpl["assistant"]
    )

    return {"input": prompt, "target": summary}


def prepare_dataset(
    df: pd.DataFrame,
    tokenizer: Any,
    model_config: Dict[str, Any],
    config: Dict[str, Any],
    max_input_length: int = 512,
    max_target_length: int = 100
) -> Dataset:
    """데이터셋 준비 (HuggingFace Dataset 형식)"""

    template_type = model_config.get("chat_template_type", None)
    system_prompt = config["data"]["system_prompt"]

    data_dicts = []
    for _, row in df.iterrows():
        dialogue = row["dialogue"]
        summary = row["summary"]

        # Chat template 적용
        templated = apply_chat_template(
            dialogue, summary, template_type, system_prompt, tokenizer
        )

        # Tokenize
        inputs = tokenizer(
            templated["input"],
            max_length=max_input_length,
            truncation=True,
            padding=False
        )

        labels = tokenizer(
            templated["target"],
            max_length=max_target_length,
            truncation=True,
            padding=False
        )

        data_dicts.append({
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"]
        })

    return Dataset.from_list(data_dicts)


def load_model_and_tokenizer(
    model_config: Dict[str, Any],
    config: Dict[str, Any]
) -> Tuple[Any, Any]:
    """모델 및 토크나이저 로드 (QLoRA + PEFT)"""

    model_name = model_config["model_name"]
    model_type = model_config["model_type"]
    use_qlora = model_config.get("use_qlora", False)

    print(f"\n{'='*80}")
    print(f"📥 모델 로딩 중: {model_name}")
    print(f"   타입: {model_type}, QLoRA: {use_qlora}")
    print(f"{'='*80}\n")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Special tokens 추가
    special_tokens = config["tokenizer"]["special_tokens"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Model loading
    if use_qlora:
        # QLoRA 4bit config
        qlora_config = config["qlora"]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qlora_config["load_in_4bit"],
            bnb_4bit_compute_dtype=getattr(torch, qlora_config["bnb_4bit_compute_dtype"]),
            bnb_4bit_quant_type=qlora_config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=qlora_config["bnb_4bit_use_double_quant"]
        )

        # Load model
        if model_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16 if model_config.get("use_bf16", False) else torch.float16,
                trust_remote_code=True
            )
        else:
            raise ValueError(f"QLoRA는 causal_lm만 지원: {model_type}")

        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)

        # LoRA config
        lora_config_dict = config["lora"]
        lora_config = LoraConfig(
            r=lora_config_dict["r"],
            lora_alpha=lora_config_dict["lora_alpha"],
            lora_dropout=model_config.get("lora_dropout", 0.1),
            target_modules=lora_config_dict["target_modules"],
            bias=lora_config_dict["bias"],
            task_type=TaskType.CAUSAL_LM
        )

        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    else:
        # Full fine-tuning (Encoder-Decoder)
        if model_type == "encoder_decoder":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            )
        else:
            raise ValueError(f"Full tuning은 encoder_decoder만 지원: {model_type}")

        # Resize token embeddings
        model.resize_token_embeddings(len(tokenizer))
        model = model.cuda()

    print(f"✅ 모델 로딩 완료")
    return model, tokenizer


def compute_metrics(eval_preds, tokenizer):
    """ROUGE 메트릭 계산 (Mecab)"""
    predictions, labels = eval_preds

    # Decode
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Replace -100 with pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE
    rouge_scores = calculate_rouge_scores(decoded_labels, decoded_preds)

    return {
        "rouge-1": rouge_scores["rouge-1"]["f"],
        "rouge-2": rouge_scores["rouge-2"]["f"],
        "rouge-l": rouge_scores["rouge-l"]["f"],
        "rouge_sum": (
            rouge_scores["rouge-1"]["f"] +
            rouge_scores["rouge-2"]["f"] +
            rouge_scores["rouge-l"]["f"]
        )
    }


def train_model(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    model_config: Dict[str, Any],
    config: Dict[str, Any],
    wandb_logger: WandBLogger
) -> Any:
    """
    모델 학습 (모델 타입별 Trainer 자동 선택)

    - Encoder-Decoder: Seq2SeqTrainer (generation 지원)
    - Causal LM: Trainer (language modeling)
    """

    nickname = model_config["nickname"]
    model_type = model_config["model_type"]
    output_dir = Path(config["general"]["output_base_dir"]) / nickname
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_config = config["training"]

    # 모델별 설정 override
    lr = model_config.get("learning_rate", training_config["learning_rate"])
    batch_size = model_config.get("batch_size", training_config["per_device_train_batch_size"])
    use_bf16 = model_config.get("use_bf16", training_config.get("bf16", False))
    use_fp16 = model_config.get("use_fp16", training_config.get("fp16", True))

    # W&B run name 생성
    run_name = wandb_logger._create_run_name(model_config) if wandb_logger.enabled else f"{nickname}"

    # Encoder-Decoder vs Causal LM에 따라 다른 Trainer 사용
    if model_type == "encoder_decoder":
        # Seq2SeqTrainer (generation 지원)
        args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=training_config["num_train_epochs"],
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            learning_rate=lr,
            warmup_ratio=training_config["warmup_ratio"],
            lr_scheduler_type=training_config["lr_scheduler_type"],
            optim=training_config["optim"],
            adam_beta1=training_config.get("adam_beta1", 0.9),
            adam_beta2=training_config.get("adam_beta2", 0.999),
            max_grad_norm=training_config.get("max_grad_norm", 0.3),
            bf16=use_bf16,
            fp16=use_fp16,
            gradient_checkpointing=training_config.get("gradient_checkpointing", True),
            predict_with_generate=True,
            generation_max_length=training_config.get("generation_max_length", 100),
            generation_num_beams=training_config.get("generation_num_beams", 4),
            save_strategy=training_config.get("save_strategy", "epoch"),
            evaluation_strategy=training_config.get("evaluation_strategy", "epoch"),
            save_total_limit=training_config.get("save_total_limit", 2),
            load_best_model_at_end=training_config.get("load_best_model_at_end", True),
            metric_for_best_model=training_config.get("metric_for_best_model", "rouge_sum"),
            greater_is_better=training_config.get("greater_is_better", True),
            logging_steps=training_config.get("logging_steps", 10),
            logging_first_step=training_config.get("logging_first_step", True),
            report_to="wandb" if wandb_logger.enabled else "none",
            run_name=run_name
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer)
        )

    elif model_type == "causal_lm":
        # Trainer (language modeling)
        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=training_config["num_train_epochs"],
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            learning_rate=lr,
            warmup_ratio=training_config["warmup_ratio"],
            lr_scheduler_type=training_config["lr_scheduler_type"],
            optim=training_config["optim"],
            adam_beta1=training_config.get("adam_beta1", 0.9),
            adam_beta2=training_config.get("adam_beta2", 0.999),
            max_grad_norm=training_config.get("max_grad_norm", 0.3),
            bf16=use_bf16,
            fp16=use_fp16,
            gradient_checkpointing=training_config.get("gradient_checkpointing", True),
            save_strategy=training_config.get("save_strategy", "epoch"),
            evaluation_strategy=training_config.get("evaluation_strategy", "epoch"),
            save_total_limit=training_config.get("save_total_limit", 2),
            load_best_model_at_end=training_config.get("load_best_model_at_end", True),
            metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
            greater_is_better=False,
            logging_steps=training_config.get("logging_steps", 10),
            logging_first_step=training_config.get("logging_first_step", True),
            report_to="wandb" if wandb_logger.enabled else "none",
            run_name=run_name
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Causal LM
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"\n🚀 학습 시작: {nickname}")
    print(f"   Model Type: {model_type}")
    print(f"   Trainer: {'Seq2SeqTrainer' if model_type == 'encoder_decoder' else 'Trainer'}")
    print(f"   LR: {lr}, Batch: {batch_size}, Float: {'bf16' if use_bf16 else 'fp16'}")
    print(f"   Output: {output_dir}\n")

    # Train
    trainer.train()

    # Save
    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print(f"✅ 학습 완료: {final_dir}")

    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"✅ 설정 파일 로드: {args.config}")

    # Seed
    set_seed(42)

    # W&B Logger 초기화
    wandb_logger = WandBLogger(config)
    print(f"✅ W&B Logger 초기화 (enabled={wandb_logger.enabled})")

    # Load data
    data_path = config["general"]["data_path"]
    train_df, dev_df = load_data(data_path)

    # Train each model
    models = config["models"]
    print(f"\n{'='*80}")
    print(f"🚀 모델 파인튜닝 시작: {len(models)}개 모델")
    print(f"{'='*80}\n")

    for idx, model_config in enumerate(models, 1):
        nickname = model_config["nickname"]

        print(f"\n{'#'*80}")
        print(f"# [{idx}/{len(models)}] {nickname}")
        print(f"# Model: {model_config['model_name']}")
        print(f"# Description: {model_config['description']}")
        print(f"{'#'*80}\n")

        # Disk check
        disk_usage = check_disk_usage()
        print(f"💾 현재 디스크 사용량: {disk_usage:.2f} GB")

        if disk_usage > config["general"]["max_disk_usage_gb"]:
            print(f"⚠️  디스크 용량 초과! 캐시 정리 중...")
            cleanup_hf_cache()

        try:
            # W&B Run 초기화 (모델별)
            wandb_logger.init_run(model_config)

            # Load model
            model, tokenizer = load_model_and_tokenizer(model_config, config)

            # Prepare datasets
            print("📊 데이터셋 준비 중...")
            train_dataset = prepare_dataset(
                train_df, tokenizer, model_config, config,
                max_input_length=config["tokenizer"]["encoder_max_len"],
                max_target_length=config["tokenizer"]["decoder_max_len"]
            )
            eval_dataset = prepare_dataset(
                dev_df, tokenizer, model_config, config,
                max_input_length=config["tokenizer"]["encoder_max_len"],
                max_target_length=config["tokenizer"]["decoder_max_len"]
            )
            print(f"✅ 데이터셋 준비 완료: Train {len(train_dataset)}, Eval {len(eval_dataset)}")

            # Train
            trainer = train_model(
                model, tokenizer, train_dataset, eval_dataset,
                model_config, config, wandb_logger
            )

            # W&B Run 종료
            wandb_logger.finish()

            # Cleanup
            del model
            del tokenizer
            del trainer
            gc.collect()
            torch.cuda.empty_cache()

            # Cleanup cache
            if config.get("disk_management", {}).get("auto_cleanup_cache", False):
                cleanup_hf_cache(model_config["model_name"])

            disk_usage_after = check_disk_usage()
            print(f"💾 정리 후 디스크 사용량: {disk_usage_after:.2f} GB\n")

        except Exception as e:
            print(f"❌ {nickname} 학습 실패: {e}")
            import traceback
            traceback.print_exc()

            # 실패 시에도 W&B Run 종료
            wandb_logger.finish()
            continue

    print(f"\n{'='*80}")
    print("🎉 모든 모델 파인튜닝 완료!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()