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
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset

# TRL import (SFTTrainer)
try:
    from trl import SFTTrainer, SFTConfig
    TRL_AVAILABLE = True
    print("✅ TRL 사용 가능")
except ImportError:
    TRL_AVAILABLE = False
    print("⚠️  TRL not installed - SFTTrainer 사용 불가")

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


def check_disk_usage(critical_limit_gb: float = 150.0) -> float:
    """
    전체 루트 디스크 사용량 체크 (GB)

    ⚠️  CRITICAL: 150GB 초과 시 서버 초기화됨!

    Args:
        critical_limit_gb: 임계값 (기본 150GB)

    Returns:
        현재 디스크 사용량 (GB)

    Raises:
        RuntimeError: 150GB 초과 시
    """
    try:
        # du -sh / 2>/dev/null 명령 실행
        result = subprocess.run(
            ['du', '-sh', '/'],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # 2>/dev/null
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            # "111G\t/" 형식에서 숫자만 추출
            size_str = result.stdout.split()[0]
            # G, M, K 등의 단위 파싱
            if size_str.endswith('G'):
                total_gb = float(size_str[:-1])
            elif size_str.endswith('M'):
                total_gb = float(size_str[:-1]) / 1024
            elif size_str.endswith('K'):
                total_gb = float(size_str[:-1]) / (1024**2)
            elif size_str.endswith('T'):
                total_gb = float(size_str[:-1]) * 1024
            else:
                # 단위 없으면 bytes로 간주
                total_gb = float(size_str) / (1024**3)

            # ⚠️ CRITICAL: 150GB 체크
            if total_gb >= critical_limit_gb:
                raise RuntimeError(
                    f"🚨 CRITICAL: 디스크 사용량 {total_gb:.1f}GB >= {critical_limit_gb}GB! "
                    f"서버 초기화 위험! 즉시 중단합니다."
                )

            # 경고: 140GB 이상
            if total_gb >= 140.0:
                print(f"⚠️  WARNING: 디스크 사용량 {total_gb:.1f}GB (한계까지 {critical_limit_gb - total_gb:.1f}GB)")

            return total_gb
        else:
            print(f"⚠️  디스크 체크 실패 (returncode={result.returncode})")
            print(f"⚠️  실제 디스크 사용량 확인 불가 - 수동 확인 권장: du -sh /")
            return 0.0

    except subprocess.TimeoutExpired:
        print(f"⚠️  디스크 체크 시간 초과 (60초)")
        print(f"⚠️  실제 디스크 사용량 확인 불가 - 수동 확인 권장: du -sh /")
        return 0.0
    except Exception as e:
        print(f"⚠️  디스크 체크 오류: {e}")
        print(f"⚠️  실제 디스크 사용량 확인 불가 - 수동 확인 권장: du -sh /")
        return 0.0


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
    tokenizer: Any,
    for_training: bool = True
) -> Dict[str, str]:
    """
    Chat template 적용 (HuggingFace 공식 API 사용)

    Args:
        for_training: True면 add_generation_prompt=False (훈련용),
                     False면 add_generation_prompt=True (추론용)

    Returns:
        dict: {"input": full_text} (훈련) 또는 {"input": prompt} (추론)
    """
    if template_type not in ("llama", "qwen"):
        # Encoder-Decoder는 chat template 불필요
        return {"input": dialogue, "target": summary}

    if for_training:
        # 훈련: add_generation_prompt=False (정답 포함)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"다음 대화를 요약하세요:\n---\n{dialogue}\n---"},
            {"role": "assistant", "content": summary}  # 정답 포함
        ]
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        # full_text는 이미 system + user + assistant(summary) 전체를 포함
        return {"input": full_text, "target": ""}  # target은 빈 문자열
    else:
        # 추론: add_generation_prompt=True (assistant 턴 시작만)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"다음 대화를 요약하세요:\n---\n{dialogue}\n---"}
            # assistant 제외
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return {"input": prompt, "target": ""}


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

        # Chat template 적용 (훈련용)
        templated = apply_chat_template(
            dialogue, summary, template_type, system_prompt, tokenizer,
            for_training=True
        )

        if template_type:  # Decoder-only (Llama, Qwen)
            # ✅ HuggingFace 공식 API 사용 (add_generation_prompt=False)
            # full_text는 이미 system + user + assistant(summary) 전체를 포함
            full_text = templated["input"]

            # 전체 토크나이즈
            full_ids = tokenizer(
                full_text,
                max_length=max_input_length + max_target_length,
                truncation=True,
                padding=False,
                add_special_tokens=True
            )["input_ids"]

            # Assistant 헤더 위치 찾기 (모델별 헤더 다름)
            # Llama: "<|start_header_id|>assistant<|end_header_id|>\n\n"
            # Qwen: "<|im_start|>assistant\n"
            if template_type == "llama":
                assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            elif template_type == "qwen":
                assistant_header = "<|im_start|>assistant\n"
            else:
                assistant_header = ""

            # Assistant 헤더 토큰 ID 리스트 생성
            assistant_header_ids = tokenizer.encode(
                assistant_header,
                add_special_tokens=False
            )

            # full_ids에서 assistant 헤더 위치 검색
            prompt_length = 0
            for i in range(len(full_ids) - len(assistant_header_ids) + 1):
                if full_ids[i:i+len(assistant_header_ids)] == assistant_header_ids:
                    # Assistant 헤더를 찾음 → 헤더 끝부터 학습 대상
                    prompt_length = i + len(assistant_header_ids)
                    break

            # Labels 계산
            if prompt_length == 0 or prompt_length >= len(full_ids):
                # Assistant 헤더를 못 찾았거나, truncation으로 잘린 경우
                # → 전체를 -100으로 처리 (학습 제외)
                input_ids = full_ids
                labels = [-100] * len(full_ids)
            else:
                # 정상적으로 찾은 경우
                # → Prompt 부분은 -100, Response 부분만 학습
                input_ids = full_ids
                labels = [-100] * prompt_length + full_ids[prompt_length:]

            data_dicts.append({
                "input_ids": input_ids,
                "labels": labels
            })
        else:  # Encoder-Decoder (koBART, koT5)
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

    # Causal LM은 left-padding 필수 (generation 시 올바른 결과 보장)
    if model_type == "causal_lm":
        tokenizer.padding_side = "left"
        # EOS/PAD 동기화
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        # Encoder-Decoder는 right-padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Special tokens 추가
    special_tokens = config["tokenizer"]["special_tokens"]

    # Chat template 토큰 추가 (모델별)
    chat_template_type = model_config.get("chat_template_type")
    if chat_template_type and "chat_template_tokens" in config["tokenizer"]:
        chat_tokens = config["tokenizer"]["chat_template_tokens"].get(chat_template_type, [])
        special_tokens = special_tokens + chat_tokens

    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Model loading
    if use_qlora:
        # QLoRA 4bit config - compute dtype은 모델 dtype과 일치시킴
        qlora_config = config["qlora"]
        # 모델별 dtype에 맞춰 compute dtype 설정 (Llama=bf16, Qwen=fp16)
        compute_dtype = torch.bfloat16 if model_config.get("use_bf16", False) else torch.float16

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qlora_config["load_in_4bit"],
            bnb_4bit_compute_dtype=compute_dtype,
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

        # Resize token embeddings BEFORE prepare_model_for_kbit_training
        # (special tokens 추가했으므로 필수!)
        model.resize_token_embeddings(len(tokenizer))

        # ✅ gradient_checkpointing 사용 시 use_cache는 False여야 함
        # (학습 시 KV cache 불필요 + gradient_checkpointing과 충돌)
        model.config.use_cache = False

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
            # FP32로 로드하고 Trainer가 mixed precision 관리하도록 함
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name
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


def clean_predictions(predictions: List[str], tokenizer: Any) -> List[str]:
    """생성 결과에서 불필요한 모델 토큰 제거 (Baseline 방식)"""
    remove_tokens = [
        '<usr>',
        tokenizer.bos_token,
        tokenizer.eos_token,
        tokenizer.pad_token,
        # Chat template 토큰 (Llama)
        '<|start_header_id|>',
        '<|end_header_id|>',
        '<|eot_id|>',
        # Chat template 토큰 (Qwen)
        '<|im_start|>',
        '<|im_end|>',
        # 역할 토큰
        'system',
        'user',
        'assistant',
    ]

    cleaned = predictions.copy()
    for token in remove_tokens:
        if token:
            cleaned = [s.replace(token, " ") for s in cleaned]

    # 연속된 공백 제거
    cleaned = [" ".join(s.split()) for s in cleaned]
    return cleaned


def run_inference_on_dev(
    model: Any,
    tokenizer: Any,
    dev_df: pd.DataFrame,
    device: str,
    model_config: Dict[str, Any],
    config: Dict[str, Any],
    batch_size: int = 4,
    max_new_tokens: int = 100,
    num_beams: int = 4,
    is_test: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Dev/Test set 추론 실행 (Chat Template 적용)

    Args:
        is_test: True면 Test mode (references 없음), False면 Dev mode

    Returns:
        (predictions, references) 튜플 (Test mode에서는 references는 빈 리스트)
    """
    model.eval()

    dialogues = dev_df['dialogue'].tolist()
    references = dev_df['summary'].tolist() if 'summary' in dev_df.columns else []
    predictions = []

    # Chat template 타입 가져오기
    template_type = model_config.get("chat_template_type", None)
    system_prompt = config["data"]["system_prompt"]

    dataset_name = "Test" if is_test else "Dev"
    print(f"\n🔄 {dataset_name} set 추론 시작 (samples={len(dialogues)}, batch_size={batch_size})")
    if template_type:
        print(f"   ✅ Chat template: {template_type}")
        # Causal LM은 left-padding + left-truncation 필수
        # - left-padding: 배치 생성 시 올바른 결과 보장
        # - left-truncation: assistant 헤더 보존 (긴 대화 시 중요!)
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        print(f"   ✅ Padding/Truncation side: left (assistant 헤더 보존)")

    for i in tqdm(range(0, len(dialogues), batch_size), desc="Inference"):
        batch_dialogues = dialogues[i:i+batch_size]

        # Chat template 적용 (Causal LM만)
        if template_type:
            batch_prompts = []
            for dialogue in batch_dialogues:
                # 추론용: add_generation_prompt=True
                templated = apply_chat_template(
                    dialogue,
                    "",  # summary는 빈 문자열 (생성할 것이므로)
                    template_type,
                    system_prompt,
                    tokenizer,
                    for_training=False  # 추론 모드
                )
                batch_prompts.append(templated["input"])
        else:
            # Encoder-Decoder는 raw dialogue
            batch_prompts = batch_dialogues

        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            max_length=1024,  # 512 → 1024 (prompt truncation 6.81% → 0% 해결)
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        # Remove token_type_ids if present (BART doesn't use it)
        inputs = {k: v.to(device) for k, v in inputs.items() if k != 'token_type_ids'}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=0.9,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode - Decoder-only는 input 제거, Encoder-Decoder는 그대로
        if template_type:
            # Decoder-only: outputs에 input_ids 포함됨 → 생성 부분만 추출
            # outputs shape: [batch_size, total_length]
            # inputs['input_ids'] shape: [batch_size, prompt_length]
            input_length = inputs['input_ids'].shape[1]
            generated_ids = outputs[:, input_length:]  # 생성된 부분만
            batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        else:
            # Encoder-Decoder: outputs는 생성된 부분만 포함
            batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=False)

        predictions.extend(batch_preds)

    # Clean predictions (remove only model tokens, keep data tokens)
    predictions = clean_predictions(predictions, tokenizer)

    print(f"✅ 추론 완료: {len(predictions)}개 생성")
    return predictions, references


class DataCollatorForSupervisedDataset(object):
    """
    Korean_DCS_2024 베이스라인 DataCollator

    SFTTrainer용 Custom DataCollator
    - input_ids와 labels만 사용 (간결함)
    - labels는 -100으로 패딩 (loss 계산 제외)
    - attention_mask 자동 생성
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(lbls) for lbls in labels],
            batch_first=True,
            padding_value=-100
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class GradientClippingCallback(TrainerCallback):
    """
    Gradient Clipping 비율을 추적하는 Callback

    매 로깅 스텝마다:
    - Clipping 발생 여부 추적
    - Clipping 비율 계산 (clipped steps / total steps)
    - W&B에 실시간 로깅
    """

    def __init__(self, max_grad_norm: float):
        self.max_grad_norm = max_grad_norm
        self.total_steps = 0
        self.clipped_steps = 0
        self.grad_norms = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """로깅 시점에 grad_norm 캡처"""
        if logs and "grad_norm" in logs:
            grad_norm = logs["grad_norm"]
            self.grad_norms.append(grad_norm)
            self.total_steps += 1

            # Clipping 발생 여부
            if grad_norm > self.max_grad_norm:
                self.clipped_steps += 1

            # Clipping 비율 계산
            clip_ratio = self.clipped_steps / self.total_steps if self.total_steps > 0 else 0.0

            # 추가 메트릭 로깅
            logs["grad_clip_ratio"] = clip_ratio
            logs["grad_clip_count"] = self.clipped_steps
            logs["grad_norm_mean"] = sum(self.grad_norms) / len(self.grad_norms) if self.grad_norms else 0.0


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

    # Gradient Clipping Callback 생성 (config에서 max_grad_norm 읽기)
    max_grad_norm = training_config.get("max_grad_norm", 1.0)
    grad_clip_callback = GradientClippingCallback(max_grad_norm)

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
            eval_strategy=training_config.get("evaluation_strategy", "epoch"),
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
            compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
            callbacks=[grad_clip_callback]
        )

    elif model_type == "causal_lm":
        # SFTTrainer (Korean_DCS_2024 베이스라인)
        if not TRL_AVAILABLE:
            raise ImportError("TRL not installed. Run: pip install trl")

        # gradient_checkpointing_kwargs 가져오기
        gradient_checkpointing_kwargs = training_config.get("gradient_checkpointing_kwargs", {"use_reentrant": False})

        args = SFTConfig(
            # 기본 설정 (Korean_DCS_2024 베이스라인)
            output_dir=str(output_dir),
            overwrite_output_dir=True,  # 출력 디렉토리 덮어쓰기
            do_train=True,
            do_eval=True,
            # 학습 설정
            num_train_epochs=training_config["num_train_epochs"],
            max_steps=-1,  # Epoch 기반 학습 (-1 = 무제한)
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            # 옵티마이저
            learning_rate=lr,
            warmup_ratio=training_config["warmup_ratio"],
            lr_scheduler_type=training_config["lr_scheduler_type"],
            weight_decay=training_config.get("weight_decay", 0.1),
            optim=training_config["optim"],
            adam_beta1=training_config.get("adam_beta1", 0.9),
            adam_beta2=training_config.get("adam_beta2", 0.999),
            max_grad_norm=training_config.get("max_grad_norm", 1.2),
            # Float precision
            bf16=use_bf16,
            fp16=use_fp16,
            # Gradient checkpointing
            gradient_checkpointing=training_config.get("gradient_checkpointing", True),
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            # 저장 & 평가
            save_strategy=training_config.get("save_strategy", "epoch"),
            eval_strategy=training_config.get("evaluation_strategy", "epoch"),
            save_total_limit=training_config.get("save_total_limit", 2),
            load_best_model_at_end=training_config.get("load_best_model_at_end", True),
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # 로깅
            log_level="info",
            logging_steps=training_config.get("logging_steps", 10),
            logging_first_step=training_config.get("logging_first_step", True),
            report_to="wandb" if wandb_logger.enabled else "none",
            run_name=run_name,
            # SFT 특화 파라미터 (TRL 0.23.1 호환)
            max_length=config["tokenizer"].get("encoder_max_len", 1024),  # TRL 0.23.1: max_length (0.9.4: max_seq_length)
            packing=True,  # 효율성 향상 (2-3x speedup)
            seed=42
        )

        # Korean_DCS_2024 DataCollator
        data_collator = DataCollatorForSupervisedDataset(tokenizer)

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,  # TRL 0.23.1: processing_class (0.9.4: tokenizer)
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[grad_clip_callback]
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"\n🚀 학습 시작: {nickname}")
    print(f"   Model Type: {model_type}")
    print(f"   Trainer: {'Seq2SeqTrainer' if model_type == 'encoder_decoder' else 'SFTTrainer (packing=True)'}")
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

            # 1. Train
            trainer = train_model(
                model, tokenizer, train_dataset, eval_dataset,
                model_config, config, wandb_logger
            )

            # 2. Inference on Test set
            print(f"\n{'='*80}")
            print(f"📊 Test Set 추론 시작: {nickname}")
            print(f"{'='*80}\n")

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Test 데이터 로드
            test_file = Path(config["general"]["data_path"]) / "test.csv"
            if not test_file.exists():
                print(f"⚠️  Test 파일 없음: {test_file}")
                print(f"   Test 추론을 건너뛰고 다음 모델로 진행합니다.\n")
                wandb_logger.finish()
                continue

            test_df = pd.read_csv(test_file)
            print(f"✅ Test 데이터 로드: {len(test_df)}개 샘플\n")

            # Test set 추론 (references 없음)
            predictions, _ = run_inference_on_dev(
                model, tokenizer, test_df, device,
                model_config, config,
                batch_size=4,
                max_new_tokens=150,
                num_beams=4,
                is_test=True  # Test mode
            )

            # 3. Save Submission CSV
            submission_dir = Path(config["general"]["output_base_dir"]) / "submissions"
            submission_dir.mkdir(parents=True, exist_ok=True)
            submission_file = submission_dir / f"{nickname}_submission.csv"

            submission_df = pd.DataFrame({
                "fname": test_df['fname'].tolist(),
                "summary": predictions
            })
            submission_df.to_csv(submission_file, index=False)

            print(f"\n{'='*80}")
            print(f"💾 Submission 파일 저장 완료")
            print(f"{'='*80}")
            print(f"경로: {submission_file}")
            print(f"샘플 수: {len(submission_df)}")
            print(f"\n샘플 (처음 3개):")
            print(submission_df.head(3))
            print(f"{'='*80}\n")

            # W&B Run 종료
            wandb_logger.finish()

            # 5. Cleanup - 체크포인트 삭제 (Config 설정에 따라)
            if config["disk_management"]["cleanup_old_checkpoints"]:
                print(f"\n🗑️  체크포인트 삭제 중...")
                checkpoint_dir = Path(config["general"]["output_base_dir"]) / nickname
                if checkpoint_dir.exists():
                    size_before = sum(f.stat().st_size for f in checkpoint_dir.rglob('*') if f.is_file()) / (1024**3)
                    shutil.rmtree(checkpoint_dir)
                    print(f"✅ 체크포인트 삭제 완료: {nickname} ({size_before:.2f}GB 확보)")
            else:
                print(f"\n💾 체크포인트 보존 모드: {nickname} 삭제 건너뛰기")

            # Cleanup model & memory
            del model
            del tokenizer
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            print("✅ 메모리 정리 완료")

            # Cleanup model cache
            print(f"\n🗑️  모델 캐시 삭제 중...")
            cleanup_hf_cache(model_config["model_name"])

            # Final disk check
            disk_usage_after = check_disk_usage()
            print(f"💾 최종 디스크 사용량: {disk_usage_after:.2f} GB\n")

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