"""
VRAM 사용량 측정 스크립트

각 모델의 메모리 사용량을 측정하여 최적 배치 사이즈를 결정합니다.
"""

import torch
import yaml
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# GPU 메모리 측정 함수
def get_gpu_memory():
    """현재 GPU 메모리 사용량 반환 (GB)"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        return allocated, reserved
    return 0, 0

def test_model_memory(model_name, chat_template_type, use_bf16=True):
    """모델 로드 후 메모리 사용량 측정"""

    print(f"\n{'='*80}")
    print(f"📊 메모리 측정: {model_name}")
    print(f"{'='*80}\n")

    # 초기 메모리
    torch.cuda.empty_cache()
    initial_allocated, initial_reserved = get_gpu_memory()
    print(f"초기 메모리: Allocated={initial_allocated:.2f}GB, Reserved={initial_reserved:.2f}GB")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # QLoRA config
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    # 모델 로드
    print("모델 로딩 중...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=True
    )

    after_load_allocated, after_load_reserved = get_gpu_memory()
    print(f"모델 로드 후: Allocated={after_load_allocated:.2f}GB, Reserved={after_load_reserved:.2f}GB")
    print(f"   증가량: {after_load_allocated - initial_allocated:.2f}GB")

    # LoRA 적용
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    after_lora_allocated, after_lora_reserved = get_gpu_memory()
    print(f"LoRA 적용 후: Allocated={after_lora_allocated:.2f}GB, Reserved={after_lora_reserved:.2f}GB")
    print(f"   증가량: {after_lora_allocated - after_load_allocated:.2f}GB")

    # 샘플 데이터로 forward pass 테스트 (batch_size별)
    test_text = "이것은 테스트 문장입니다. " * 50  # 긴 문장

    for batch_size in [1, 2, 4, 8, 16]:
        try:
            torch.cuda.empty_cache()

            # 배치 생성
            inputs = tokenizer(
                [test_text] * batch_size,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to("cuda")

            labels = inputs["input_ids"].clone()

            # Forward pass
            model.train()
            with torch.cuda.amp.autocast(dtype=compute_dtype):
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss

            # Backward pass
            loss.backward()

            forward_allocated, forward_reserved = get_gpu_memory()
            print(f"\nBatch={batch_size}: Allocated={forward_allocated:.2f}GB, Reserved={forward_reserved:.2f}GB")
            print(f"   Forward+Backward 증가량: {forward_allocated - after_lora_allocated:.2f}GB")

            # 그래디언트 초기화
            model.zero_grad()
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n❌ Batch={batch_size}: OOM!")
                break
            else:
                raise e

    # 정리
    del model
    del tokenizer
    torch.cuda.empty_cache()

    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    print("🔍 VRAM 사용량 측정 시작\n")

    # Llama-3.2-Korean-3B 테스트
    test_model_memory(
        "Bllossom/llama-3.2-Korean-Bllossom-3B",
        "llama",
        use_bf16=True
    )

    # Qwen3-4B 테스트
    test_model_memory(
        "Qwen/Qwen3-4B-Instruct-2507",
        "qwen",
        use_bf16=False
    )

    print("✅ 측정 완료!")