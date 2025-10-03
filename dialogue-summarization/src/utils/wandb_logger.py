"""
W&B 로깅 전용 유틸리티 모듈

구조화된 W&B 로깅을 위한 클래스와 헬퍼 함수를 제공합니다.

주요 기능:
- Run name 생성 (nickname_ep{epochs}_bs{effective_bs}_lr{lr}_{timestamp})
- 모델별 태그 자동 생성 (model-type, training, size, arch, float)
- Config 로깅 (하이퍼파라미터 추적)
- 메트릭 로깅 (train/eval/best)
- Group 관리 (encoder-decoder vs decoder-only)

사용 예:
    from src.utils.wandb_logger import WandBLogger

    logger = WandBLogger(config)
    logger.init_run(model_config)
    logger.log_metrics({"eval/rouge_sum": 45.2})
    logger.finish()
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


class WandBLogger:
    """W&B 로깅을 위한 클래스"""

    def __init__(self, config: Dict[str, Any]):
        """
        W&B Logger 초기화

        Args:
            config: 전체 설정 (wandb 섹션 포함)
        """
        self.config = config
        self.wandb_config = config.get("wandb", {})
        self.enabled = self.wandb_config.get("enabled", False)
        self.run = None

        # wandb import (optional)
        if self.enabled:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("⚠️  wandb not installed, logging disabled")
                self.enabled = False

    def init_run(self, model_config: Dict[str, Any]) -> Optional[Any]:
        """
        W&B Run 초기화

        Args:
            model_config: 모델별 설정

        Returns:
            wandb.Run 객체 (또는 None)
        """
        if not self.enabled:
            return None

        # Run name
        run_name = self._create_run_name(model_config)

        # Group
        model_type = model_config["model_type"]
        group = self.wandb_config["groups"].get(model_type, None)

        # Tags
        tags = self._create_tags(model_config)

        # Config
        wandb_config = self._create_config(model_config)

        # Initialize
        self.run = self.wandb.init(
            project=self.wandb_config["project"],
            entity=self.wandb_config.get("entity", None),
            name=run_name,
            group=group,
            tags=tags,
            config=wandb_config,
            reinit=True  # 여러 모델 순차 학습 시 필요
        )

        print(f"✅ W&B Run 초기화: {run_name}")
        print(f"   Project: {self.wandb_config['project']}")
        print(f"   Group: {group}")
        print(f"   Tags: {tags[:3]}...")

        return self.run

    def _create_run_name(self, model_config: Dict[str, Any]) -> str:
        """
        구조화된 Run name 생성

        Format: {nickname}_ep{epochs}_bs{effective_bs}_lr{lr}_{timestamp}
        예: Llama-3.2-Korean-3B_ep3_bs16_lr2e-4_20250103-143025
        """
        nickname = model_config["nickname"]
        epochs = self.config["training"]["num_train_epochs"]

        batch_size = model_config.get(
            "batch_size",
            self.config["training"]["per_device_train_batch_size"]
        )
        grad_accum = self.config["training"]["gradient_accumulation_steps"]
        effective_bs = batch_size * grad_accum

        lr = model_config.get(
            "learning_rate",
            self.config["training"]["learning_rate"]
        )

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Format: nickname_ep{epochs}_bs{effective_bs}_lr{lr}_{timestamp}
        return f"{nickname}_ep{epochs}_bs{effective_bs}_lr{lr:.0e}_{timestamp}"

    def _create_tags(self, model_config: Dict[str, Any]) -> List[str]:
        """
        모델 특성 기반 태그 생성

        자동 생성되는 태그:
        - model-type:{encoder_decoder|causal_lm}
        - training:{full-finetune|qlora-4bit}
        - size:{3B|4B|7B|8B|base}
        - arch:{llama|qwen|bart|t5}
        - float:{bf16|fp16}
        """
        tags = self.wandb_config.get("base_tags", []).copy()

        # Model type
        model_type = model_config["model_type"]
        tags.append(f"model-type:{model_type}")

        # Training type
        if model_config.get("use_qlora", False):
            tags.append("training:qlora-4bit")
        else:
            tags.append("training:full-finetune")

        # Model size
        nickname = model_config["nickname"]
        if "3B" in nickname:
            tags.append("size:3B")
        elif "4B" in nickname:
            tags.append("size:4B")
        elif "7B" in nickname:
            tags.append("size:7B")
        elif "8B" in nickname:
            tags.append("size:8B")
        elif "base" in nickname.lower():
            tags.append("size:base")

        # Architecture
        if "llama" in nickname.lower():
            tags.append("arch:llama")
        elif "qwen" in nickname.lower():
            tags.append("arch:qwen")
        elif "bart" in nickname.lower():
            tags.append("arch:bart")
        elif "t5" in nickname.lower():
            tags.append("arch:t5")

        # Float type
        if model_config.get("use_bf16", False):
            tags.append("float:bf16")
        elif model_config.get("use_fp16", False):
            tags.append("float:fp16")

        return tags

    def _create_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        W&B Config 생성 (하이퍼파라미터 추적)

        자동 기록되는 설정:
        - 모델 정보 (name, nickname, type)
        - 학습 설정 (lr, batch_size, epochs)
        - LoRA 설정 (r, alpha, dropout, target_modules)
        - 토크나이저 설정 (max_len)
        """
        batch_size = model_config.get(
            "batch_size",
            self.config["training"]["per_device_train_batch_size"]
        )
        grad_accum = self.config["training"]["gradient_accumulation_steps"]

        wandb_config = {
            # 모델 정보
            "model_name": model_config["model_name"],
            "nickname": model_config["nickname"],
            "model_type": model_config["model_type"],
            "description": model_config.get("description", ""),

            # 학습 설정
            "use_qlora": model_config.get("use_qlora", False),
            "learning_rate": model_config.get(
                "learning_rate",
                self.config["training"]["learning_rate"]
            ),
            "batch_size": batch_size,
            "effective_batch_size": batch_size * grad_accum,
            "gradient_accumulation_steps": grad_accum,
            "num_train_epochs": self.config["training"]["num_train_epochs"],
            "warmup_ratio": self.config["training"]["warmup_ratio"],
            "lr_scheduler_type": self.config["training"]["lr_scheduler_type"],
            "optim": self.config["training"]["optim"],

            # 토크나이저 설정
            "encoder_max_len": self.config["tokenizer"]["encoder_max_len"],
            "decoder_max_len": self.config["tokenizer"]["decoder_max_len"],
        }

        # LoRA config (QLoRA 모델만)
        if model_config.get("use_qlora", False):
            lora_config = self.config["lora"]
            wandb_config.update({
                "lora_r": lora_config["r"],
                "lora_alpha": lora_config["lora_alpha"],
                "lora_dropout": model_config.get("lora_dropout", 0.1),
                "target_modules": lora_config["target_modules"],
                "lora_bias": lora_config["bias"],
            })

            # QLoRA quantization config
            qlora_config = self.config["qlora"]
            wandb_config.update({
                "bnb_4bit_compute_dtype": qlora_config["bnb_4bit_compute_dtype"],
                "bnb_4bit_quant_type": qlora_config["bnb_4bit_quant_type"],
                "bnb_4bit_use_double_quant": qlora_config["bnb_4bit_use_double_quant"],
            })

        # Float type
        if model_config.get("use_bf16", False):
            wandb_config["float_type"] = "bf16"
        elif model_config.get("use_fp16", False):
            wandb_config["float_type"] = "fp16"

        return wandb_config

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        메트릭 로깅

        Args:
            metrics: 로깅할 메트릭 딕셔너리
            step: 스텝 번호 (선택)
        """
        if not self.enabled or self.run is None:
            return

        self.wandb.log(metrics, step=step)

    def log_model_info(self, model_info: Dict[str, Any]):
        """
        모델 정보 로깅 (trainable params, total params 등)

        Args:
            model_info: 모델 정보 딕셔너리
        """
        if not self.enabled or self.run is None:
            return

        self.wandb.summary.update(model_info)

    def finish(self):
        """W&B Run 종료"""
        if not self.enabled or self.run is None:
            return

        self.wandb.finish()
        print("✅ W&B Run 종료")
        self.run = None


# 헬퍼 함수 (독립적으로 사용 가능)

def format_run_name(
    nickname: str,
    epochs: int,
    effective_batch_size: int,
    learning_rate: float,
    timestamp: Optional[str] = None
) -> str:
    """
    구조화된 Run name 생성 (독립 함수)

    Args:
        nickname: 모델 닉네임
        epochs: 에폭 수
        effective_batch_size: 유효 배치 크기
        learning_rate: 학습률
        timestamp: 타임스탬프 (없으면 자동 생성)

    Returns:
        포맷된 Run name
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    return f"{nickname}_ep{epochs}_bs{effective_batch_size}_lr{learning_rate:.0e}_{timestamp}"


def extract_model_size(nickname: str) -> Optional[str]:
    """
    모델 닉네임에서 크기 추출

    Args:
        nickname: 모델 닉네임

    Returns:
        모델 크기 ("3B", "4B", "7B", "8B", "base", None)
    """
    nickname_upper = nickname.upper()

    for size in ["3B", "4B", "7B", "8B"]:
        if size in nickname_upper:
            return size

    if "base" in nickname.lower():
        return "base"

    return None


def extract_architecture(nickname: str) -> Optional[str]:
    """
    모델 닉네임에서 아키텍처 추출

    Args:
        nickname: 모델 닉네임

    Returns:
        아키텍처 ("llama", "qwen", "bart", "t5", None)
    """
    nickname_lower = nickname.lower()

    if "llama" in nickname_lower:
        return "llama"
    elif "qwen" in nickname_lower:
        return "qwen"
    elif "bart" in nickname_lower:
        return "bart"
    elif "t5" in nickname_lower:
        return "t5"

    return None
