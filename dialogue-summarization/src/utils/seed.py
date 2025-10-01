"""
시드 고정 유틸리티 모듈

재현 가능한 실험을 위해 랜덤 시드를 고정하는 함수를 제공합니다.
"""

import random
import numpy as np
import torch
from transformers import set_seed as transformers_set_seed


def set_seed(seed: int) -> None:
    """
    모든 랜덤 시드를 고정하여 재현 가능한 실험 환경을 구축합니다.

    Python의 random, NumPy, PyTorch, Transformers 라이브러리의 시드를 모두 고정합니다.
    CUDA 연산의 결정성(determinism)도 설정하여 GPU 연산의 재현성을 보장합니다.

    Args:
        seed (int): 고정할 시드 값

    Example:
        >>> set_seed(42)
        >>> # 이후 모든 랜덤 연산이 재현 가능해집니다

    Note:
        - CUDA deterministic 모드는 성능이 약간 저하될 수 있습니다
        - 완벽한 재현성을 위해서는 동일한 하드웨어 환경이 필요합니다
    """
    # Python random 모듈 시드 고정
    random.seed(seed)

    # NumPy 시드 고정
    np.random.seed(seed)

    # PyTorch CPU 시드 고정
    torch.manual_seed(seed)

    # PyTorch GPU 시드 고정 (CUDA 사용 시)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 멀티 GPU 환경

        # CUDA 연산의 결정성 보장
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Transformers 라이브러리 시드 고정
    transformers_set_seed(seed)
