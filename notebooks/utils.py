"""
경진대회 파이프라인을 위한 핵심 유틸리티 함수들.

이 모듈은 WandB 통합, ROUGE 평가, Git 자동 백업, 재현성 보장, 텍스트 전처리 등
전체 노트북에서 공통으로 사용되는 기능을 제공합니다.
"""

import os
import random
import re
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import transformers
import wandb
from konlpy.tag import Okt
from rouge_score import rouge_scorer


# WandB 설정 및 초기화
def setup_wandb(
    project_name: str,
    config_dict: Dict[str, Any],
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> wandb.run:
    """WandB 실험 추적을 초기화합니다.

    Args:
        project_name: WandB 프로젝트 이름
        config_dict: 실험 설정 (하이퍼파라미터, 모델 정보 등)
        run_name: 실험 run 이름 (None이면 자동 생성)
        tags: 실험 태그 리스트

    Returns:
        초기화된 WandB run 객체

    Example:
        >>> config = {"model": "kobart", "lr": 5e-5, "batch_size": 16}
        >>> run = setup_wandb("dialogue-summarization", config)
    """
    # Run name 자동 생성
    if run_name is None:
        model_name = config_dict.get("model_name", "model")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{model_name}-{timestamp}"

    # WandB 초기화
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config_dict,
        tags=tags or [],
        reinit=True  # 노트북에서 여러 번 실행 가능하도록
    )

    print(f"✅ WandB initialized: {run_name}")
    print(f"📊 Dashboard: {run.url}")

    return run


# ROUGE 점수 계산
def compute_rouge(
    predictions: List[str],
    references: Union[List[str], List[List[str]]],
    use_korean_tokenizer: bool = True
) -> Dict[str, float]:
    """ROUGE 점수를 계산합니다.

    한국어 형태소 분석기(Okt)를 사용하여 정확한 ROUGE 점수를 계산합니다.
    경진대회 평가 방식에 따라 3개 reference 중 최대값을 선택합니다.

    Args:
        predictions: 모델 예측 결과 리스트
        references: 정답 요약 리스트 (각 예측당 1개 또는 3개)
        use_korean_tokenizer: 한국어 토크나이저 사용 여부

    Returns:
        ROUGE-1, ROUGE-2, ROUGE-L F1 점수를 담은 딕셔너리

    Example:
        >>> preds = ["요약문 1", "요약문 2"]
        >>> refs = [["정답1-1", "정답1-2", "정답1-3"], ["정답2-1", "정답2-2", "정답2-3"]]
        >>> scores = compute_rouge(preds, refs)
        >>> print(scores["rouge1"])  # 71.5
    """
    # 한국어 토크나이저 초기화
    if use_korean_tokenizer:
        okt = Okt()
        def tokenizer(text: str) -> List[str]:
            return okt.morphs(text)
    else:
        tokenizer = None

    # ROUGE scorer 생성
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=False,
        tokenizer=tokenizer
    )

    # 점수 계산
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        # Reference가 리스트인 경우 (3개 reference)
        if isinstance(ref, list):
            # 각 reference에 대해 점수 계산 후 최대값 선택
            scores_per_ref = [scorer.score(r, pred) for r in ref]
            max_rouge1 = max(s['rouge1'].fmeasure for s in scores_per_ref)
            max_rouge2 = max(s['rouge2'].fmeasure for s in scores_per_ref)
            max_rougeL = max(s['rougeL'].fmeasure for s in scores_per_ref)

            rouge1_scores.append(max_rouge1)
            rouge2_scores.append(max_rouge2)
            rougeL_scores.append(max_rougeL)
        else:
            # Reference가 단일 문자열인 경우
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

    # 평균 점수 계산
    result = {
        "rouge1": np.mean(rouge1_scores) * 100,  # 0-100 스케일
        "rouge2": np.mean(rouge2_scores) * 100,
        "rougeL": np.mean(rougeL_scores) * 100,
    }
    result["rouge_sum"] = result["rouge1"] + result["rouge2"] + result["rougeL"]

    return result


# Git 자동 백업
def auto_git_backup(
    exp_num: str,
    model_name: str,
    rouge_score: float,
    config: Dict[str, Any],
    message_prefix: str = "Exp"
) -> bool:
    """실험 결과를 Git에 자동으로 백업합니다.

    실험 번호, 모델 이름, ROUGE 점수, 설정을 포함한 커밋 메시지를 생성하고
    자동으로 커밋 및 푸시를 수행합니다.

    Args:
        exp_num: 실험 번호 (예: "001", "002")
        model_name: 모델 이름 (예: "KoBART-base")
        rouge_score: ROUGE 합계 점수
        config: 실험 설정 딕셔너리
        message_prefix: 커밋 메시지 prefix

    Returns:
        백업 성공 여부

    Example:
        >>> config = {"lr": 5e-5, "batch_size": 16, "epochs": 10}
        >>> success = auto_git_backup("001", "KoBART", 72.5, config)
    """
    try:
        # 커밋 메시지 생성
        lr = config.get("learning_rate", config.get("lr", "N/A"))
        bs = config.get("batch_size", config.get("bs", "N/A"))
        epochs = config.get("num_train_epochs", config.get("epochs", "N/A"))

        commit_message = f"""{message_prefix} #{exp_num}: {model_name} | ROUGE: {rouge_score:.1f} | lr={lr}, bs={bs}, epochs={epochs}

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"""

        # Git 작업 디렉토리로 이동 (CLAUDE.md에 명시된 규칙 준수)
        git_dir = "/Competition/NLP/naturallanguageprocessingcompetition-nlp-1"

        # Git add
        subprocess.run(
            ["git", "add", "."],
            cwd=git_dir,
            check=True,
            capture_output=True
        )

        # Git commit
        subprocess.run(
            ["git", "commit", "-m", commit_message],
            cwd=git_dir,
            check=True,
            capture_output=True
        )

        # Git push (3회 재시도)
        for attempt in range(3):
            try:
                subprocess.run(
                    ["git", "push", "origin", "main"],
                    cwd=git_dir,
                    check=True,
                    capture_output=True,
                    timeout=120
                )
                print(f"✅ Git backup successful: Exp #{exp_num}")
                return True
            except subprocess.TimeoutExpired:
                print(f"⚠️ Push timeout (attempt {attempt + 1}/3)")
                if attempt == 2:
                    print("❌ Git push failed after 3 attempts")
                    return False
            except subprocess.CalledProcessError as e:
                print(f"❌ Git push failed: {e.stderr.decode()}")
                return False

        return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Git backup failed: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"❌ Git backup error: {str(e)}")
        return False


# 재현성을 위한 시드 고정
def set_seed(seed: int = 42) -> None:
    """재현성을 위해 모든 난수 생성기의 시드를 고정합니다.

    Python random, NumPy, PyTorch, Transformers의 시드를 모두 설정하여
    실험 결과를 재현 가능하게 만듭니다.

    Args:
        seed: 설정할 시드 값 (기본값: 42)

    Example:
        >>> set_seed(42)
        >>> # 이제 모든 난수 생성이 재현 가능합니다
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # CUDA 연산의 재현성 보장 (성능 trade-off 있음)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    transformers.set_seed(seed)

    # 환경 변수 설정
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"✅ Seed set to {seed} for reproducibility")


# 대화 텍스트 정제
def clean_dialogue(text: str) -> str:
    """대화 텍스트의 노이즈를 제거하고 정제합니다.

    경진대회 데이터에 포함된 다양한 노이즈 패턴을 제거합니다:
    - Escaped newlines (\\\\n → \\n)
    - HTML tags (<br> → \\n)
    - 연속된 공백 → 단일 공백
    - Informal tokens (ㅋㅋ, ㅇㅇ 등) 정리

    Args:
        text: 정제할 대화 텍스트

    Returns:
        정제된 텍스트

    Example:
        >>> raw = "#Person1#: 안녕하세요\\\\n#Person2#: 네<br>반갑습니다ㅋㅋ"
        >>> clean = clean_dialogue(raw)
        >>> print(clean)
        #Person1#: 안녕하세요
        #Person2#: 네 반갑습니다
    """
    # Escaped newlines 처리
    text = text.replace('\\\\n', '\n')

    # HTML tags 처리
    text = text.replace('<br>', '\n')
    text = text.replace('<br/>', '\n')
    text = text.replace('<br />', '\n')

    # Informal tokens 정리 (선택적 - 경우에 따라 주석 처리)
    # text = re.sub(r'ㅋ+', '', text)  # ㅋㅋㅋ 제거
    # text = re.sub(r'ㅎ+', '', text)  # ㅎㅎㅎ 제거
    # text = re.sub(r'ㅇㅇ', '', text)  # ㅇㅇ 제거

    # 연속된 공백 제거
    text = re.sub(r' +', ' ', text)

    # 연속된 줄바꿈 제거 (최대 2개까지 유지)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 앞뒤 공백 제거
    text = text.strip()

    return text


# 추가 유틸리티: 특수 토큰 추출
def extract_special_tokens(text: str) -> List[str]:
    """텍스트에서 특수 토큰을 추출합니다.

    #Person1#, #PhoneNumber# 같은 특수 토큰을 식별하여 리스트로 반환합니다.
    Tokenizer에 추가할 special tokens을 준비할 때 사용합니다.

    Args:
        text: 분석할 텍스트

    Returns:
        발견된 특수 토큰 리스트

    Example:
        >>> text = "#Person1#: 제 번호는 #PhoneNumber# 입니다."
        >>> tokens = extract_special_tokens(text)
        >>> print(tokens)  # ['#Person1#', '#PhoneNumber#']
    """
    # #로 시작하고 끝나는 패턴 찾기
    pattern = r'#\w+#'
    tokens = re.findall(pattern, text)

    # 중복 제거 및 정렬
    unique_tokens = sorted(set(tokens))

    return unique_tokens


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 Testing utility functions...")

    # 1. set_seed 테스트
    set_seed(42)

    # 2. clean_dialogue 테스트
    dirty_text = "#Person1#: 안녕하세요\\\\n#Person2#: 네<br>반갑습니다ㅋㅋ"
    clean_text = clean_dialogue(dirty_text)
    print(f"\n📝 Cleaned text:\n{clean_text}")

    # 3. extract_special_tokens 테스트
    tokens = extract_special_tokens(clean_text)
    print(f"\n🏷️ Special tokens: {tokens}")

    print("\n✅ All tests passed!")
