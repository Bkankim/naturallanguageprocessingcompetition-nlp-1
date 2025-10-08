# 🗨️ Korean Dialogue Summarization Competition

> 한국어 대화 요약 경진대회 프로젝트

## 📋 프로젝트 개요

한국어 대화 데이터를 입력받아 핵심 내용을 요약하는 **Seq2Seq 모델**을 개발하는 경진대회입니다.

- **평가 지표**: ROUGE-1, ROUGE-2, ROUGE-L F1 평균 (3개의 참조 요약 대비)
- **모델**: BART 기반 한국어 요약 모델
- **데이터**: DialogSum 한국어 번역 버전 (Train 12,457 / Dev 499 / Test 250)

## 🚀 빠른 시작

### 새로운 모듈화 구조 (권장)

```bash
cd /Competition/NLP/dialogue-summarization

# 학습
python scripts/train_baseline.py --config configs/train_config.yaml

# 추론
python scripts/generate_predictions.py \
    --config configs/base_config.yaml \
    --checkpoint checkpoints/baseline_run/checkpoint-1750
```

### 원본 Baseline (참고용)

```bash
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code

# Jupyter Notebook
jupyter notebook baseline.ipynb
```

## 📁 프로젝트 구조

```
/Competition/NLP/
├── dialogue-summarization/          # ⭐ 새로운 모듈화 프로젝트
│   ├── src/                         # 모듈 소스 코드
│   │   ├── data/                    # 데이터 전처리
│   │   ├── models/                  # 모델 로더
│   │   ├── evaluation/              # ROUGE 메트릭 (konlpy 제거!)
│   │   └── utils/                   # 유틸리티
│   ├── scripts/                     # 실행 스크립트
│   │   ├── train_baseline.py
│   │   └── generate_predictions.py
│   ├── configs/                     # YAML 설정 파일
│   ├── train_demo.ipynb            # 학습 재현 노트북
│   ├── inference_demo.ipynb        # 추론 재현 노트북
│   ├── EXPERIMENT_LOG.md           # 실험 기록
│   └── README.md                    # 상세 가이드
│
├── naturallanguageprocessingcompetition-nlp-1/  # Git 저장소
│   ├── data/                        # 원본 데이터셋
│   │   ├── train.csv
│   │   ├── dev.csv
│   │   └── test.csv
│   ├── code/
│   │   └── baseline.ipynb           # 검증된 baseline (ROUGE Sum ~47)
│   └── .git/                        # ⚠️ Git 작업은 여기서만!
│
├── docs/                            # 경진대회 문서
│   ├── Competition_Overview/
│   └── Competition_Advanced/
│
├── .claude/                         # Epic/Task 관리
└── CLAUDE.md                        # Claude Code 가이드
```

## ⚡ 주요 특징

### dialogue-summarization/ (새 구조)

- ✅ **Java 의존성 제거** - konlpy 대신 rouge 라이브러리 사용
- ✅ **모듈화** - baseline.ipynb를 재사용 가능한 모듈로 분해
- ✅ **타입 안전성** - 모든 함수에 Type Hints 적용
- ✅ **YAML 설정** - 하이퍼파라미터 관리 용이
- ✅ **재현 노트북** - 학습/추론을 쉽게 재현 가능
- ✅ **문서화** - Docstring 및 가이드 완비

### baseline.ipynb (원본)

- ✅ **검증된 성능** - ROUGE Sum ~47 달성
- ✅ **단순 구조** - 단일 노트북에서 완결
- ⚠️ **Java 의존성** - konlpy 필요 (JVM 설치 필요)
- ⚠️ **재사용성 낮음** - 모듈화되지 않음

## 📊 성능 비교

### Dev Set (Validation)
| 모델 | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE Sum |
|------|---------|---------|---------|-----------|
| baseline.ipynb | ~16 | ~9 | ~14 | **~47** |
| Modular Structure | 32.28 | 13.46 | 30.03 | **75.77** |

### Test Set (경진대회 제출)
| 실험 | ROUGE-1 | ROUGE-2 | ROUGE-L | Final Score |
|------|---------|---------|---------|-------------|
| **Experiment #1** (Baseline) | **0.5660** | **0.3675** | **0.4719** | **46.8487** |

**📝 상세 기록**: [dialogue-summarization/EXPERIMENT_LOG.md](./dialogue-summarization/EXPERIMENT_LOG.md)

## ⚠️ Git 작업 주의사항

**절대 헷갈리지 말 것!**

```bash
# ✅ 올바른 Git 작업 디렉토리
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1
git status
git add .
git commit -m "message"
git push origin main

# ❌ 잘못된 디렉토리 - 루트에서 Git 명령 금지!
cd /Competition/NLP
git add .  # 작동 안 함!
```

**Git 저장소**: `https://github.com/Bkankim/naturallanguageprocessingcompetition-nlp-1.git`

## 📚 문서

- **프로젝트 가이드**: `dialogue-summarization/README.md`
- **실험 기록**: `dialogue-summarization/EXPERIMENT_LOG.md`
- **경진대회 소개**: `docs/Competition_Overview/introduction.md`
- **데이터 설명**: `docs/Competition_Overview/data_overview.md`
- **평가 방법**: `docs/Competition_Overview/evaluation_method.md`
- **하이퍼파라미터 튜닝**: `docs/Competition_Advanced/hyperparameter_tuning_tip.md`
- **Claude 가이드**: `CLAUDE.md`

## 🔧 개발 환경

### 필수 패키지

```bash
# Python 3.10+
pip install torch transformers datasets
pip install rouge pandas pyyaml tqdm

# ⚠️ 새 구조는 konlpy 불필요!
# pip install konlpy  # baseline.ipynb만 필요
```

### PM CLI 사용

```bash
/pm:install     # 전체 의존성 설치
/pm:add rouge   # 새 패키지 추가
/pm:run test    # 테스트 실행
```

## 🎯 실험 진행 상황

### ✅ 완료된 실험

#### Experiment #1: koBART Baseline
- **모델**: gogamza/kobart-base-v2
- **최종 성능**: ROUGE Sum **94.51** (Dev Set)
  - ROUGE-1: 51.98
  - ROUGE-2: 26.61
  - ROUGE-L: 47.03
- **학습 설정**: 20 epochs, lr=5e-5, batch=8
- **상태**: ✅ 완료

#### Technical Fixes Applied
1. **Metric Configuration Fix**
   - 문제: `metric_for_best_model`이 config에 하드코딩되어 있어 다른 메트릭 사용 불가
   - 해결: TrainingArguments에서 동적으로 설정하도록 수정

2. **Chat Template Tokens**
   - 문제: LLM 모델의 chat template 토큰들이 tokenizer에 추가되지 않음
   - 해결: 모든 chat template 토큰 자동 추출 및 추가

3. **QLoRA Compute Dtype Alignment**
   - 문제: 모델별로 다른 dtype 사용 (Llama=bf16, Qwen=fp16)
   - 해결: 모델별 compute_dtype을 모델의 기본 dtype과 일치시킴

### ⏳ 진행 중인 실험

#### Experiment #2: LLM Fine-tuning with QLoRA 4bit
- **실험 시작**: 2025-10-04 14:22 KST
- **W&B 추적**: [dialogue-summarization-finetuning](https://wandb.ai/bkan-ai/dialogue-summarization-finetuning)

**모델 학습 순서** (Sequential Training):
1. 🔄 **Llama-3.2-Korean-3B** - In Progress (33% 완료)
   - Bllossom/llama-3.2-Korean-Bllossom-3B
   - QLoRA 4bit, bf16 compute dtype

2. ⏸️ **Qwen3-4B-Instruct** - Pending
   - Qwen/Qwen3-4B-Instruct-2507
   - QLoRA 4bit, fp16 compute dtype

3. ⏸️ **Qwen2.5-7B-Instruct** - Pending
   - Qwen/Qwen2.5-7B-Instruct
   - QLoRA 4bit, fp16 compute dtype

4. ⏸️ **Llama-3-Korean-8B** - Pending
   - MLP-KTLim/llama-3-Korean-Bllossom-8B
   - QLoRA 4bit, bf16 compute dtype

**학습 설정**:
- LoRA: r=16, alpha=32, dropout=0.1
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Batch size: 4 (gradient_accumulation_steps=2)
- Learning rate: 2e-4
- Max epochs: 3
- Early stopping: patience=2

### 📊 스토리지 현황
- **현재 사용량**: ~110GB / 150GB
- **안전 마진**: 40GB 여유
- **상태**: ✅ 정상

### 📝 다음 단계

1. ✅ 기존 코드 백업 완료
2. ✅ 새로운 모듈화 구조 구축 완료
3. ✅ Java 의존성 제거 완료
4. ✅ 실제 학습 실행 및 성능 검증 완료
5. ✅ 실험 기록 시스템 구축 완료
6. ✅ Git 저장소 동기화
7. ✅ koBART Fine-tuning 완료 (ROUGE Sum: 94.51)
8. ✅ Critical Issues 수정 완료
9. 🔄 LLM Fine-tuning 진행 중 (4개 모델 순차 학습)
10. ⏳ 최종 모델 선택 및 앙상블 (예정)

## 📝 라이선스

경진대회 제출용 프로젝트입니다.

---

**Built with** ❤️ **by Claude Code**
