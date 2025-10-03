# 🗨️ Korean Dialogue Summarization

> baseline.ipynb 기반 깔끔한 모듈화 프로젝트

## 📋 프로젝트 개요

한국어 대화 요약 경진대회를 위한 **BART 기반 Seq2Seq 모델**입니다.
baseline.ipynb의 검증된 코드를 모듈화하여 재사용성과 유지보수성을 높였습니다.

## 📊 최신 실험 결과

### Zero-shot LLM Screening (2025-10-04)

5개 한국어 LLM을 대화 요약 태스크로 제로샷 평가한 결과:

| 모델 | 파라미터 | ROUGE Sum | 상태 |
|------|----------|-----------|------|
| Llama-3.2-Korean-3B | 3.2B | **49.52** | ✅ 1위 |
| Llama-3-Korean-8B | 8.0B | 48.61 | 🥈 2위 |
| Qwen2.5-7B | 7.6B | 46.84 | 🥉 3위 |
| Qwen3-4B-Instruct | 4.0B | 45.02 | 4위 |
| Llama-3.2-AICA-5B | 4.3B | 41.99 | 5위 |

**다음 단계**: Llama-3.2-Korean-3B 파인튜닝 (예상 소요 시간: ~1시간)

자세한 결과는 [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) 참조.

### 주요 특징

- ✅ **LLM Screening 완료** - 5개 모델 Zero-shot 평가 (Llama-3.2-3B 선정)
- ✅ **konlpy/Java 의존성 제거** - rouge 라이브러리만 사용
- ✅ **모듈화된 구조** - 각 기능을 독립적인 모듈로 분리
- ✅ **타입 안전성** - 모든 함수에 Type hints 적용
- ✅ **상세한 문서화** - Docstring 및 가이드 완비
- ✅ **baseline.ipynb 검증** - 원본 코드의 안정성 보장

## 🚀 빠른 시작

### 1. 학습 실행

```bash
cd /Competition/NLP/dialogue-summarization

python scripts/train_baseline.py --config configs/train_config.yaml
```

### 2. 추론 실행

```bash
python scripts/generate_predictions.py \
    --config configs/base_config.yaml \
    --checkpoint checkpoints/baseline_run/final_model
```

## 📁 프로젝트 구조

```
dialogue-summarization/
├── src/
│   ├── data/              # 데이터 전처리
│   │   ├── preprocessor.py
│   │   └── dataset.py
│   ├── models/            # 모델 로딩
│   │   └── model_loader.py
│   ├── training/          # 학습 관련 (향후 확장)
│   ├── evaluation/        # 평가 메트릭
│   │   └── metrics.py
│   ├── inference/         # 추론 관련 (향후 확장)
│   └── utils/             # 유틸리티
│       └── seed.py
├── scripts/
│   ├── train_baseline.py           # 학습 스크립트
│   ├── generate_predictions.py     # 추론 스크립트
│   └── run_inference.sh            # 추론 쉘 스크립트
├── configs/
│   ├── base_config.yaml            # 기본 설정
│   └── train_config.yaml           # 학습 설정
├── checkpoints/                    # 모델 체크포인트
├── logs/                           # 로그 파일
└── submissions/                    # 제출 파일
```

## 🔧 설정 파일

### train_config.yaml

```yaml
general:
  data_path: "/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/data/"
  model_name: "digit82/kobart-summarization"  # 요약 특화 모델
  output_dir: "./checkpoints/baseline_run"

training:
  num_train_epochs: 20
  per_device_train_batch_size: 50
  per_device_eval_batch_size: 32
  learning_rate: 1.0e-5
  warmup_ratio: 0.1
  lr_scheduler_type: "cosine"
  # ... 기타 설정
```

## 📊 성능

### Dev Set (Validation)
| 모델 | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE Sum |
|------|---------|---------|---------|-----------|
| baseline.ipynb | ~16 | ~9 | ~14 | **~47** |
| Modular Structure | 32.28 | 13.46 | 30.03 | **75.77** |

### Test Set (경진대회 제출)
| 실험 | ROUGE-1 | ROUGE-2 | ROUGE-L | Final Score |
|------|---------|---------|---------|-------------|
| **Experiment #1** (Baseline) | **0.5660** | **0.3675** | **0.4719** | **46.8487** |

**📝 상세 기록**: [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md)

## 🛠️ 개발 가이드

### 모듈 사용 예시

```python
# 데이터 전처리
from src.data.preprocessor import Preprocess
from src.data.dataset import DatasetForTrain

preprocessor = Preprocess(bos_token="<s>", eos_token="</s>")
dataset = DatasetForTrain(encoder_input, decoder_input, labels, len=100)

# 모델 로딩
from src.models.model_loader import load_tokenizer_and_model

model, tokenizer = load_tokenizer_and_model(
    model_name="digit82/kobart-summarization",
    special_tokens=['#Person1#', '#Person2#']
)

# 평가 메트릭
from src.evaluation.metrics import compute_metrics_for_trainer

compute_metrics = compute_metrics_for_trainer(tokenizer)
```

### 의존성

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.35+
- rouge 1.0.1
- pandas, PyYAML, tqdm

**중요**: konlpy나 Java는 필요 없습니다!

## 📝 주요 변경사항

### baseline.ipynb 대비 개선점

1. **Java 의존성 제거**
   - konlpy → rouge 라이브러리로 대체
   - JVM 초기화 오류 완전 해결

2. **모듈화**
   - 단일 노트북 → 독립적인 모듈들
   - 재사용성 및 테스트 용이성 향상

3. **타입 안전성**
   - 모든 함수에 Type hints 추가
   - IDE 자동완성 및 타입 체킹 지원

4. **문서화**
   - 상세한 Docstring (Google 스타일)
   - README 및 사용 가이드 완비

## 💡 개발 인사이트 & 배운 점

### 1. 모델 크기 ≠ 성능

**문제**: 처음에는 "큰 모델이 무조건 좋을 것"이라고 생각했습니다.

**발견**: Llama-3.2-Korean-3B (3.21B)가 AICA-5B (4.31B)와 Llama-3-8B (8.03B)보다 더 나은 성능을 보였습니다.

| 모델 | 파라미터 | ROUGE Sum | 비고 |
|------|----------|-----------|------|
| Llama-3.2-Korean-3B | 3.21B | **49.52** | 🏆 |
| Llama-3-Korean-8B | 8.03B | 48.61 | 2.6배 크지만 0.91점만 차이 |
| Llama-3.2-AICA-5B | 4.31B | 41.99 | 1.3배 크지만 7.53점 낮음 |

**교훈**:
- **Task Alignment이 모델 크기보다 중요합니다**
- Instruction-tuned 모델 > Conversation-specialized 모델 (요약 태스크의 경우)
- AICA는 "대화 생성"에 특화되어 Zero-shot 요약 성능이 낮았습니다
- 파인튜닝으로 역전 가능성은 있지만, 좋은 출발점을 선택하는 게 효율적입니다

### 2. SOLAR 모델의 극심한 속도 저하

**문제**: SOLAR-10.7B가 다른 모델 대비 **40배 느렸습니다** (22분/배치 vs 27-33초/배치).

**가설 검증**:
- ✅ 모델 크기 (10.7B vs 7-8B): 1.5배 차이는 40배 속도 저하를 설명하지 못함
- ✅ 양자화 문제: 4bit NF4 정상 작동 (15GB GPU, 2GB RAM)
- ✅ FlashAttention: SDPA 활성화됨
- ✅ 배치 크기, 시퀀스 길이: 모두 동일
- 🚨 **Root Cause**: **Depth Upscaling 아키텍처**

**발견**: SOLAR은 두 개의 Llama2-7B 모델을 수직으로 병합하여 ~48 layers를 가집니다.
- 일반 10B 모델: ~32 layers (~20 TFLOPs/token)
- SOLAR: ~48 layers (~30 TFLOPs/token)
- Beam search (4 beams) × 긴 forward pass = 극심한 속도 저하

**교훈**:
- 모델 아키텍처가 추론 속도에 큰 영향을 미칩니다
- 파라미터 수만으로 속도를 예측할 수 없습니다
- Zero-shot 스크리닝은 실제 추론 환경에서 테스트해야 합니다

**해결**: SOLAR을 Qwen3-4B-Instruct-2507로 교체 (정상 속도 복구)

### 3. 초기 스크리닝의 10가지 문제점

첫 스크리닝에서 ROUGE Sum이 1-2점대로 비정상적으로 낮게 나왔습니다.

**발견한 문제들** (상세: [SCREENING_ISSUES_ANALYSIS.md](./screening_results/SCREENING_ISSUES_ANALYSIS.md)):

**Critical Issues**:
1. ❌ Chat Template 미적용 → 모델이 시스템 프롬프트를 무시
2. ❌ 한국어 프롬프트 부족 → 영어/일본어 혼합 출력
3. ❌ Character-level 토큰화 → 한국어 ROUGE 부정확
4. ❌ 외국어 토큰 차단 미적용 → Latin/Japanese 출력
5. ❌ QLoRA 미적용 → OOM 또는 느린 속도
6. ❌ TF32 미활성화 → RTX 3090 성능 미활용

**수정 후 성능 개선**:
- Llama-3.2-Korean-3B: 1.84 → **49.52** (26.9배 향상)
- Llama-3-Korean-8B: 1.18 → **48.61** (41.2배 향상)

**교훈**:
- LLM은 **Chat Template이 필수**입니다 (특히 Instruction-tuned 모델)
- 한국어 평가는 **Mecab 형태소 토큰화**가 정확합니다
- `bad_words_ids`로 121,413개 외국어 토큰 차단 → 순수 한국어 출력 보장
- RTX 3090에서는 TF32 활성화가 필수입니다

### 4. 디스크 관리 전략

**문제**: Qwen3-4B 스크리닝 시작 시 디스크 용량 초과 (91.53GB > 80GB 한도)

**발견**:
```bash
HuggingFace 모델 캐시: 20GB (SOLAR, Qwen2.5 등 불필요한 캐시)
HuggingFace xet 캐시: 9.3GB
UV archive: 7.8GB
W&B artifacts: 473MB
```

**해결**: 불필요한 캐시 정리 → 68GB로 축소

**캐시 삭제 전략**:
```bash
# 사용 완료된 모델 캐시 삭제
rm -rf ~/.cache/huggingface/hub/models--<model_name>

# 전체 xet 캐시 삭제 (LFS 관련)
rm -rf ~/.cache/huggingface/xet

# UV 패키지 관리자 아카이브 삭제
rm -rf ~/.cache/uv/archive-v0

# W&B 오래된 artifacts 삭제
rm -rf ~/.cache/wandb/artifacts
```

**교훈**:
- 실험 중에는 정기적으로 디스크 사용량을 체크하세요
- 각 모델 스크리닝 후 즉시 캐시를 삭제하는 것이 좋습니다
- `delete_cache=True` 옵션을 코드에 통합하면 자동화할 수 있습니다

### 5. Zero-shot 평가의 중요성

**교훈**: 파인튜닝 전에 Zero-shot 평가로 모델을 스크리닝하면:
- 시간 절약: 5개 모델 × 1시간 학습 = 5시간 vs 5개 × 15분 추론 = 75분
- 초기 성능 파악: 좋은 출발점 선택 → 파인튜닝 효율 향상
- Task Alignment 검증: 모델이 태스크를 이해하는지 미리 확인

**다음 단계**: Llama-3.2-Korean-3B를 선택하여 파인튜닝 진행 (예상: 1시간)

### 6. 실패한 시도들

**시도 1**: AICA-5B가 5B 모델이니 좋을 것이라 예상
- **결과**: 5위 (41.99)
- **이유**: 대화 생성에 특화되어 요약 성능 낮음

**시도 2**: SOLAR-10.7B가 10B 모델이니 가장 좋을 것이라 예상
- **결과**: 중단 (극심한 속도 저하)
- **이유**: Depth Upscaling 아키텍처로 인한 높은 FLOPs

**시도 3**: Character-level ROUGE로 한국어 평가
- **결과**: 부정확한 점수 (형태소 단위가 정확함)
- **수정**: Mecab 형태소 토큰화 적용

## 🐛 트러블슈팅

### Q: Java/konlpy 오류가 발생합니다
**A**: 이 프로젝트는 Java를 사용하지 않습니다. `src/evaluation/metrics.py`가 rouge 라이브러리만 사용하는지 확인하세요.

### Q: 모델이 로딩되지 않습니다
**A**:
```bash
# 모델명 확인
model_name: "digit82/kobart-summarization"

# 또는 원본 모델
model_name: "gogamza/kobart-base-v2"
```

### Q: ROUGE 점수가 너무 낮습니다
**A**:
- config에서 `num_train_epochs`를 20으로 설정했는지 확인
- 학습률(`learning_rate: 1e-5`)이 적절한지 확인
- Early stopping patience(3)가 너무 작지 않은지 확인

## 📚 참고 자료

- 원본 baseline: `/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code/baseline.ipynb`
- 대회 문서: `/Competition/NLP/docs/`
- Claude 가이드: `/Competition/NLP/CLAUDE.md`

## 🔄 다음 단계

- [ ] WandB 통합 (선택)
- [ ] Hyperparameter tuning (Optuna)
- [ ] Advanced models (mBART, KoGPT 등)
- [ ] Ensemble 전략

## 📄 라이선스

이 프로젝트는 경진대회 제출용입니다.

---

**Built with** ❤️ **by Claude Code**
