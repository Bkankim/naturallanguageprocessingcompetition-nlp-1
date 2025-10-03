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
