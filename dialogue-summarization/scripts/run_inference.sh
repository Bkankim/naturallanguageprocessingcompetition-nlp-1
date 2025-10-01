#!/bin/bash
# 추론 실행 예제 스크립트
#
# 사용법:
#   bash scripts/run_inference.sh <checkpoint_path> [output_path] [batch_size]
#
# 예시:
#   bash scripts/run_inference.sh ./output/checkpoint-1000
#   bash scripts/run_inference.sh ./output/checkpoint-1000 submissions/my_pred.csv
#   bash scripts/run_inference.sh ./output/checkpoint-1000 submissions/my_pred.csv 64

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 스크립트 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo -e "${GREEN}==============================================================${NC}"
echo -e "${GREEN}     대화 요약 모델 추론 스크립트${NC}"
echo -e "${GREEN}==============================================================${NC}"

# 체크포인트 경로 확인
if [ -z "$1" ]; then
    echo -e "${RED}오류: 체크포인트 경로를 지정해야 합니다.${NC}"
    echo ""
    echo "사용법:"
    echo "  bash scripts/run_inference.sh <checkpoint_path> [output_path] [batch_size]"
    echo ""
    echo "예시:"
    echo "  bash scripts/run_inference.sh ./output/checkpoint-1000"
    echo "  bash scripts/run_inference.sh ./output/checkpoint-1000 submissions/my_pred.csv"
    echo "  bash scripts/run_inference.sh ./output/checkpoint-1000 submissions/my_pred.csv 64"
    exit 1
fi

CHECKPOINT_PATH="$1"
OUTPUT_PATH="${2:-}"
BATCH_SIZE="${3:-}"

# 체크포인트 존재 확인
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}오류: 체크포인트 디렉토리를 찾을 수 없습니다: ${CHECKPOINT_PATH}${NC}"
    exit 1
fi

# 설정 파일 확인
CONFIG_PATH="configs/base_config.yaml"
if [ ! -f "$CONFIG_PATH" ]; then
    echo -e "${RED}오류: 설정 파일을 찾을 수 없습니다: ${CONFIG_PATH}${NC}"
    exit 1
fi

echo -e "${YELLOW}설정:${NC}"
echo "  - Config: $CONFIG_PATH"
echo "  - Checkpoint: $CHECKPOINT_PATH"
if [ -n "$OUTPUT_PATH" ]; then
    echo "  - Output: $OUTPUT_PATH"
else
    echo "  - Output: (기본값 사용: config의 result_path/output.csv)"
fi
if [ -n "$BATCH_SIZE" ]; then
    echo "  - Batch Size: $BATCH_SIZE"
else
    echo "  - Batch Size: (기본값 사용: config의 batch_size)"
fi
echo ""

# Python 명령 구성
PYTHON_CMD="python scripts/generate_predictions.py --config $CONFIG_PATH --checkpoint $CHECKPOINT_PATH"

if [ -n "$OUTPUT_PATH" ]; then
    PYTHON_CMD="$PYTHON_CMD --output $OUTPUT_PATH"
fi

if [ -n "$BATCH_SIZE" ]; then
    PYTHON_CMD="$PYTHON_CMD --batch_size $BATCH_SIZE"
fi

# 추론 실행
echo -e "${GREEN}추론을 시작합니다...${NC}"
echo ""

eval $PYTHON_CMD

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}==============================================================${NC}"
    echo -e "${GREEN}     추론 완료!${NC}"
    echo -e "${GREEN}==============================================================${NC}"

    # 출력 파일 경로 결정
    if [ -n "$OUTPUT_PATH" ]; then
        FINAL_OUTPUT="$OUTPUT_PATH"
    else
        # config에서 기본 경로 읽기 (간단히 예측)
        FINAL_OUTPUT="prediction/output.csv"
    fi

    # 출력 파일 존재 확인 및 정보 표시
    if [ -f "$FINAL_OUTPUT" ]; then
        echo ""
        echo -e "${YELLOW}결과 파일: ${FINAL_OUTPUT}${NC}"
        echo -e "${YELLOW}파일 크기: $(du -h "$FINAL_OUTPUT" | cut -f1)${NC}"
        echo -e "${YELLOW}행 개수: $(wc -l < "$FINAL_OUTPUT")${NC}"
    fi
else
    echo ""
    echo -e "${RED}==============================================================${NC}"
    echo -e "${RED}     추론 실패${NC}"
    echo -e "${RED}==============================================================${NC}"
    exit 1
fi
