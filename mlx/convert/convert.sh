#!/bin/bash
# ===========================================
# Whisper 模型轉換腳本
# 將 HuggingFace Whisper 模型轉換為 MLX 格式
# ===========================================

set -e

# 顏色定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 顯示使用說明
usage() {
    echo "使用方式: $0 <hf-repo> [options]"
    echo ""
    echo "參數:"
    echo "  hf-repo    HuggingFace 模型路徑"
    echo ""
    echo "選項:"
    echo "  --force    強制重新轉換（即使模型已存在）"
    echo "  --float32  使用 float32（預設為 float16）"
    echo ""
    echo "範例:"
    echo "  $0 formospeech/whisper-large-v2-taiwanese-hakka-v1"
    echo "  $0 openai/whisper-large-v3 --force"
    exit 1
}

# 檢查參數
if [ $# -lt 1 ]; then
    usage
fi

HF_REPO="$1"
shift

# 解析其他參數
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            EXTRA_ARGS="$EXTRA_ARGS --force"
            shift
            ;;
        --float32)
            EXTRA_ARGS="$EXTRA_ARGS --dtype float32"
            shift
            ;;
        *)
            echo -e "${RED}未知選項: $1${NC}"
            usage
            ;;
    esac
done

echo -e "${GREEN}=== Whisper MLX 模型轉換 ===${NC}"
echo ""

# 取得腳本所在目錄
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 步驟 1: 設置環境
echo -e "${YELLOW}設置環境...${NC}"
if [ ! -d ".venv" ]; then
    uv venv
fi

uv pip install mlx mlx-whisper numpy huggingface_hub safetensors --quiet

# 步驟 2: 執行轉換
echo ""
uv run python convert.py "$HF_REPO" $EXTRA_ARGS
