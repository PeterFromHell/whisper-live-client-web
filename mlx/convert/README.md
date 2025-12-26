# Whisper 模型轉換工具

將 HuggingFace Whisper 模型轉換為 MLX 格式，以便在 Apple Silicon 上使用 GPU 加速。

## 快速開始

```bash
cd /Users/winston/Projects/whisper-live-client/mlx/convert

# 轉換臺灣客語模型
./convert.sh formospeech/whisper-large-v2-taiwanese-hakka-v1

# 轉換其他模型
./convert.sh openai/whisper-large-v3
```

轉換完成後，模型會保存在 `../models/{模型名稱}-mlx/` 目錄。

## 使用方式

### Shell 腳本

```bash
# 基本用法
./convert.sh <hf-repo>

# 強制重新轉換
./convert.sh <hf-repo> --force

# 使用 float32 精度
./convert.sh <hf-repo> --float32
```

### Python 直接呼叫

```bash
# 設置環境
uv venv
uv pip install mlx mlx-whisper numpy huggingface_hub safetensors

# 執行轉換
uv run python convert.py formospeech/whisper-large-v2-taiwanese-hakka-v1

# 查看所有選項
uv run python convert.py --help
```

## 選項說明

| 選項 | 說明 |
|------|------|
| `--output-dir` | 輸出目錄（預設：`../models`）|
| `--dtype` | 數據類型：`float16`（預設）或 `float32` |
| `--force` | 強制重新轉換，即使模型已存在 |

## 輸出結構

```
mlx/
└── models/
    ├── whisper-large-v2-taiwanese-hakka-v1-mlx/
    │   ├── config.json
    │   └── weights.npz
    └── whisper-large-v3-mlx/
        ├── config.json
        └── weights.npz
```

## 使用轉換後的模型

### 即時語音辨識

```bash
cd /Users/winston/Projects/whisper-live-client/mlx

# 列出可用模型
uv run python realtime.py --list

# 使用模型（轉錄）
uv run python realtime.py --model whisper-large-v2-taiwanese-hakka-v1-mlx

# 翻譯成英文
uv run python realtime.py -m whisper-large-v2-taiwanese-hakka-v1-mlx --task translate

# 指定語言
uv run python realtime.py -m whisper-large-v2-taiwanese-hakka-v1-mlx -l zh
```

### Python API

```python
import mlx_whisper

result = mlx_whisper.transcribe(
    "audio.wav",
    path_or_hf_repo="models/whisper-large-v2-taiwanese-hakka-v1-mlx",
    language="zh",      # 可選，預設自動偵測
    task="transcribe"   # 或 "translate"
)

print(result["text"])
```

## 支援的模型

理論上支援所有 HuggingFace 上的 Whisper 模型，包括：

- OpenAI 官方模型：`openai/whisper-large-v3`
- 社群微調模型：`formospeech/whisper-large-v2-taiwanese-hakka-v1`
- 其他 Whisper 變體

## 注意事項

- 首次轉換會從 HuggingFace 下載模型
- large 模型約 3GB，需要足夠的記憶體
- 轉換後使用 float16 格式以節省空間
- 如果模型已轉換過，會自動跳過（使用 `--force` 強制重新轉換）

## 疑難排解

### 記憶體不足

關閉其他應用程式，或使用 float32 格式：

```bash
./convert.sh <hf-repo> --float32
```

### 模型載入失敗

確認輸出目錄中有完整的檔案：

```bash
ls -la ../models/<model-name>-mlx/
# 應該看到:
# config.json
# weights.npz
```

### 網路問題

如果下載失敗，可以手動下載模型後指定本地路徑。
