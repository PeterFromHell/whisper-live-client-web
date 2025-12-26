# MLX Whisper 客戶端

使用 Apple Silicon GPU 加速的即時語音轉文字工具。

## 優勢

| 特性 | WhisperLive (faster-whisper) | MLX Whisper |
|------|------------------------------|-------------|
| Apple GPU | ❌ 不支援 | ✅ 支援 |
| 運算裝置 | CPU only | Apple Silicon GPU |
| 架構 | Client-Server | 單一程式 |

## 設置步驟

### 1. 安裝系統依賴

```bash
brew install ffmpeg portaudio
```

### 2. 建立虛擬環境

```bash
cd /Users/winston/Projects/whisper-live-client/mlx
uv venv
uv pip install mlx-whisper pyaudio numpy
```

---

## 🎤 即時語音辨識

### 轉換模型

首先將 HuggingFace 模型轉換為 MLX 格式：

```bash
cd convert
./convert.sh formospeech/whisper-large-v2-taiwanese-hakka-v1
```

### 使用方式

```bash
# 基本使用（自動偵測語言，純轉錄）
uv run python realtime.py

# 列出可用模型
uv run python realtime.py --list

# 指定模型
uv run python realtime.py --model whisper-large-v2-taiwanese-hakka-v1-mlx

# 翻譯成英文
uv run python realtime.py --task translate

# 指定語言
uv run python realtime.py --language zh

# 組合使用
uv run python realtime.py -m whisper-large-v2-taiwanese-hakka-v1-mlx -l zh -t transcribe
```

### 參數說明

| 參數 | 簡寫 | 說明 | 預設值 |
|------|------|------|--------|
| `--model` | `-m` | 模型名稱或路徑 | 第一個可用模型 |
| `--task` | `-t` | `transcribe`（轉錄）或 `translate`（翻譯成英文）| `transcribe` |
| `--language` | `-l` | 語言代碼（zh, en, ja...）| 自動偵測 |
| `--list` | | 列出可用模型 | |

---

## 🖥️ 浮動字幕視窗（簡報用）

適用於全螢幕簡報時即時顯示字幕。

```bash
cd subtitle
uv pip install pyobjc-framework-Cocoa

# 基本使用
uv run python subtitle.py

# 翻譯成英文
uv run python subtitle.py --task translate

# 指定模型和語言
uv run python subtitle.py -m whisper-large-v2-taiwanese-hakka-v1-mlx -l zh
```

詳細說明請參考 [subtitle/README.md](subtitle/README.md)。

---

## 轉換自訂模型

可以將 HuggingFace 上的任何 Whisper 模型轉換為 MLX 格式。

```bash
cd convert

# 轉換模型
./convert.sh <hf-repo>

# 範例
./convert.sh formospeech/whisper-large-v2-taiwanese-hakka-v1
./convert.sh openai/whisper-large-v3

# 強制重新轉換
./convert.sh formospeech/whisper-large-v2-taiwanese-hakka-v1 --force
```

轉換後的模型存放在 `models/` 目錄。

詳細說明請參考 [convert/README.md](convert/README.md)。

---

## 使用 HuggingFace 模型（自動下載）

這些腳本使用 mlx-community 的模型，會自動下載：

```bash
# 中文翻譯成英文
uv run python transcribe.py

# 純中文轉錄
uv run python transcribe_only.py
```

---

## 可用模型

### HuggingFace 模型（自動下載）

⚠️ **注意：turbo 版本不支援翻譯功能！**

| 模型 | 大小 | 翻譯支援 |
|------|------|----------|
| `mlx-community/whisper-large-v3-mlx` | ~3 GB | ✅ 支援 |
| `mlx-community/whisper-large-v3-turbo` | ~1.6 GB | ❌ 不支援 |
| `mlx-community/whisper-small` | ~488 MB | ✅ 支援 |

### 本地轉換模型

可以轉換 HuggingFace 上的任何 Whisper 模型：

- `formospeech/whisper-large-v2-taiwanese-hakka-v1` - 臺灣客語
- `openai/whisper-large-v3` - OpenAI 官方模型

---

## 確認 GPU 使用

執行時打開「活動監視器」→「GPU」分頁，應該會看到 Python 正在使用 GPU。

## 與 WhisperLive 的差異

- **MLX Whisper**：單一程式，使用 Apple GPU，說完一句話後才辨識
- **WhisperLive**：Client-Server 架構，使用 CPU，可以即時串流顯示

如果需要「邊說邊顯示」的即時效果，請使用上層目錄的 WhisperLive 版本。

---

## 目錄結構

```
mlx/
├── realtime.py           # 🎤 即時語音辨識
├── convert/              # 模型轉換工具
│   ├── convert.sh
│   ├── convert.py
│   └── README.md
├── models/               # 轉換後的模型
│   └── {model-name}-mlx/
├── subtitle/             # 🖥️ 浮動字幕視窗
│   ├── subtitle.py
│   └── README.md
├── transcribe.py         # 中→英翻譯（HF 模型）
└── transcribe_only.py    # 純轉錄（HF 模型）
```
