# WhisperLive 客戶端

即時語音轉文字與翻譯工具，使用 OpenAI Whisper 模型。支援 Apple Silicon GPU 加速。

## 功能特色

- ✅ 即時語音轉文字（Transcribe）
- ✅ 即時語音翻譯成英文（Translate）
- ✅ Apple Silicon GPU 加速（MLX 版本）
- ✅ **浮動字幕視窗** - 適用於全螢幕簡報時顯示即時翻譯

## 三種版本

| 版本 | 目錄 | 運算 | 特性 |
|------|------|------|------|
| WhisperLive | `/` (根目錄) | CPU | 即時串流，邊說邊顯示 |
| MLX Whisper | `/mlx` | Apple GPU | 更快速度，說完才辨識 |
| 浮動字幕 | `/mlx/subtitle` | Apple GPU | 全螢幕簡報時顯示字幕 |

**建議：** 如果你有 Apple Silicon Mac，推薦使用 `mlx/` 目錄的 MLX 版本，速度更快。

---

## WhisperLive 版本（CPU）

### 設置步驟

```bash
# 安裝 PortAudio
brew install portaudio

# 建立虛擬環境並安裝依賴
cd whisper-live-client
uv venv
uv pip install whisper-live pyaudio
```

### 使用方式

需要開啟兩個終端視窗。

**啟動伺服器（終端視窗 1）：**
```bash
uv run python server.py
```

**啟動客戶端（終端視窗 2）：**

```bash
# 中文翻譯成英文
uv run python transcribe.py

# 純中文轉錄（不翻譯）
uv run python transcribe_only.py
```

---

## MLX Whisper 版本（Apple GPU）⚡

👉 詳細說明請看 [mlx/README.md](mlx/README.md)

### 快速開始

```bash
# 安裝依賴
brew install ffmpeg portaudio

# 設置環境
cd whisper-live-client/mlx
uv venv
uv pip install mlx-whisper pyaudio numpy

# 執行（不需要啟動伺服器）
uv run python transcribe.py
```

---

## 浮動字幕視窗（簡報用）🎤

適用於使用 Google Slides、PowerPoint、Keynote 等全螢幕簡報時，即時顯示英文翻譯字幕。

👉 詳細說明請看 [mlx/subtitle/README.md](mlx/subtitle/README.md)

### 特色

- 字幕視窗始終顯示在最上層（包括全螢幕應用上方）
- 半透明背景，不會過度遮擋簡報內容
- 可拖動調整位置
- 可自訂視窗大小、字體大小、顏色
- 使用 Apple Silicon GPU 加速

### 快速開始

```bash
# 安裝依賴
brew install ffmpeg portaudio

# 設置環境
cd whisper-live-client/mlx/subtitle
uv venv
uv pip install mlx-whisper pyaudio numpy pyobjc-framework-Cocoa

# 執行
uv run python floating_subtitle_native.py
```

### 簡報流程

1. 啟動字幕程式，等待顯示「準備就緒」
2. 拖動字幕視窗到適合的位置（例如螢幕底部）
3. 開啟簡報軟體並進入全螢幕模式
4. 開始說中文，英文翻譯會即時顯示
5. 按 `Ctrl+C` 結束程式

### 自訂設定

編輯 `floating_subtitle_native.py` 開頭的設定：

```python
# 視窗設定
WINDOW_WIDTH_RATIO = 0.8      # 視窗寬度佔螢幕比例
WINDOW_HEIGHT = 100           # 視窗高度 (像素)
WINDOW_OPACITY = 0.85         # 視窗透明度

# 文字設定
FONT_SIZE = 28                # 字體大小
TEXT_COLOR = "white"          # 文字顏色：white / yellow / green / cyan
```

---

## 功能說明

### Transcribe（轉錄）

將語音轉成文字，保持原本的語言。

### Translate（翻譯）

將任何語言的語音翻譯成英文。

**注意：** Whisper 的 `translate` 功能只能翻譯成英文。

---

## 模型大小

### MLX 版本模型

⚠️ **重要：turbo 版本不支援翻譯功能！**

| 模型 | 大小 | 翻譯支援 | 說明 |
|------|------|----------|------|
| `mlx-community/whisper-large-v3-mlx` | ~3 GB | ✅ 支援 | 翻譯功能推薦 |
| `mlx-community/whisper-large-v3-turbo` | ~1.6 GB | ❌ 不支援 | 只能轉錄，速度快 |
| `mlx-community/whisper-small` | ~488 MB | ✅ 支援 | 一般使用 |

### WhisperLive 版本模型

| 模型 | 大小 | 速度 | 準確度 |
|------|------|------|--------|
| `tiny` | ~75 MB | 最快 | 較低 |
| `base` | ~145 MB | 快 | 基本 |
| `small` | ~488 MB | 中等 | 好 |
| `medium` | ~1.5 GB | 較慢 | 很好 |
| `large-v3` | ~3 GB | 最慢 | 最好 |

首次使用新模型時會自動下載。

---

## 語言代碼

| 語言 | 代碼 |
|------|------|
| 中文 | `zh` |
| 英文 | `en` |
| 日文 | `ja` |
| 韓文 | `ko` |
| 自動偵測 | 不設定 `lang` 參數 |

---

## 疑難排解

### 麥克風測試

如果說話後沒有顯示結果，先測試麥克風：

```bash
uv run python test_mic.py
```

若音量條不動：
1. 檢查 **系統設定** → **隱私與安全性** → **麥克風** → 確認終端機有權限
2. 檢查 **系統設定** → **聲音** → **輸入** → 確認選對麥克風且音量足夠

### Port 9090 被佔用（WhisperLive 版本）

```bash
# 方法 1：使用清理腳本
./kill_server.sh

# 方法 2：手動指令
lsof -ti:9090 | xargs kill -9
```

### 確認 MLX 版本有使用 GPU

執行時打開「活動監視器」→「GPU」分頁，應該會看到 Python 正在使用 GPU。

### 字幕沒有顯示在全螢幕上方

請使用 `floating_subtitle_native.py` 版本，它使用 macOS 原生 API 確保視窗層級。

---

## 檔案說明

### WhisperLive 版本（根目錄）

| 檔案 | 說明 |
|------|------|
| `server.py` | WhisperLive 伺服器 |
| `transcribe.py` | 客戶端：中文 → 英文翻譯 |
| `transcribe_only.py` | 客戶端：純轉錄（不翻譯） |
| `test_mic.py` | 麥克風測試工具 |
| `kill_server.sh` | 清理 port 9090 腳本 |

### MLX 版本（mlx/ 目錄）

| 檔案 | 說明 |
|------|------|
| `transcribe.py` | 中文 → 英文翻譯（GPU 加速） |
| `transcribe_only.py` | 純轉錄（GPU 加速） |

### 浮動字幕（mlx/subtitle/ 目錄）

| 檔案 | 說明 |
|------|------|
| `floating_subtitle_native.py` | macOS 原生版（推薦，支援全螢幕） |
| `floating_subtitle.py` | tkinter 版（備用） |

---

## 授權

MIT License
