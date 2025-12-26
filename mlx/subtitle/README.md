# 即時字幕浮動視窗

在簡報時顯示即時字幕，視窗會顯示在**全螢幕簡報上方**。

## 特色

- ✅ 即時語音辨識 / 翻譯成英文
- ✅ 浮動視窗，始終在最上層
- ✅ 支援全螢幕模式（Google Slides、PowerPoint、Keynote）
- ✅ 可拖動調整位置
- ✅ 使用 Apple Silicon GPU 加速
- ✅ 支援本地轉換的模型
- ✅ 可自訂視窗大小、字體大小、顏色

## 設置

```bash
cd /Users/winston/Projects/whisper-live-client/mlx/subtitle
uv venv
uv pip install mlx-whisper pyaudio numpy pyobjc-framework-Cocoa
```

## 使用方式

### 基本使用

```bash
# 使用第一個可用的本地模型
uv run python subtitle.py

# 列出可用模型
uv run python subtitle.py --list
```

### 指定模型

```bash
uv run python subtitle.py --model whisper-large-v2-taiwanese-hakka-v1-mlx
```

### 翻譯成英文

```bash
uv run python subtitle.py --task translate
```

### 指定語言

```bash
uv run python subtitle.py --language zh
```

### 組合使用

```bash
uv run python subtitle.py -m whisper-large-v2-taiwanese-hakka-v1-mlx -l zh -t transcribe
```

## 參數說明

| 參數 | 簡寫 | 說明 | 預設值 |
|------|------|------|--------|
| `--model` | `-m` | 模型名稱或路徑 | 第一個可用模型 |
| `--task` | `-t` | `transcribe`（轉錄）或 `translate`（翻譯成英文）| `transcribe` |
| `--language` | `-l` | 語言代碼（zh, en, ja...）| 自動偵測 |
| `--list` | | 列出可用模型 | |

## 操作說明

1. 執行程式後，字幕視窗會出現在螢幕底部
2. **拖動**字幕視窗可移動位置
3. 開始全螢幕簡報
4. 對著麥克風說話，字幕會即時顯示
5. 按 **Ctrl+C** 關閉程式

## 自訂設定

編輯 `subtitle.py` 開頭的設定：

### 視窗設定

```python
WINDOW_WIDTH_RATIO = 0.8      # 視窗寬度佔螢幕比例 (0.0 ~ 1.0)
WINDOW_HEIGHT = 100           # 視窗高度 (像素)
WINDOW_BOTTOM_MARGIN = 50     # 視窗距離螢幕底部的距離 (像素)
WINDOW_OPACITY = 0.85         # 視窗透明度 (0.0 ~ 1.0)
```

### 文字設定

```python
FONT_SIZE = 48                # 字體大小 (像素)
FONT_NAME = None              # 字體名稱，None 為系統預設粗體
```

### 顏色設定

```python
BACKGROUND_COLOR = (0.1, 0.1, 0.1)  # 背景顏色 (R, G, B)
TEXT_COLOR = "white"                 # 文字顏色：white / yellow / green / cyan
```

### 錄音設定

```python
SILENCE_THRESHOLD = 500       # 靜音門檻
SILENCE_DURATION = 1.2        # 靜音多久後結束錄音（秒）
```

## 常見調整

| 需求 | 修改 |
|------|------|
| 字更大 | `FONT_SIZE = 56` |
| 字更小 | `FONT_SIZE = 36` |
| 視窗更高 | `WINDOW_HEIGHT = 150` |
| 視窗更窄 | `WINDOW_WIDTH_RATIO = 0.6` |
| 黃色字幕 | `TEXT_COLOR = "yellow"` |
| 更透明 | `WINDOW_OPACITY = 0.7` |
| 環境吵雜 | `SILENCE_THRESHOLD = 800` |
| 說話較快 | `SILENCE_DURATION = 0.8` |

## 簡報流程建議

1. 先轉換模型（如果還沒有的話）：
   ```bash
   cd ../convert
   ./convert.sh formospeech/whisper-large-v2-taiwanese-hakka-v1
   ```

2. 啟動字幕程式，等待模型載入完成：
   ```bash
   cd ../subtitle
   uv run python subtitle.py
   ```

3. 調整字幕視窗位置

4. 開始全螢幕簡報

5. 正常說話，字幕會自動顯示

## 疑難排解

### 找不到模型

請先轉換模型：
```bash
cd ../convert
./convert.sh <hf-repo>
```

### 麥克風沒有反應

確認終端機有麥克風權限：
**系統設定** → **隱私與安全性** → **麥克風** → 勾選終端機

### 辨識品質不佳

- 說話清晰、語速適中
- 減少背景噪音
- 每句話之間稍微停頓

### 環境太吵

提高 `SILENCE_THRESHOLD` 的值，例如改成 `800` 或 `1000`。

### 字幕更新太慢

降低 `SILENCE_DURATION` 的值，例如改成 `0.8` 或 `1.0`。
