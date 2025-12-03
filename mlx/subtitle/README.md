# 即時字幕浮動視窗

在簡報時顯示即時英文翻譯字幕，字幕視窗會顯示在**全螢幕簡報上方**。

## 特色

- ✅ 即時中文轉英文翻譯
- ✅ 浮動視窗，始終在最上層
- ✅ 支援全螢幕模式（Google Slides、PowerPoint、Keynote）
- ✅ 可拖動調整位置
- ✅ 使用 Apple Silicon GPU 加速
- ✅ 可自訂視窗大小、字體大小、顏色

## 設置

```bash
cd whisper-live-client/mlx/subtitle
uv venv
uv pip install mlx-whisper pyaudio numpy pyobjc-framework-Cocoa
```

## 使用方式

```bash
uv run python floating_subtitle_native.py
```

## 操作說明

1. 執行程式後，字幕視窗會出現在螢幕底部
2. **拖動**字幕視窗可移動位置
3. 開啟 Google Slides 並進入全螢幕簡報模式
4. 對著麥克風說中文，英文翻譯會即時顯示
5. 按 **Ctrl+C** 關閉程式

## 自訂設定

編輯 `floating_subtitle_native.py` 開頭的設定：

### 視窗設定

```python
WINDOW_WIDTH_RATIO = 0.8      # 視窗寬度佔螢幕比例 (0.0 ~ 1.0)
WINDOW_HEIGHT = 100           # 視窗高度 (像素)
WINDOW_BOTTOM_MARGIN = 50     # 視窗距離螢幕底部的距離 (像素)
WINDOW_OPACITY = 0.85         # 視窗透明度 (0.0 ~ 1.0，1.0 為不透明)
```

### 文字設定

```python
FONT_SIZE = 28                # 字體大小 (像素)
FONT_NAME = None              # 字體名稱，None 為系統預設粗體
                              # 可改為 "PingFang TC"、"Helvetica Neue" 等
```

### 顏色設定

```python
BACKGROUND_COLOR = (0.1, 0.1, 0.1)  # 背景顏色 (R, G, B)，範圍 0.0 ~ 1.0
TEXT_COLOR = "white"                 # 文字顏色：white / yellow / green / cyan
```

### 錄音設定

```python
SILENCE_THRESHOLD = 500       # 靜音門檻（數值越高，需要越大聲才會開始錄音）
SILENCE_DURATION = 1.2        # 靜音多久後結束錄音（秒）
```

### 常見調整範例

| 需求 | 修改 |
|------|------|
| 字更大 | `FONT_SIZE = 36` |
| 字更小 | `FONT_SIZE = 22` |
| 視窗更高（多行文字） | `WINDOW_HEIGHT = 150` |
| 視窗更窄 | `WINDOW_WIDTH_RATIO = 0.6` |
| 視窗更寬 | `WINDOW_WIDTH_RATIO = 0.95` |
| 黃色字幕 | `TEXT_COLOR = "yellow"` |
| 更透明背景 | `WINDOW_OPACITY = 0.7` |
| 更不透明背景 | `WINDOW_OPACITY = 0.95` |
| 環境吵雜時 | `SILENCE_THRESHOLD = 800` |
| 說話較快時 | `SILENCE_DURATION = 0.8` |

## 簡報流程建議

1. 先啟動字幕程式，等待模型載入完成
2. 調整字幕視窗位置到不會遮擋重要內容的地方
3. 開始 Google Slides 全螢幕簡報
4. 正常說話，字幕會自動顯示

## 注意事項

- 首次執行會下載約 3GB 的模型
- 使用 `whisper-large-v3` 模型（支援翻譯）
- 翻譯有約 1-2 秒的延遲
- 說完一句話後會顯示翻譯

## 疑難排解

### 字幕沒有顯示在全螢幕上方

請使用 `floating_subtitle_native.py` 版本，它使用 macOS 原生 API 確保視窗層級。

### 麥克風沒有反應

確認終端機有麥克風權限：
**系統設定** → **隱私與安全性** → **麥克風** → 勾選終端機

### 翻譯品質不佳

- 說話清晰、語速適中
- 減少背景噪音
- 每句話之間稍微停頓

### 環境太吵，一直誤觸發

提高 `SILENCE_THRESHOLD` 的值，例如改成 `800` 或 `1000`。

### 字幕更新太慢

降低 `SILENCE_DURATION` 的值，例如改成 `0.8` 或 `1.0`。
