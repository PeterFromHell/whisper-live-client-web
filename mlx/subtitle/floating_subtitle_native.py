"""
即時字幕浮動視窗 - macOS 原生版本
使用 PyObjC 確保在全螢幕簡報上方也能顯示
"""
import signal
import sys
import threading
import queue
import numpy as np
import pyaudio
import mlx_whisper

import AppKit
from AppKit import (
    NSApplication, NSWindow, NSTextField, NSColor, NSFont,
    NSWindowStyleMaskBorderless, NSBackingStoreBuffered,
    NSScreenSaverWindowLevel,
    NSMakeRect, NSScreen,
    NSTextAlignmentCenter,
    NSApplicationActivationPolicyAccessory
)
from PyObjCTools import AppHelper

# ===========================================
# 📐 視窗設定（可自行調整）
# ===========================================
WINDOW_WIDTH_RATIO = 0.8      # 視窗寬度佔螢幕比例 (0.0 ~ 1.0)
WINDOW_HEIGHT = 100           # 視窗高度 (像素)
WINDOW_BOTTOM_MARGIN = 50     # 視窗距離螢幕底部的距離 (像素)
WINDOW_OPACITY = 0.85         # 視窗透明度 (0.0 ~ 1.0，1.0 為不透明)

# ===========================================
# 🔤 文字設定（可自行調整）
# ===========================================
FONT_SIZE = 48                # 字體大小 (像素)
FONT_NAME = None              # 字體名稱，None 為系統預設粗體
                              # 可改為 "PingFang TC"、"Helvetica Neue" 等

# ===========================================
# 🎨 顏色設定（可自行調整）
# ===========================================
# 背景顏色 (R, G, B)，範圍 0.0 ~ 1.0
BACKGROUND_COLOR = (0.1, 0.1, 0.1)  # 深灰色
# 文字顏色：使用 "white" 或 "yellow" 或 "green"
TEXT_COLOR = "white"

# ===========================================
# 🎤 模型設定
# ===========================================
MODEL_NAME = "mlx-community/whisper-large-v3-mlx"

# ===========================================
# 🎙️ 錄音設定
# ===========================================
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 500       # 靜音門檻（數值越高，需要越大聲才會開始錄音）
SILENCE_DURATION = 1.2        # 靜音多久後結束錄音（秒）

# 全域變數
running = True


def get_text_color():
    """取得文字顏色"""
    colors = {
        "white": NSColor.whiteColor(),
        "yellow": NSColor.yellowColor(),
        "green": NSColor.greenColor(),
        "cyan": NSColor.cyanColor(),
    }
    return colors.get(TEXT_COLOR, NSColor.whiteColor())


class SubtitleWindow:
    def __init__(self):
        # 取得主螢幕尺寸
        screen = NSScreen.mainScreen()
        screen_frame = screen.frame()
        screen_width = screen_frame.size.width
        screen_height = screen_frame.size.height
        
        # 視窗尺寸和位置
        window_width = screen_width * WINDOW_WIDTH_RATIO
        window_height = WINDOW_HEIGHT
        x = (screen_width - window_width) / 2
        y = WINDOW_BOTTOM_MARGIN
        
        # 建立視窗
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, window_width, window_height),
            NSWindowStyleMaskBorderless,
            NSBackingStoreBuffered,
            False
        )
        
        # 視窗設定：始終在最上層，包括全螢幕應用上方
        self.window.setLevel_(NSScreenSaverWindowLevel)
        self.window.setOpaque_(False)
        self.window.setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                BACKGROUND_COLOR[0], 
                BACKGROUND_COLOR[1], 
                BACKGROUND_COLOR[2], 
                WINDOW_OPACITY
            )
        )
        self.window.setHasShadow_(True)
        self.window.setMovableByWindowBackground_(True)
        self.window.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces |
            AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        
        # 建立文字標籤
        content_view = self.window.contentView()
        self.label = NSTextField.alloc().initWithFrame_(
            NSMakeRect(20, 10, window_width - 40, window_height - 20)
        )
        self.label.setStringValue_("🎤 等待說話...")
        
        # 設定字體
        if FONT_NAME:
            font = NSFont.fontWithName_size_(FONT_NAME, FONT_SIZE)
            if font is None:
                font = NSFont.boldSystemFontOfSize_(FONT_SIZE)
        else:
            font = NSFont.boldSystemFontOfSize_(FONT_SIZE)
        self.label.setFont_(font)
        
        # 設定文字顏色
        self.label.setTextColor_(get_text_color())
        self.label.setBackgroundColor_(NSColor.clearColor())
        self.label.setBezeled_(False)
        self.label.setEditable_(False)
        self.label.setSelectable_(False)
        self.label.setAlignment_(NSTextAlignmentCenter)
        
        content_view.addSubview_(self.label)
        
        # 顯示視窗
        self.window.makeKeyAndOrderFront_(None)
    
    def update_text(self, text):
        """更新字幕文字（執行緒安全）"""
        def update():
            self.label.setStringValue_(text)
        AppHelper.callAfter(update)
    
    def close(self):
        def do_close():
            self.window.close()
            AppHelper.stopEventLoop()
        AppHelper.callAfter(do_close)


def get_audio_level(data):
    samples = np.frombuffer(data, dtype=np.int16)
    return np.abs(samples).mean()


def record_until_silence(stream):
    global running
    frames = []
    silent_chunks = 0
    chunks_for_silence = int(SILENCE_DURATION * RATE / CHUNK)
    is_speaking = False
    
    while running:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
        except:
            break
        level = get_audio_level(data)
        
        if level > SILENCE_THRESHOLD:
            is_speaking = True
            silent_chunks = 0
            frames.append(data)
        elif is_speaking:
            frames.append(data)
            silent_chunks += 1
            if silent_chunks > chunks_for_silence:
                break
    
    return b''.join(frames)


def transcribe_audio(audio_data):
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    result = mlx_whisper.transcribe(
        audio_np,
        path_or_hf_repo=MODEL_NAME,
        language="zh",
        task="translate",
    )
    
    return result["text"].strip()


def audio_thread(subtitle_window):
    """錄音和翻譯的執行緒"""
    global running
    
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    subtitle_window.update_text("⏳ 載入模型中...")
    
    # 預熱模型
    dummy = np.zeros(RATE, dtype=np.float32)
    mlx_whisper.transcribe(dummy, path_or_hf_repo=MODEL_NAME)
    
    subtitle_window.update_text("🎤 準備就緒，開始說話...")
    
    try:
        while running:
            audio_data = record_until_silence(stream)
            
            if not running:
                break
            
            if len(audio_data) > CHUNK * 10:
                subtitle_window.update_text("⏳ 翻譯中...")
                text = transcribe_audio(audio_data)
                if text and running:
                    subtitle_window.update_text(text)
    
    except Exception as e:
        if running:
            subtitle_window.update_text(f"錯誤: {str(e)}")
    
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


def signal_handler(signum, frame):
    """處理 Ctrl+C 信號"""
    global running
    print("\n\n正在關閉...")
    running = False
    AppHelper.stopEventLoop()


def main():
    global running
    
    print("=" * 50)
    print("即時字幕浮動視窗 (macOS 原生版)")
    print("=" * 50)
    print("\n目前設定：")
    print(f"  視窗寬度：螢幕的 {int(WINDOW_WIDTH_RATIO * 100)}%")
    print(f"  視窗高度：{WINDOW_HEIGHT} 像素")
    print(f"  字體大小：{FONT_SIZE} 像素")
    print(f"  文字顏色：{TEXT_COLOR}")
    print("\n操作說明：")
    print("  • 拖動字幕視窗可移動位置")
    print("  • 按 Ctrl+C 關閉程式")
    print("  • 說中文，會顯示英文翻譯")
    print("  • 會顯示在全螢幕簡報上方")
    print("\n正在啟動...\n")
    
    # 設定信號處理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 初始化應用程式
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    
    # 建立字幕視窗
    subtitle_window = SubtitleWindow()
    
    # 在背景執行緒中執行錄音和翻譯
    thread = threading.Thread(target=audio_thread, args=(subtitle_window,), daemon=True)
    thread.start()
    
    # 設定定時器來檢查是否需要關閉
    def check_running():
        if not running:
            AppHelper.stopEventLoop()
        else:
            threading.Timer(0.5, lambda: AppHelper.callAfter(check_running)).start()
    
    AppHelper.callAfter(check_running)
    
    # 執行主迴圈
    AppHelper.runEventLoop()
    
    print("已關閉")


if __name__ == "__main__":
    main()
