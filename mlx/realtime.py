"""
MLX Whisper 即時語音辨識（使用 Apple Silicon GPU）
支援使用本地轉換的模型

使用方式:
  # 使用預設設定（transcribe, 自動偵測語言）
  uv run python realtime.py
  
  # 指定模型
  uv run python realtime.py --model whisper-large-v2-taiwanese-hakka-v1-mlx
  
  # 翻譯成英文
  uv run python realtime.py --task translate
  
  # 指定語言
  uv run python realtime.py --language zh
  
  # 列出可用模型
  uv run python realtime.py --list
"""
import argparse
import sys
import numpy as np
import pyaudio
import mlx_whisper
from pathlib import Path

# ===========================================
# 路徑設定
# ===========================================
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"

# ===========================================
# 錄音設定
# ===========================================
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5


def list_available_models() -> list[str]:
    """列出所有可用的本地模型"""
    if not MODELS_DIR.exists():
        return []
    
    models = []
    for path in MODELS_DIR.iterdir():
        if path.is_dir():
            config_file = path / "config.json"
            weights_file = path / "weights.npz"
            if config_file.exists() and weights_file.exists():
                models.append(path.name)
    
    return sorted(models)


def get_model_path(model_name: str) -> Path:
    """取得模型路徑"""
    # 如果是完整路徑
    if "/" in model_name or model_name.startswith("."):
        return Path(model_name)
    
    # 否則在 models 目錄中尋找
    model_path = MODELS_DIR / model_name
    
    # 嘗試加上 -mlx 後綴
    if not model_path.exists() and not model_name.endswith("-mlx"):
        model_path = MODELS_DIR / f"{model_name}-mlx"
    
    return model_path


def get_audio_level(data):
    """計算音量"""
    samples = np.frombuffer(data, dtype=np.int16)
    return np.abs(samples).mean()


def record_until_silence(stream):
    """錄音直到靜音"""
    frames = []
    silent_chunks = 0
    chunks_for_silence = int(SILENCE_DURATION * RATE / CHUNK)
    is_speaking = False
    
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
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


def transcribe_audio(audio_data, model_path: str, language: str | None, task: str):
    """使用 MLX Whisper 辨識"""
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    kwargs = {
        "path_or_hf_repo": model_path,
        "task": task,
    }
    
    # 只有在指定語言時才傳入
    if language:
        kwargs["language"] = language
    
    result = mlx_whisper.transcribe(audio_np, **kwargs)
    
    return result["text"].strip()


def main():
    parser = argparse.ArgumentParser(
        description="MLX Whisper 即時語音辨識（使用本地模型）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本使用（自動偵測語言，純轉錄）
  uv run python realtime.py
  
  # 翻譯成英文
  uv run python realtime.py --task translate
  
  # 指定語言為中文
  uv run python realtime.py --language zh
  
  # 使用特定模型
  uv run python realtime.py --model whisper-large-v2-taiwanese-hakka-v1-mlx
  
  # 組合使用
  uv run python realtime.py -m whisper-large-v2-taiwanese-hakka-v1-mlx -l zh -t transcribe
"""
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="模型名稱或路徑（預設：使用第一個可用的本地模型）",
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        choices=["transcribe", "translate"],
        default="transcribe",
        help="任務類型：transcribe（轉錄）或 translate（翻譯成英文）（預設：transcribe）",
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        help="語言代碼，如 zh, en, ja（預設：自動偵測）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出可用的本地模型",
    )
    
    args = parser.parse_args()
    
    # 列出模型
    if args.list:
        models = list_available_models()
        if models:
            print("可用的本地模型:")
            for m in models:
                print(f"  • {m}")
        else:
            print("尚未有轉換的模型")
            print(f"\n請先使用 convert/convert.sh 轉換模型:")
            print(f"  cd convert")
            print(f"  ./convert.sh <hf-repo>")
        return
    
    # 取得模型
    available_models = list_available_models()
    
    if args.model:
        model_path = get_model_path(args.model)
    elif available_models:
        # 使用第一個可用的模型
        model_path = MODELS_DIR / available_models[0]
        print(f"使用模型: {available_models[0]}")
    else:
        print("錯誤：找不到任何本地模型")
        print(f"\n請先轉換模型:")
        print(f"  cd convert")
        print(f"  ./convert.sh <hf-repo>")
        print(f"\n範例:")
        print(f"  ./convert.sh formospeech/whisper-large-v2-taiwanese-hakka-v1")
        sys.exit(1)
    
    # 檢查模型是否存在
    if not model_path.exists():
        print(f"錯誤：找不到模型 {model_path}")
        if available_models:
            print(f"\n可用的模型:")
            for m in available_models:
                print(f"  • {m}")
        sys.exit(1)
    
    # 檢查模型檔案
    if not (model_path / "config.json").exists() or not (model_path / "weights.npz").exists():
        print(f"錯誤：模型不完整 {model_path}")
        print("需要 config.json 和 weights.npz")
        sys.exit(1)
    
    # 顯示設定
    task_display = "轉錄" if args.task == "transcribe" else "翻譯成英文"
    lang_display = args.language if args.language else "自動偵測"
    
    print("=" * 50)
    print("MLX Whisper 即時語音辨識")
    print("使用 Apple Silicon GPU 加速")
    print("=" * 50)
    print(f"模型: {model_path.name}")
    print(f"任務: {task_display}")
    print(f"語言: {lang_display}")
    print("=" * 50)
    print("\n說話後，文字會即時顯示")
    print("按 Ctrl+C 停止\n")
    print("-" * 50)
    
    # 初始化 PyAudio
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    print("正在載入模型...")
    
    # 預熱模型
    dummy = np.zeros(RATE, dtype=np.float32)
    warmup_kwargs = {"path_or_hf_repo": str(model_path), "task": args.task}
    if args.language:
        warmup_kwargs["language"] = args.language
    mlx_whisper.transcribe(dummy, **warmup_kwargs)
    print("模型載入完成！開始監聽...\n")
    
    try:
        while True:
            print("🎤 等待說話...", end="\r")
            audio_data = record_until_silence(stream)
            
            if len(audio_data) > CHUNK * 10:
                print("⏳ 辨識中...   ", end="\r")
                text = transcribe_audio(audio_data, str(model_path), args.language, args.task)
                if text:
                    print(f"📝 {text}")
    
    except KeyboardInterrupt:
        print("\n\n正在關閉...")
    
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("已停止")


if __name__ == "__main__":
    main()
