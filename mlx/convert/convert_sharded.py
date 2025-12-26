"""
將 HuggingFace Whisper 模型轉換為 MLX 格式
支援分片 (sharded) safetensors 格式

專為 formospeech/whisper-large-v2-taiwanese-hakka-v1 設計
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open


def load_sharded_safetensors(model_path: Path) -> dict:
    """載入分片的 safetensors 權重"""
    weights = {}
    
    # 找出所有 safetensors 檔案
    safetensor_files = sorted(model_path.glob("model*.safetensors"))
    
    # 排除 index 檔案
    safetensor_files = [f for f in safetensor_files if "index" not in f.name]
    
    print(f"找到 {len(safetensor_files)} 個權重檔案")
    
    for sf_file in safetensor_files:
        print(f"  載入: {sf_file.name}")
        with safe_open(sf_file, framework="numpy") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    
    return weights


def get_mlx_config(hf_config: dict) -> dict:
    """從 HuggingFace config 建立 MLX Whisper 配置"""
    return {
        "n_mels": hf_config.get("num_mel_bins", 80),
        "n_audio_ctx": hf_config.get("max_source_positions", 1500),
        "n_audio_state": hf_config.get("d_model", 1280),
        "n_audio_head": hf_config.get("encoder_attention_heads", 20),
        "n_audio_layer": hf_config.get("encoder_layers", 32),
        "n_vocab": hf_config.get("vocab_size", 51865),
        "n_text_ctx": hf_config.get("max_target_positions", 448),
        "n_text_state": hf_config.get("d_model", 1280),
        "n_text_head": hf_config.get("decoder_attention_heads", 20),
        "n_text_layer": hf_config.get("decoder_layers", 32),
        "model_type": "whisper",
    }


def convert_key(key: str) -> str:
    """轉換 HuggingFace 權重鍵名為 MLX 格式"""
    # 移除 model. 前綴
    key = key.replace("model.", "")
    
    # 位置編碼
    key = key.replace("encoder.embed_positions.weight", "encoder.positional_embedding")
    key = key.replace("decoder.embed_positions.weight", "decoder.positional_embedding")
    
    # Token embedding
    key = key.replace("decoder.embed_tokens.weight", "decoder.token_embedding.weight")
    
    # layers -> blocks
    key = key.replace(".layers.", ".blocks.")
    
    # Self attention
    key = key.replace(".self_attn.k_proj.", ".attn.key.")
    key = key.replace(".self_attn.v_proj.", ".attn.value.")
    key = key.replace(".self_attn.q_proj.", ".attn.query.")
    key = key.replace(".self_attn.out_proj.", ".attn.out.")
    
    # Cross attention (decoder only)
    key = key.replace(".encoder_attn.k_proj.", ".cross_attn.key.")
    key = key.replace(".encoder_attn.v_proj.", ".cross_attn.value.")
    key = key.replace(".encoder_attn.q_proj.", ".cross_attn.query.")
    key = key.replace(".encoder_attn.out_proj.", ".cross_attn.out.")
    
    # Layer norms
    key = key.replace(".self_attn_layer_norm.", ".attn_ln.")
    key = key.replace(".encoder_attn_layer_norm.", ".cross_attn_ln.")
    key = key.replace(".final_layer_norm.", ".mlp_ln.")
    key = key.replace("encoder.layer_norm.", "encoder.ln_post.")
    key = key.replace("decoder.layer_norm.", "decoder.ln_post.")
    
    # MLP
    key = key.replace(".fc1.", ".mlp.0.")
    key = key.replace(".fc2.", ".mlp.2.")
    
    return key


def convert_weights(hf_weights: dict) -> dict:
    """轉換所有權重為 MLX 格式"""
    mlx_weights = {}
    
    for hf_key, value in hf_weights.items():
        mlx_key = convert_key(hf_key)
        
        # Conv 權重需要轉置: (out, in, kernel) -> (out, kernel, in)
        if "conv1.weight" in mlx_key or "conv2.weight" in mlx_key:
            if len(value.shape) == 3:
                value = np.transpose(value, (0, 2, 1))
        
        mlx_weights[mlx_key] = value
    
    return mlx_weights


def convert_model(hf_repo: str, output_path: str, dtype: str = "float16"):
    """
    轉換 HuggingFace Whisper 模型到 MLX 格式
    """
    print(f"=== 轉換 {hf_repo} ===\n")
    
    # 下載模型
    print("步驟 1: 下載模型")
    model_path = Path(snapshot_download(
        repo_id=hf_repo,
        allow_patterns=["*.json", "*.safetensors", "*.txt"],
    ))
    print(f"模型路徑: {model_path}\n")
    
    # 載入 config
    print("步驟 2: 載入配置")
    with open(model_path / "config.json") as f:
        hf_config = json.load(f)
    
    mlx_config = get_mlx_config(hf_config)
    print(f"  n_audio_layer: {mlx_config['n_audio_layer']}")
    print(f"  n_text_layer: {mlx_config['n_text_layer']}")
    print(f"  n_audio_state: {mlx_config['n_audio_state']}")
    print(f"  n_vocab: {mlx_config['n_vocab']}\n")
    
    # 載入權重
    print("步驟 3: 載入權重")
    hf_weights = load_sharded_safetensors(model_path)
    print(f"  總共 {len(hf_weights)} 個張量\n")
    
    # 轉換權重
    print("步驟 4: 轉換權重格式")
    mlx_weights = convert_weights(hf_weights)
    print(f"  轉換完成: {len(mlx_weights)} 個張量\n")
    
    # 轉換數據類型
    if dtype == "float16":
        print("步驟 5: 轉換為 float16")
        for key in mlx_weights:
            if mlx_weights[key].dtype in [np.float32, np.float64]:
                mlx_weights[key] = mlx_weights[key].astype(np.float16)
        print("  完成\n")
    
    # 保存模型
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("步驟 6: 保存模型")
    
    # 保存權重
    weights_file = output_path / "weights.npz"
    print(f"  保存權重: {weights_file}")
    np.savez(str(weights_file), **mlx_weights)
    
    # 保存配置
    config_file = output_path / "config.json"
    print(f"  保存配置: {config_file}")
    with open(config_file, "w") as f:
        json.dump(mlx_config, f, indent=2)
    
    # 計算檔案大小
    weights_size = weights_file.stat().st_size / (1024 * 1024 * 1024)
    
    print(f"\n=== 轉換完成！===")
    print(f"輸出目錄: {output_path}")
    print(f"權重大小: {weights_size:.2f} GB")
    print(f"\n使用方式:")
    print(f'  import mlx_whisper')
    print(f'  result = mlx_whisper.transcribe("audio.wav", path_or_hf_repo="{output_path}")')


def main():
    parser = argparse.ArgumentParser(description="將 HuggingFace Whisper 模型轉換為 MLX 格式")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="formospeech/whisper-large-v2-taiwanese-hakka-v1",
        help="HuggingFace 模型路徑",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../hakka-v1-mlx",
        help="輸出目錄",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32"],
        default="float16",
        help="輸出數據類型",
    )
    
    args = parser.parse_args()
    convert_model(args.hf_repo, args.output, args.dtype)


if __name__ == "__main__":
    main()
