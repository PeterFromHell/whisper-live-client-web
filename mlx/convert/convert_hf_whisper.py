"""
將 HuggingFace Whisper 模型轉換為 MLX 格式

支援轉換 transformers 格式的 Whisper 模型（如 formospeech/whisper-large-v2-taiwanese-hakka-v1）
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open


# Whisper 模型配置對應
WHISPER_CONFIGS = {
    "tiny": {"n_mels": 80, "n_audio_ctx": 1500, "n_audio_state": 384, "n_audio_head": 6, "n_audio_layer": 4, "n_vocab": 51865, "n_text_ctx": 448, "n_text_state": 384, "n_text_head": 6, "n_text_layer": 4},
    "base": {"n_mels": 80, "n_audio_ctx": 1500, "n_audio_state": 512, "n_audio_head": 8, "n_audio_layer": 6, "n_vocab": 51865, "n_text_ctx": 448, "n_text_state": 512, "n_text_head": 8, "n_text_layer": 6},
    "small": {"n_mels": 80, "n_audio_ctx": 1500, "n_audio_state": 768, "n_audio_head": 12, "n_audio_layer": 12, "n_vocab": 51865, "n_text_ctx": 448, "n_text_state": 768, "n_text_head": 12, "n_text_layer": 12},
    "medium": {"n_mels": 80, "n_audio_ctx": 1500, "n_audio_state": 1024, "n_audio_head": 16, "n_audio_layer": 24, "n_vocab": 51865, "n_text_ctx": 448, "n_text_state": 1024, "n_text_head": 16, "n_text_layer": 24},
    "large": {"n_mels": 80, "n_audio_ctx": 1500, "n_audio_state": 1280, "n_audio_head": 20, "n_audio_layer": 32, "n_vocab": 51865, "n_text_ctx": 448, "n_text_state": 1280, "n_text_head": 20, "n_text_layer": 32},
    "large-v2": {"n_mels": 80, "n_audio_ctx": 1500, "n_audio_state": 1280, "n_audio_head": 20, "n_audio_layer": 32, "n_vocab": 51865, "n_text_ctx": 448, "n_text_state": 1280, "n_text_head": 20, "n_text_layer": 32},
    "large-v3": {"n_mels": 128, "n_audio_ctx": 1500, "n_audio_state": 1280, "n_audio_head": 20, "n_audio_layer": 32, "n_vocab": 51866, "n_text_ctx": 448, "n_text_state": 1280, "n_text_head": 20, "n_text_layer": 32},
}


def get_config_from_hf(hf_config):
    """從 HuggingFace config 提取 MLX Whisper 配置"""
    # 根據 d_model 判斷模型大小
    d_model = hf_config.get("d_model", 1280)
    
    # 判斷是 large-v3 還是其他版本
    n_mels = hf_config.get("num_mel_bins", 80)
    n_vocab = hf_config.get("vocab_size", 51865)
    
    config = {
        "n_mels": n_mels,
        "n_audio_ctx": hf_config.get("max_source_positions", 1500),
        "n_audio_state": d_model,
        "n_audio_head": hf_config.get("encoder_attention_heads", 20),
        "n_audio_layer": hf_config.get("encoder_layers", 32),
        "n_vocab": n_vocab,
        "n_text_ctx": hf_config.get("max_target_positions", 448),
        "n_text_state": d_model,
        "n_text_head": hf_config.get("decoder_attention_heads", 20),
        "n_text_layer": hf_config.get("decoder_layers", 32),
    }
    
    return config


def convert_hf_to_mlx_weights(hf_weights):
    """將 HuggingFace 權重轉換為 MLX 格式"""
    mlx_weights = {}
    
    for key, value in hf_weights.items():
        # 轉換鍵名
        new_key = key
        
        # encoder 權重轉換
        new_key = new_key.replace("model.encoder.", "encoder.")
        new_key = new_key.replace("model.decoder.", "decoder.")
        
        # 位置編碼
        new_key = new_key.replace("embed_positions.weight", "positional_embedding")
        
        # Conv 層
        new_key = new_key.replace("conv1.weight", "conv1.weight")
        new_key = new_key.replace("conv1.bias", "conv1.bias")
        new_key = new_key.replace("conv2.weight", "conv2.weight")
        new_key = new_key.replace("conv2.bias", "conv2.bias")
        
        # Attention 層
        new_key = new_key.replace("self_attn.k_proj", "attn.key")
        new_key = new_key.replace("self_attn.v_proj", "attn.value")
        new_key = new_key.replace("self_attn.q_proj", "attn.query")
        new_key = new_key.replace("self_attn.out_proj", "attn.out")
        
        # Cross attention (decoder)
        new_key = new_key.replace("encoder_attn.k_proj", "cross_attn.key")
        new_key = new_key.replace("encoder_attn.v_proj", "cross_attn.value")
        new_key = new_key.replace("encoder_attn.q_proj", "cross_attn.query")
        new_key = new_key.replace("encoder_attn.out_proj", "cross_attn.out")
        
        # Layer norm
        new_key = new_key.replace("self_attn_layer_norm", "attn_ln")
        new_key = new_key.replace("encoder_attn_layer_norm", "cross_attn_ln")
        new_key = new_key.replace("final_layer_norm", "mlp_ln")
        new_key = new_key.replace("layer_norm", "ln_post")
        
        # MLP 層
        new_key = new_key.replace("fc1", "mlp.0")
        new_key = new_key.replace("fc2", "mlp.2")
        
        # Embedding
        new_key = new_key.replace("embed_tokens.weight", "token_embedding.weight")
        new_key = new_key.replace("proj_out.weight", "token_embedding.weight")  # 有些模型共享權重
        
        # layers -> blocks
        new_key = new_key.replace(".layers.", ".blocks.")
        
        # 轉換 numpy array
        if hasattr(value, 'numpy'):
            value = value.numpy()
        
        # Conv 權重需要轉置 (PyTorch: out_ch, in_ch, kernel -> MLX: out_ch, kernel, in_ch)
        if 'conv1.weight' in new_key or 'conv2.weight' in new_key:
            if len(value.shape) == 3:
                value = np.transpose(value, (0, 2, 1))
        
        mlx_weights[new_key] = value
    
    return mlx_weights


def load_safetensors(model_path):
    """載入 safetensors 格式的權重"""
    weights = {}
    safetensor_files = list(Path(model_path).glob("*.safetensors"))
    
    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="numpy") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    
    return weights


def load_pytorch_weights(model_path):
    """載入 PyTorch 格式的權重"""
    import torch
    
    weights = {}
    pt_files = list(Path(model_path).glob("pytorch_model*.bin"))
    if not pt_files:
        pt_files = list(Path(model_path).glob("model.bin"))
    
    for pt_file in pt_files:
        state_dict = torch.load(pt_file, map_location="cpu")
        for key, value in state_dict.items():
            weights[key] = value.numpy()
    
    return weights


def convert_model(hf_repo: str, output_path: str, dtype: str = "float16"):
    """
    轉換 HuggingFace Whisper 模型到 MLX 格式
    
    Args:
        hf_repo: HuggingFace 模型路徑（如 formospeech/whisper-large-v2-taiwanese-hakka-v1）
        output_path: 輸出目錄
        dtype: 輸出數據類型 (float16 或 float32)
    """
    print(f"下載模型: {hf_repo}")
    
    # 下載模型
    model_path = snapshot_download(
        repo_id=hf_repo,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt"],
    )
    model_path = Path(model_path)
    print(f"模型已下載到: {model_path}")
    
    # 載入 HuggingFace config
    config_file = model_path / "config.json"
    with open(config_file) as f:
        hf_config = json.load(f)
    
    print(f"模型類型: {hf_config.get('model_type', 'whisper')}")
    print(f"d_model: {hf_config.get('d_model', 'N/A')}")
    
    # 轉換配置
    mlx_config = get_config_from_hf(hf_config)
    mlx_config["model_type"] = "whisper"
    
    print(f"MLX 配置:")
    for k, v in mlx_config.items():
        print(f"  {k}: {v}")
    
    # 載入權重
    print("\n載入權重...")
    if list(model_path.glob("*.safetensors")):
        weights = load_safetensors(model_path)
        print("使用 safetensors 格式")
    else:
        weights = load_pytorch_weights(model_path)
        print("使用 PyTorch 格式")
    
    print(f"原始權重數量: {len(weights)}")
    
    # 轉換權重
    print("\n轉換權重格式...")
    mlx_weights = convert_hf_to_mlx_weights(weights)
    print(f"轉換後權重數量: {len(mlx_weights)}")
    
    # 轉換數據類型
    if dtype == "float16":
        print("\n轉換為 float16...")
        for key in mlx_weights:
            if mlx_weights[key].dtype == np.float32:
                mlx_weights[key] = mlx_weights[key].astype(np.float16)
    
    # 保存模型
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存權重
    weights_file = output_path / "weights.npz"
    print(f"\n保存權重到: {weights_file}")
    np.savez(str(weights_file), **mlx_weights)
    
    # 保存配置
    config_file = output_path / "config.json"
    print(f"保存配置到: {config_file}")
    with open(config_file, "w") as f:
        json.dump(mlx_config, f, indent=2)
    
    print(f"\n✅ 轉換完成！模型已保存到: {output_path}")
    print(f"\n使用方式:")
    print(f'  import mlx_whisper')
    print(f'  result = mlx_whisper.transcribe("audio.wav", path_or_hf_repo="{output_path}")')
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="將 HuggingFace Whisper 模型轉換為 MLX 格式")
    parser.add_argument(
        "--hf-repo",
        type=str,
        required=True,
        help="HuggingFace 模型路徑 (如 formospeech/whisper-large-v2-taiwanese-hakka-v1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./mlx_model",
        help="輸出目錄 (預設: ./mlx_model)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32"],
        default="float16",
        help="輸出數據類型 (預設: float16)",
    )
    
    args = parser.parse_args()
    convert_model(args.hf_repo, args.output, args.dtype)


if __name__ == "__main__":
    main()
