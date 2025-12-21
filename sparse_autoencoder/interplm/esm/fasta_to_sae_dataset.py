"""
Convert a directory of FASTA files to a directories of ESM layer activations organized
by layer and shard with specific metadata used for SAE training.

Supports loading custom weights from both .pt and .safetensors format files.
For .safetensors support, install the safetensors package: pip install safetensors
"""
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from esm import FastaBatchedDataset
from torch.utils.data import Subset
from tqdm import tqdm

# from interplm.esm.embed import get_model_converter_alphabet  # Not needed anymore
import esm

try:
    from peft import PeftModel
    peft_available = True
except ImportError:
    peft_available = False

try:
    from safetensors import safe_open
    safetensors_available = True
except ImportError:
    safetensors_available = False

def load_weights_from_file(weight_file: Path, device: torch.device) -> Dict[str, Any]:
    """
    Load weights from either .pt or .safetensors format files.
    
    Args:
        weight_file: Path to the weight file (.pt or .safetensors)
        device: Device to load the weights on
        
    Returns:
        Dictionary containing the loaded weights
        
    Raises:
        ValueError: If the file format is not supported or safetensors is not available
    """
    if not weight_file.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_file}")
    
    file_extension = weight_file.suffix.lower()
    
    if file_extension == '.pt':
        # Load PyTorch format
        return torch.load(weight_file, map_location=device)
    
    elif file_extension == '.safetensors':
        # Load safetensors format
        if not safetensors_available:
            raise ImportError(
                "safetensors is not installed. Please install 'safetensors' to use .safetensors files. "
                "Run: pip install safetensors"
            )
        
        # Load all tensors from safetensors file
        state_dict = {}
        with safe_open(weight_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        
        # Move tensors to the target device after loading
        for key in state_dict:
            state_dict[key] = state_dict[key].to(device)
        
        return state_dict
    
    else:
        raise ValueError(
            f"Unsupported file format: {file_extension}. "
            f"Supported formats are .pt and .safetensors"
        )

def load_esm_model(esm_model_name: str, weight_file: Path | None = None, use_peft: bool = False):
    """Load base ESM model and optionally apply PEFT weights or custom weights."""
    model_loader = getattr(esm.pretrained, esm_model_name, None)
    if model_loader is None:
        raise ValueError(f"esm_model_name '{esm_model_name}' is not valid.")

    base_model, alphabet = model_loader()
    base_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = base_model.to(device)

    if weight_file is not None:
        if use_peft:
            if not peft_available:
                raise ImportError("PEFT is not installed. Please install 'peft' to use LoRA models.")
            print(f"Loading PEFT adapter weights from {weight_file}...")
            base_model = PeftModel.from_pretrained(base_model, weight_file)
            base_model.eval()
            base_model = base_model.to(device)
        else:
            # Load custom weights as regular state dict
            print(f"Loading custom weights from {weight_file}...")
            ckpt = load_weights_from_file(weight_file, device)
            # 典型的な 3 パターンに対応
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                # Direct state_dict (like SFT_hot.pt)
                state_dict = ckpt
                
            # Extract ESM part if keys have 'esm.' prefix (from ESM2_TmRegressor)
            if any(k.startswith('esm.') for k in state_dict.keys()):
                print("Extracting ESM part from full model state_dict...")
                esm_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('esm.'):
                        # Remove 'esm.' prefix to match ESM model structure
                        new_key = k[4:]  # Remove 'esm.'
                        esm_state_dict[new_key] = v
                state_dict = esm_state_dict
                
            # strict=False で不足 / 余剰キーは無視
            missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
            print(
                f"Loaded external weights from {weight_file} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )

    return base_model, alphabet

def get_activations(
    model: torch.nn.Module,
    batch_tokens: torch.Tensor,
    batch_mask: torch.Tensor,
    layers: List[int],
    dtype: torch.dtype = torch.float32, 
) -> dict:
    """
    Extract activations from multiple ESM layers in a memory-efficient way.

    * 元コードとの主な差分 *
      1. torch.inference_mode() で余計なバッファを生成しない
      2. 取得直後に **CPU へ転送 & 指定 dtype にキャスト** して GPU メモリを開放
      3. 余分なテンソルを確実に捨てるため del と empty_cache() を追加
    """
    with torch.inference_mode():                # ★ 2) no_grad より軽量
        results = model(batch_tokens, repr_layers=layers)

    mask = batch_tokens > 2                     # cls/pad/eos を除外
    activations = {}

    for layer in layers:
        rep = results["representations"][layer]          # GPU 上の fp32
        rep = rep[mask]                                  # パディング除外
        # ★ 3) 取得直後に CPU & 半精度へ退避 → GPU メモリから即座に解放
        activations[layer] = rep.to(dtype=dtype, device="cpu").clone()

    # ★ 4) 不要になった GPU テンソルを確実に解放
    del results
    torch.cuda.empty_cache()

    return activations

def embed_fasta_file_for_all_layers(
    esm_model_name: str,
    fasta_file: Path,
    output_dir: Path,
    layers: List[int],
    shard_num: int,
    corrupt_esm: bool = False,
    toks_per_batch: int = 1024,
    truncation_seq_length: int = 1022,
    weight_file: Path | None = None, 
    sequences_per_chunk: int = 100000, # 【変更点】データチャンクのサイズを引数で受け取る
    use_peft: bool = False, # 【変更点】PEFTモデルを使用するかどうかを明示的に指定
    disable_chunking: bool = False, # 【新規追加】chunk処理を無効にするフラグ
):
    """
    Process a FASTA file through an ESM model and save layer activations.

    Processes sequences in batches, extracts activations from specified layers,
    shuffles the results, and saves them along with metadata. Uses GPU if available
    and not explicitly disabled.

    Args:
        esm_model_name: Name of the ESM model being used
        fasta_file: Path to input FASTA file
        output_dir: Directory to save outputs
        layers: List of layer numbers to extract
        shard_num: Current shard number being processed
        corrupt_esm: Whether to use corrupted model parameters
        toks_per_batch: Maximum tokens per batch
        truncation_seq_length: Maximum sequence length before truncation
        weight_file: Path to custom weight file (.pt or .safetensors format)
        sequences_per_chunk: Number of sequences to process in each chunk
        use_peft: Whether to treat weight_file as PEFT adapter weights
        disable_chunking: Whether to disable chunking and process all data at once

    Outputs:
        - Saves activation tensors as .pt files
        - Saves metadata as JSON files
        - Creates directory structure for outputs
    """
    model, alphabet = load_esm_model(esm_model_name, weight_file, use_peft)
    batch_converter = alphabet.get_batch_converter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 【変更点】FASTAファイルを一度だけ読み込み、データセット全体を取得
    full_dataset = FastaBatchedDataset.from_file(fasta_file)
    
    if not disable_chunking:
        print(f"Read {fasta_file} with {len(full_dataset):,} sequences. Processing in chunks of {sequences_per_chunk:,}.")
        # 【変更点】データセットをチャンクに分割してループ処理
        chunk_files = {layer: [] for layer in layers}
        
        for chunk_idx, i in enumerate(range(0, len(full_dataset), sequences_per_chunk)):
            print(f"\n--- Processing data chunk {chunk_idx} for layers {layers} ---")
            
            # 【修正】Subsetを使う代わりに、各チャンクで新しいFastaBatchedDatasetを作成
            start_idx = i
            end_idx = min(i + sequences_per_chunk, len(full_dataset))
            
            # 元のデータセットから現在のチャンクの配列ラベルと文字列を抽出
            chunk_labels = full_dataset.sequence_labels[start_idx:end_idx]
            chunk_strs = full_dataset.sequence_strs[start_idx:end_idx]
            
            # 抽出したデータで、このチャンク専用のFastaBatchedDatasetを作成
            dataset = FastaBatchedDataset(chunk_labels, chunk_strs)
            
            # これで 'dataset' は get_batch_indices メソッドを持つ正しいオブジェクトになる
            batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
            data_loader = torch.utils.data.DataLoader(
                dataset,
                collate_fn=batch_converter,
                batch_sampler=batches,
                num_workers=1, # 警告を避けるため1に変更
                pin_memory=True,
            )

            total_tokens = 0
            all_activations = {layer: [] for layer in layers}

            for (_, _, toks) in tqdm(data_loader, desc=f"Chunk {chunk_idx} Batches"):
                activations = get_activations(model,
                                              toks.to(device),
                                              (toks != alphabet.padding_idx).to(device),
                                              layers=layers)
                for layer in layers:
                    all_activations[layer].append(activations[layer])

                # Count total tokens processed
                total_tokens += activations[layers[0]].shape[0]

                torch.cuda.empty_cache()

            # Save activations and metadata for each layer in the proper directory structure
            for layer in layers:
                # 【変更点】出力ディレクトリ/ファイル名にチャンク番号を追加
                layer_output_dir = output_dir / f"layer_{layer}" / f"shard_{shard_num}"
                layer_output_dir.mkdir(parents=True, exist_ok=True)
                output_file = layer_output_dir / f"activations_chunk_{chunk_idx}.pt"
                metadata_file = layer_output_dir / f"metadata_chunk_{chunk_idx}.json"

                # Concatenate all activations for this layer
                if not all_activations[layer]:
                    print(f"Warning: No activations generated for layer {layer}, chunk {chunk_idx}. Skipping.")
                    continue

                layer_activations = torch.cat(all_activations[layer])

                # Shuffle the activations
                shuffled_indices = torch.randperm(total_tokens)
                layer_activations = layer_activations[shuffled_indices]

                # Save the tensor
                torch.save(layer_activations, output_file)
                chunk_files[layer].append(output_file)
                print(f"Saved activations for layer {layer}, shard {shard_num}, chunk {chunk_idx} to {output_file}")

                # Save metadata
                metadata = {
                    "model": esm_model_name,
                    "total_tokens": total_tokens,
                    "d_model": model.embed_dim,
                    "dtype": str(layer_activations.dtype),
                    "layer": layer,
                    "shard": shard_num,
                    "chunk": chunk_idx,
                }
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f)
        
        # 【新規追加】chunkファイルを結合してactivation.ptを保存
        print(f"\n--- Combining chunk files for layers {layers} ---")
        for layer in layers:
            if not chunk_files[layer]:
                print(f"Warning: No chunk files found for layer {layer}. Skipping combination.")
                continue
                
            layer_output_dir = output_dir / f"layer_{layer}" / f"shard_{shard_num}"
            combined_activations = []
            total_combined_tokens = 0
            
            # 各chunkファイルを読み込んで結合
            for chunk_file in chunk_files[layer]:
                chunk_activations = torch.load(chunk_file)
                combined_activations.append(chunk_activations)
                total_combined_tokens += chunk_activations.shape[0]
                print(f"Loaded chunk file: {chunk_file} with {chunk_activations.shape[0]:,} tokens")
            
            # 全てのchunkを結合
            if combined_activations:
                final_activations = torch.cat(combined_activations, dim=0)
                
                # 最終的なシャッフル
                shuffled_indices = torch.randperm(total_combined_tokens)
                final_activations = final_activations[shuffled_indices]
                
                # activation.ptとして保存
                final_output_file = layer_output_dir / "activations.pt"
                torch.save(final_activations, final_output_file)
                print(f"Combined and saved {total_combined_tokens:,} total tokens for layer {layer} to {final_output_file}")
                
                # 結合後のメタデータを保存
                final_metadata = {
                    "model": esm_model_name,
                    "total_tokens": total_combined_tokens,
                    "d_model": model.embed_dim,
                    "dtype": str(final_activations.dtype),
                    "layer": layer,
                    "shard": shard_num,
                    "num_chunks": len(chunk_files[layer]),
                    "chunk_files": [str(f) for f in chunk_files[layer]],
                }
                final_metadata_file = layer_output_dir / "metadata.json"
                with open(final_metadata_file, "w") as f:
                    json.dump(final_metadata, f)
                print(f"Saved combined metadata to {final_metadata_file}")
    
    else:
        # 【新規追加】chunk処理を無効にした場合の従来の処理
        print(f"Read {fasta_file} with {len(full_dataset):,} sequences. Processing without chunking.")
        
        # データセット全体を一度に処理
        batches = full_dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            full_dataset,
            collate_fn=batch_converter,
            batch_sampler=batches,
            num_workers=1,
            pin_memory=True,
        )

        total_tokens = 0
        all_activations = {layer: [] for layer in layers}

        for (_, _, toks) in tqdm(data_loader, desc="Processing all batches"):
            activations = get_activations(model,
                                          toks.to(device),
                                          (toks != alphabet.padding_idx).to(device),
                                          layers=layers)
            for layer in layers:
                all_activations[layer].append(activations[layer])

            # Count total tokens processed
            total_tokens += activations[layers[0]].shape[0]

            torch.cuda.empty_cache()

        # Save activations and metadata for each layer
        for layer in layers:
            layer_output_dir = output_dir / f"layer_{layer}" / f"shard_{shard_num}"
            layer_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = layer_output_dir / "activations.pt"
            metadata_file = layer_output_dir / "metadata.json"

            # Concatenate all activations for this layer
            if not all_activations[layer]:
                print(f"Warning: No activations generated for layer {layer}. Skipping.")
                continue

            layer_activations = torch.cat(all_activations[layer])

            # Shuffle the activations
            shuffled_indices = torch.randperm(total_tokens)
            layer_activations = layer_activations[shuffled_indices]

            # Save the tensor
            torch.save(layer_activations, output_file)
            print(f"Saved activations for layer {layer}, shard {shard_num} to {output_file}")

            # Save metadata
            metadata = {
                "model": esm_model_name,
                "total_tokens": total_tokens,
                "d_model": model.embed_dim,
                "dtype": str(layer_activations.dtype),
                "layer": layer,
                "shard": shard_num,
                "chunking_enabled": False,
            }
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)


def process_shard_range(
    fasta_dir: Path,
    output_dir: Path = Path("../../data/embeddings"),
    esm_model_name: str = "esm2_t6_8M_UR50D",
    layers: List[int] = [1, 2, 3, 4, 5, 6],
    start_shard: int | None = None,
    end_shard: int | None = None,
    corrupt_esm: bool = False,
    weight_file: Path | None = None,
    use_peft: bool = False, # 【変更点】PEFTモデルを使用するかどうかを明示的に指定
    disable_chunking: bool = False, # 【新規追加】chunk処理を無効にするフラグ
):
    """
    Process a range of FASTA shards through an ESM model.

    Processes each shard in the specified range, extracting and saving activations
    from specified layers. Can optionally use a corrupted model with shuffled
    parameters. Skips shards that have already been processed.

    Args:
        fasta_dir: Directory containing FASTA shard files
        output_dir: Directory to save outputs
        esm_model_name: Name of the ESM model to use
        layers: List of layer numbers to extract
        start_shard: First shard number to process
        end_shard: Last shard number to process (inclusive)
        corrupt_esm: Whether to shuffle model parameters
        weight_file: Path to custom weight file (.pt or .safetensors format)
        use_peft: Whether to treat weight_file as PEFT adapter weights
        disable_chunking: Whether to disable chunking and process all data at once

    Outputs:
        Creates directory structure with:
        - Activation tensors for each layer
        - Metadata files for each processed shard
    """

    # identify the number of shards in the fasta_dir
    fasta_files = list(fasta_dir.glob("*.fasta"))
    if not fasta_files:
        raise ValueError(f"No FASTA files found in {fasta_dir}")

    if start_shard is None:
        start_shard = 0
    if end_shard is None:
        end_shard = len(fasta_files) - 1

    # 【変更点】一度に処理するレイヤー数を定義
    layers_per_chunk = 3
    
    # 【変更点】レイヤーをチャンクに分割してループ
    print(f"Total layers to process: {layers}")
    print(f"Processing in chunks of {layers_per_chunk} layers at a time.")
    
    for i in range(0, len(layers), layers_per_chunk):
        layer_chunk = layers[i:i + layers_per_chunk]
        print(f"\n==============================================")
        print(f"=== Starting processing for layers: {layer_chunk} ===")
        print(f"==============================================")

        for shard_idx in range(start_shard, end_shard + 1):
            embed_fasta_file_for_all_layers(
                esm_model_name=esm_model_name,
                corrupt_esm=corrupt_esm,
                fasta_file=fasta_dir / f"shard_{shard_idx}.fasta",
                output_dir=output_dir,
                layers=layer_chunk, # ← チャンク化されたレイヤーを渡す
                shard_num=shard_idx,
                weight_file=weight_file,
                sequences_per_chunk=50000, # ← データチャンクのサイズを指定
                use_peft=use_peft, # ← PEFTモデルを使用するかどうかを渡す
                disable_chunking=disable_chunking # ← chunk処理の無効フラグを渡す
            )


if __name__ == "__main__":
    from tap import tapify
    tapify(process_shard_range)
