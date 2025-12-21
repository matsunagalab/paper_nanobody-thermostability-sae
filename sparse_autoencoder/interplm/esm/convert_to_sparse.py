"""
Convert ESM layer activations to sparse representations using SAE (Sparse Autoencoder).
Loads all layer activations and replaces specified layers with their sparse representations.
"""
import json
import sys
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Add the old interplm sae module to the path to import existing SAE classes
sys.path.append("/data3/yasu/interplm/old")
from interplm.sae.dictionary import AutoEncoder
from interplm.sae.inference import load_sae


def load_activation_files(
    activation_dir: Path,
    layers: List[int]
) -> tuple[Dict[int, List[torch.Tensor]], Dict[str, Any]]:
    """
    Load all activation files and sequence mapping (per-protein list format).
    
    Args:
        activation_dir: Directory containing activation files
        layers: List of layer numbers to load
        
    Returns:
        Tuple of (activations_dict, sequence_mapping)
        activations_dict[layer] = List[Tensor[seq_len_i, d_model]] for each protein
    """
    activations = {}
    
    # Load activation files (each is a list of per-protein tensors)
    for layer in layers:
        activation_file = activation_dir / f"all_sequences_layer_{layer}_activations.pt"
        if activation_file.exists():
            protein_list = torch.load(activation_file, map_location="cpu")
            activations[layer] = protein_list
            print(f"Loaded layer {layer}: {len(protein_list)} proteins")
            print(f"  Example shapes: {[tensor.shape for tensor in protein_list[:3]]}...")
        else:
            raise FileNotFoundError(f"Activation file not found: {activation_file}")
    
    # Load sequence mapping
    mapping_file = activation_dir / "sequence_mapping.json"
    if mapping_file.exists():
        with open(mapping_file, "r") as f:
            sequence_mapping = json.load(f)
        print(f"Loaded sequence mapping for {sequence_mapping['total_sequences']} sequences")
    else:
        raise FileNotFoundError(f"Sequence mapping file not found: {mapping_file}")
    
    return activations, sequence_mapping


def convert_layer_to_sparse(
    activations: Dict[int, List[torch.Tensor]],
    sequence_mapping: Dict[str, Any],
    target_layer: int,
    sae_model: AutoEncoder,
    output_dir: Path,
    batch_size: int = 1024,
    device: str = "cpu"
) -> None:
    """
    Convert specified layer activations to sparse representations and save (per-protein format).
    
    Args:
        activations: Dictionary of layer activations (each is list of per-protein tensors)
        sequence_mapping: Sequence mapping information
        target_layer: Layer to convert to sparse representation
        sae_model: Loaded SAE model
        output_dir: Directory to save outputs
        batch_size: Batch size for processing (per protein tokens)
        device: Device for computation
    """
    if target_layer not in activations:
        raise ValueError(f"Target layer {target_layer} not found in activations")
    
    print(f"Converting layer {target_layer} to sparse representation...")
    
    # Get target layer activations (list of per-protein tensors)
    target_protein_list = activations[target_layer]
    num_proteins = len(target_protein_list)
    
    # Process each protein individually
    sparse_protein_list = []
    sae_model = sae_model.to(device)
    
    total_tokens_processed = 0
    
    with torch.no_grad():
        for protein_idx, protein_activations in enumerate(target_protein_list):
            protein_activations = protein_activations.to(device)
            seq_len, d_model = protein_activations.shape
            
            # Process protein tokens in batches if sequence is long
            sparse_tokens_list = []
            
            for i in range(0, seq_len, batch_size):
                end_idx = min(i + batch_size, seq_len)
                batch_tokens = protein_activations[i:end_idx]
                
                # Convert to sparse representation
                sparse_batch = sae_model.encode(batch_tokens)
                sparse_tokens_list.append(sparse_batch.cpu())
            
            # Concatenate batches for this protein
            protein_sparse = torch.cat(sparse_tokens_list, dim=0)
            sparse_protein_list.append(protein_sparse)
            
            total_tokens_processed += seq_len
            
            if (protein_idx + 1) % 50 == 0:
                print(f"Processed {protein_idx + 1:,}/{num_proteins:,} proteins")
    
    print(f"Converted {num_proteins} proteins, {total_tokens_processed:,} total tokens")
    print(f"Example conversion: {target_protein_list[0].shape} -> {sparse_protein_list[0].shape}")
    
    # Calculate sparsity
    all_sparse_tokens = torch.cat(sparse_protein_list, dim=0)
    sparsity = (all_sparse_tokens == 0).float().mean()
    print(f"Overall sparsity: {sparsity:.3f}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sparse activations for target layer (as list of protein tensors)
    sparse_file = output_dir / f"all_sequences_layer_{target_layer}_sparse_activations.pt"
    torch.save(sparse_protein_list, sparse_file)
    print(f"Saved sparse activations: {sparse_file}")
    
    # Copy other layer activations unchanged
    for layer, layer_protein_list in activations.items():
        if layer != target_layer:
            output_file = output_dir / f"all_sequences_layer_{layer}_activations.pt"
            torch.save(layer_protein_list, output_file)
            print(f"Copied layer {layer} activations: {output_file}")
    
    # Update sequence mapping with sparse information
    updated_mapping = sequence_mapping.copy()
    updated_mapping["sparse_conversion"] = {
        "target_layer": target_layer,
        "original_d_model": target_protein_list[0].shape[1],
        "sparse_dict_size": sparse_protein_list[0].shape[1],
        "sparsity": float(sparsity),
        "conversion_timestamp": str(torch.tensor(0).numpy()),  # Simple timestamp
    }
    
    # Save updated sequence mapping
    mapping_file = output_dir / "sequence_mapping.json"
    with open(mapping_file, "w") as f:
        json.dump(updated_mapping, f, indent=2)
    print(f"Updated sequence mapping: {mapping_file}")


def convert_multiple_layers_to_sparse(
    activations: Dict[int, List[torch.Tensor]],
    sequence_mapping: Dict[str, Any],
    sparse_layers: Dict[int, AutoEncoder],  # layer -> AutoEncoder model mapping
    output_dir: Path,
    batch_size: int = 1024,
    device: str = "cpu"
) -> None:
    """
    Convert multiple layers to sparse representations (per-protein format).
    
    Args:
        activations: Dictionary of layer activations (each is list of per-protein tensors)
        sequence_mapping: Sequence mapping information
        sparse_layers: Dictionary mapping layer numbers to SAE models
        output_dir: Directory to save outputs
        batch_size: Batch size for processing (per protein tokens)
        device: Device for computation
    """
    print(f"Converting {len(sparse_layers)} layers to sparse representation...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    sparse_info = {}
    
    # Convert each specified layer
    for layer, sae_model in sparse_layers.items():
        if layer not in activations:
            print(f"Warning: Layer {layer} not found in activations, skipping...")
            continue
        
        print(f"\nProcessing layer {layer}...")
        target_protein_list = activations[layer]
        num_proteins = len(target_protein_list)
        
        # Process each protein individually
        sparse_protein_list = []
        sae_model = sae_model.to(device)
        total_tokens_processed = 0
        
        with torch.no_grad():
            for protein_idx, protein_activations in enumerate(target_protein_list):
                protein_activations = protein_activations.to(device)
                seq_len, d_model = protein_activations.shape
                
                # Process protein tokens in batches if sequence is long
                sparse_tokens_list = []
                
                for i in range(0, seq_len, batch_size):
                    end_idx = min(i + batch_size, seq_len)
                    batch_tokens = protein_activations[i:end_idx]
                    
                    sparse_batch = sae_model.encode(batch_tokens)
                    sparse_tokens_list.append(sparse_batch.cpu())
                
                # Concatenate batches for this protein
                protein_sparse = torch.cat(sparse_tokens_list, dim=0)
                sparse_protein_list.append(protein_sparse)
                
                total_tokens_processed += seq_len
                
                if (protein_idx + 1) % 50 == 0:
                    print(f"  Processed {protein_idx + 1:,}/{num_proteins:,} proteins")
        
        # Save sparse activations as list of protein tensors
        sparse_file = output_dir / f"all_sequences_layer_{layer}_sparse_activations.pt"
        torch.save(sparse_protein_list, sparse_file)
        
        # Calculate sparsity
        all_sparse_tokens = torch.cat(sparse_protein_list, dim=0)
        sparsity = (all_sparse_tokens == 0).float().mean()
        
        sparse_info[layer] = {
            "original_d_model": target_protein_list[0].shape[1],
            "sparse_dict_size": sparse_protein_list[0].shape[1],
            "sparsity": float(sparsity)
        }
        
        print(f"  Converted {num_proteins} proteins, {total_tokens_processed:,} tokens")
        print(f"  Example: {target_protein_list[0].shape} -> {sparse_protein_list[0].shape}")
        print(f"  Sparsity: {sparse_info[layer]['sparsity']:.3f}")
        print(f"  Saved: {sparse_file}")
    
    # Copy non-converted layers
    for layer, layer_protein_list in activations.items():
        if layer not in sparse_layers:
            output_file = output_dir / f"all_sequences_layer_{layer}_activations.pt"
            torch.save(layer_protein_list, output_file)
            print(f"Copied layer {layer}: {output_file}")
    
    # Update sequence mapping
    updated_mapping = sequence_mapping.copy()
    updated_mapping["sparse_conversion"] = sparse_info
    
    mapping_file = output_dir / "sequence_mapping.json"
    with open(mapping_file, "w") as f:
        json.dump(updated_mapping, f, indent=2)
    print(f"\nUpdated sequence mapping: {mapping_file}")


if __name__ == "__main__":
    from tap import tapify
    
    def main(
        activation_dir: str,
        output_dir: str,
        sae_weight_file: str,
        target_layers: List[int],
        all_layers: List[int] = [1, 2, 3, 4, 5, 6],
        batch_size: int = 1024,
        device: str = "auto",
    ):
        """
        Convert layer activations to sparse representations using SAE.
        
        Args:
            activation_dir: Directory containing activation files
            output_dir: Directory to save sparse representations
            sae_weight_file: Path to SAE weight file
            target_layers: List of layers to convert to sparse representation
            all_layers: List of all layer numbers to load (must include target_layers)
            batch_size: Batch size for processing
            device: Device for computation ("auto", "cpu", "cuda")
            
        Usage:
            # Convert single layer to sparse
            python convert_to_sparse.py \
                --activation_dir "./nbthermo" \
                --output_dir "./nbthermo_sparse" \
                --sae_weight_file "path/to/sae_weights.pt" \
                --target_layers 3 \
                --all_layers 1 2 3 4 5 6
                
            # Convert multiple layers to sparse (if you have SAE weights for each)
            python convert_to_sparse.py \
                --activation_dir "./nbthermo" \
                --output_dir "./nbthermo_sparse" \
                --sae_weight_file "path/to/sae_weights.pt" \
                --target_layers 2 3 4 \
                --all_layers 1 2 3 4 5 6
        """
        activation_path = Path(activation_dir)
        output_path = Path(output_dir)
        sae_weight_path = Path(sae_weight_file)
        
        # Auto-detect device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {device}")
        
        # Validate that target_layers are in all_layers
        for target_layer in target_layers:
            if target_layer not in all_layers:
                raise ValueError(f"Target layer {target_layer} must be in all_layers {all_layers}")
        
        # Load activations
        print("Loading activation files...")
        activations, sequence_mapping = load_activation_files(activation_path, all_layers)
        
        # Load SAE model using existing function
        print(f"Loading SAE model...")
        sae_model = load_sae(sae_weight_path, device)
        
        if len(target_layers) == 1:
            # Convert single layer to sparse representation
            target_layer = target_layers[0]
            print(f"Converting single layer {target_layer} to sparse representation...")
            convert_layer_to_sparse(
                activations=activations,
                sequence_mapping=sequence_mapping,
                target_layer=target_layer,
                sae_model=sae_model,
                output_dir=output_path,
                batch_size=batch_size,
                device=device
            )
        else:
            # Convert multiple layers to sparse representation
            print(f"Converting multiple layers {target_layers} to sparse representation...")
            # Note: This assumes the same SAE model works for all target layers
            # In practice, you might need different SAE models for different layers
            sparse_layers = {layer: sae_model for layer in target_layers}
            convert_multiple_layers_to_sparse(
                activations=activations,
                sequence_mapping=sequence_mapping,
                sparse_layers=sparse_layers,
                output_dir=output_path,
                batch_size=batch_size,
                device=device
            )
        
        print(f"\nConversion complete! Output saved to {output_path}")
    
    tapify(main) 