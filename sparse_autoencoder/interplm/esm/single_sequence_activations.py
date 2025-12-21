"""
Extract ESM layer activations from a single amino acid sequence without shuffling.
Preserves the original sequence order and saves activations for each layer.
"""
import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple

# from interplm.esm.embed import get_model_converter_alphabet  # Not needed anymore


def get_single_sequence_activations(
    model: torch.nn.Module,
    sequence_tokens: torch.Tensor,
    sequence_mask: torch.Tensor,
    layers: List[int]
) -> Dict[int, torch.Tensor]:
    """
    Extract activations for a single sequence from specified layers using esm.pretrained API.
    
    Args:
        model: ESM model instance (from esm.pretrained)
        sequence_tokens: Tokenized sequence [1, seq_len]
        sequence_mask: Not used for esm.pretrained, but kept for compatibility
        layers: List of layer numbers to extract
        
    Returns:
        Dictionary mapping layer numbers to their activation tensors [seq_len_no_special, d_model]
    """
    with torch.no_grad():
        # Use esm.pretrained API (like SFT_hot.ipynb)
        results = model(sequence_tokens, repr_layers=layers)
        
        layer_activations = {}
        # Remove special tokens (CLS, PAD, EOS) - keep only actual amino acid tokens
        for layer in layers:
            # Get representations for this layer
            representations = results["representations"][layer]  # [batch, seq_len, d_model]
            
            # Remove batch dimension and special tokens
            # Assuming tokens: [CLS, AA1, AA2, ..., AAn, EOS, PAD, ...]
            # Keep only amino acid tokens (positions 1 to -1, excluding special tokens)
            seq_len = (sequence_tokens[0] != model.alphabet.padding_idx).sum()  # Actual sequence length
            activations = representations[0, 1:seq_len-1]  # Remove CLS and EOS
            
            layer_activations[layer] = activations
            
    return layer_activations


def process_single_sequence(
    sequence: str,
    output_dir: Path,
    sequence_id: str,
    esm_model_name: str = "esm2_t6_8M_UR50D",
    layers: List[int] = [1, 2, 3, 4, 5, 6],
    corrupt_esm: bool = False,
    truncation_seq_length: int = 1022,
    weight_file: Path | None = None,
):
    """
    Process a single amino acid sequence and save layer activations with preserved order.
    
    Args:
        sequence: Amino acid sequence string (e.g., "MKTALVFLGIT...")
        output_dir: Directory to save outputs
        sequence_id: Identifier for this sequence (used in filenames)
        esm_model_name: Name of the ESM model to use
        layers: List of layer numbers to extract
        corrupt_esm: Whether to use corrupted model
        truncation_seq_length: Maximum sequence length
        weight_file: Optional path to custom model weights
        
    Outputs:
        Creates files:
        - {output_dir}/{sequence_id}_layer_{layer}_activations.pt
        - {output_dir}/{sequence_id}_metadata.json
    """
    
    # Load model using esm.pretrained (to match SFT_hot.ipynb)
    import esm
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load custom weights if provided
    if weight_file is not None:
        file_extension = weight_file.suffix.lower()
        if file_extension == '.pt':
            ckpt: Dict[str, Any] = torch.load(weight_file, map_location=device)
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        elif file_extension == '.safetensors':
            try:
                from safetensors import safe_open
                state_dict = {}
                with safe_open(weight_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
                for key in state_dict:
                    state_dict[key] = state_dict[key].to(device)
            except ImportError:
                raise ImportError(
                    "safetensors is not installed. Please install 'safetensors' to use .safetensors files. "
                    "Run: pip install safetensors"
                )
        else:
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats are .pt and .safetensors"
            )
        # Extract ESM part if keys have 'esm.' prefix (from ESM2_TmRegressor)
        if any(k.startswith('esm.') for k in state_dict.keys()):
            print("Extracting ESM part from full model state_dict...")
            esm_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('esm.'):
                    new_key = k[4:]
                    esm_state_dict[new_key] = v
            state_dict = esm_state_dict
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {weight_file} (missing={len(missing)}, unexpected={len(unexpected)})")
        loaded_params = list(state_dict.keys())[:5]
        print(f"First few loaded ESM parameters: {loaded_params}")
    
    # Prepare sequence
    if len(sequence) > truncation_seq_length:
        print(f"Warning: Sequence truncated from {len(sequence)} to {truncation_seq_length}")
        sequence = sequence[:truncation_seq_length]
    
    # Tokenize sequence
    batch_labels, batch_strs, batch_tokens = batch_converter([(sequence_id, sequence)])
    batch_tokens = batch_tokens.to(device)
    batch_mask = (batch_tokens != alphabet.padding_idx).to(device)
    
    print(f"Processing sequence '{sequence_id}' with {len(sequence)} amino acids")
    
    # Get activations
    activations = get_single_sequence_activations(
        model, batch_tokens, batch_mask, layers
    )
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save activations for each layer
    for layer in layers:
        output_file = output_dir / f"{sequence_id}_layer_{layer}_activations.pt"
        torch.save(activations[layer].cpu(), output_file)
        print(f"Saved layer {layer} activations: {output_file}")
        print(f"  Shape: {activations[layer].shape}")
    
    # Save metadata
    metadata = {
        "sequence_id": sequence_id,
        "sequence": sequence,
        "sequence_length": len(sequence),
        "model": esm_model_name,
        "layers": layers,
        "d_model": model.embed_dim,
        "dtype": str(activations[layers[0]].dtype),
        "weight_file": str(weight_file) if weight_file else None,
        "preserved_order": True,
    }
    
    metadata_file = output_dir / f"{sequence_id}_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata: {metadata_file}")
    return activations


def process_all_sequences_from_fasta(
    fasta_file: Path,
    output_dir: Path,
    esm_model_name: str = "esm2_t6_8M_UR50D",
    layers: List[int] = [1, 2, 3, 4, 5, 6],
    corrupt_esm: bool = False,
    truncation_seq_length: int = 1022,
    weight_file: Path | None = None,
):
    """
    Process all sequences from a FASTA file and save combined activations with preserved order.
    
    Args:
        fasta_file: Path to FASTA file
        output_dir: Directory to save outputs
        esm_model_name: Name of the ESM model to use
        layers: List of layer numbers to extract
        corrupt_esm: Whether to use corrupted model
        truncation_seq_length: Maximum sequence length
        weight_file: Optional path to custom model weights
        
    Outputs:
        Combined files:
        - {output_dir}/all_sequences_layer_{layer}_activations.pt
        - {output_dir}/sequence_mapping.json
    """
    from Bio import SeqIO
    
    # Read FASTA file
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    print(f"Found {len(sequences)} sequences in {fasta_file}")
    
    # Load model once for all sequences using esm.pretrained (to match SFT_hot.ipynb)
    import esm
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load custom weights if provided
    if weight_file is not None:
        file_extension = weight_file.suffix.lower()
        if file_extension == '.pt':
            ckpt: Dict[str, Any] = torch.load(weight_file, map_location=device)
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        elif file_extension == '.safetensors':
            try:
                from safetensors import safe_open
                state_dict = {}
                with safe_open(weight_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
                for key in state_dict:
                    state_dict[key] = state_dict[key].to(device)
            except ImportError:
                raise ImportError(
                    "safetensors is not installed. Please install 'safetensors' to use .safetensors files. "
                    "Run: pip install safetensors"
                )
        else:
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats are .pt and .safetensors"
            )
        # Extract ESM part if keys have 'esm.' prefix (from ESM2_TmRegressor)
        if any(k.startswith('esm.') for k in state_dict.keys()):
            print("Extracting ESM part from full model state_dict...")
            esm_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('esm.'):
                    new_key = k[4:]
                    esm_state_dict[new_key] = v
            state_dict = esm_state_dict
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {weight_file} (missing={len(missing)}, unexpected={len(unexpected)})")
        loaded_params = list(state_dict.keys())[:5]
        print(f"First few loaded ESM parameters: {loaded_params}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # For storing per-protein activations
    combined_activations = {layer: [] for layer in layers}
    sequence_mapping = []
    
    # Process each sequence
    for seq_idx, seq_record in enumerate(sequences):
        sequence_id = seq_record.id
        sequence = str(seq_record.seq)
        
        # Truncate if necessary
        original_length = len(sequence)
        if len(sequence) > truncation_seq_length:
            print(f"Warning: Sequence {sequence_id} truncated from {original_length} to {truncation_seq_length}")
            sequence = sequence[:truncation_seq_length]
        
        # Tokenize sequence
        batch_labels, batch_strs, batch_tokens = batch_converter([(sequence_id, sequence)])
        batch_tokens = batch_tokens.to(device)
        batch_mask = (batch_tokens != alphabet.padding_idx).to(device)
        
        print(f"Processing sequence {seq_idx + 1}/{len(sequences)}: '{sequence_id}' ({len(sequence)} amino acids)")
        
        # Get activations
        activations = get_single_sequence_activations(
            model, batch_tokens, batch_mask, layers
        )
        
        # Add to combined data
        sequence_length = activations[layers[0]].shape[0]
        
        # Record mapping information
        sequence_mapping.append({
            "sequence_id": sequence_id,
            "sequence": sequence,
            "sequence_index": seq_idx,
            "sequence_length": sequence_length,
            "original_length": original_length,
            # Note: activations are stored as list[protein_idx][amino_acid_idx, feature_dim]
        })
        
        # Add activations to combined data
        for layer in layers:
            combined_activations[layer].append(activations[layer].cpu())
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    # Save combined data as list of per-protein activations
    print(f"\nSaving per-protein activations...")
    
    # Save activations as list of tensors (one per protein)
    for layer in layers:
        # combined_activations[layer] is already a list of tensors
        protein_activations_list = combined_activations[layer]
        output_file = output_dir / f"all_sequences_layer_{layer}_activations.pt"
        torch.save(protein_activations_list, output_file)
        print(f"Saved layer {layer}: {output_file}")
        print(f"  Number of proteins: {len(protein_activations_list)}")
        print(f"  Example shapes: {[act.shape for act in protein_activations_list[:3]]}...")  # Show first 3 shapes
    
    # Save sequence mapping
    total_tokens = sum(seq["sequence_length"] for seq in sequence_mapping)
    mapping_metadata = {
        "fasta_file": str(fasta_file),
        "total_sequences": len(sequences),
        "total_tokens": total_tokens,
        "model": esm_model_name,
        "layers": layers,
        "d_model": model.embed_dim,
        "weight_file": str(weight_file) if weight_file else None,
        "preserved_order": True,
        "data_structure": "list_of_protein_tensors",  # Each element is [seq_len, d_model]
        "sequence_mapping": sequence_mapping,
    }
    
    mapping_file = output_dir / "sequence_mapping.json"
    with open(mapping_file, "w") as f:
        json.dump(mapping_metadata, f, indent=2)
    
    print(f"Saved sequence mapping: {mapping_file}")
    print(f"Total tokens across all sequences: {total_tokens:,}")
    print(f"Data structure: List of {len(sequences)} protein tensors")
    print(f"Access pattern: activations_list[protein_idx][amino_acid_idx, feature_dim]")
    print(f"\nProcessing complete! Output saved to {output_dir}")


def process_sequence_from_fasta(
    fasta_file: Path,
    sequence_index: int,
    output_dir: Path,
    esm_model_name: str = "esm2_t6_8M_UR50D",
    layers: List[int] = [1, 2, 3, 4, 5, 6],
    corrupt_esm: bool = False,
    truncation_seq_length: int = 1022,
    weight_file: Path | None = None,
):
    """
    Extract a specific sequence from a FASTA file and process it.
    
    Args:
        fasta_file: Path to FASTA file
        sequence_index: Index of sequence to extract (0-based)
        output_dir: Directory to save outputs
        esm_model_name: Name of the ESM model to use
        layers: List of layer numbers to extract
        corrupt_esm: Whether to use corrupted model
        truncation_seq_length: Maximum sequence length
        weight_file: Optional path to custom model weights
    """
    from Bio import SeqIO
    
    # Read FASTA file
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    
    if sequence_index >= len(sequences):
        raise ValueError(f"Sequence index {sequence_index} out of range. File has {len(sequences)} sequences.")
    
    selected_seq = sequences[sequence_index]
    sequence_id = selected_seq.id
    sequence = str(selected_seq.seq)
    
    print(f"Selected sequence {sequence_index}: {sequence_id}")
    print(f"Length: {len(sequence)} amino acids")
    
    return process_single_sequence(
        sequence=sequence,
        output_dir=output_dir,
        sequence_id=sequence_id,
        esm_model_name=esm_model_name,
        layers=layers,
        corrupt_esm=corrupt_esm,
        truncation_seq_length=truncation_seq_length,
        weight_file=weight_file,
    )


if __name__ == "__main__":
    from tap import tapify
    
    def main(
        sequence: str = "",
        fasta_file: str = "",
        sequence_index: int = 0,
        sequence_id: str = "sequence",
        output_dir: str = "./single_sequence_output",
        esm_model_name: str = "esm2_t6_8M_UR50D",
        layers: List[int] = [1, 2, 3, 4, 5, 6],
        corrupt_esm: bool = False,
        truncation_seq_length: int = 1022,
        weight_file: str = "",
        process_all: bool = False,
    ):
        """
        Process amino acid sequences.
        
        Usage examples:
        1. Direct sequence:
           python single_sequence_activations.py --sequence "MKTALVFLGIT" --sequence_id "my_protein"
           
        2. Single sequence from FASTA:
           python single_sequence_activations.py --fasta_file "protein.fasta" --sequence_index 0
           
        3. All sequences from FASTA:
           python single_sequence_activations.py --fasta_file "protein.fasta" --process_all
        """
        output_path = Path(output_dir)
        weight_path = Path(weight_file) if weight_file else None
        
        if sequence:
            # Process direct sequence
            process_single_sequence(
                sequence=sequence,
                output_dir=output_path,
                sequence_id=sequence_id,
                esm_model_name=esm_model_name,
                layers=layers,
                corrupt_esm=corrupt_esm,
                truncation_seq_length=truncation_seq_length,
                weight_file=weight_path,
            )
        elif fasta_file:
            if process_all:
                # Process all sequences from FASTA
                process_all_sequences_from_fasta(
                    fasta_file=Path(fasta_file),
                    output_dir=output_path,
                    esm_model_name=esm_model_name,
                    layers=layers,
                    corrupt_esm=corrupt_esm,
                    truncation_seq_length=truncation_seq_length,
                    weight_file=weight_path,
                )
            else:
                # Process single sequence from FASTA
                process_sequence_from_fasta(
                    fasta_file=Path(fasta_file),
                    sequence_index=sequence_index,
                    output_dir=output_path,
                    esm_model_name=esm_model_name,
                    layers=layers,
                    corrupt_esm=corrupt_esm,
                    truncation_seq_length=truncation_seq_length,
                    weight_file=weight_path,
                )
        else:
            print("Error: Either --sequence or --fasta_file must be provided")
    
    tapify(main) 