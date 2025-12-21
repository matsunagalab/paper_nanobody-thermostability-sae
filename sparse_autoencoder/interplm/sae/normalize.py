"""
Normalize SAE model features based on maximum activation values, adjusting the model weights
to maintain the same reconstructions while ensuring that the maximum activation value for each
feature is 1 across the provided dataset.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from interplm.sae.inference import (
    get_sae_feats_in_batches,
    load_sae,
    split_up_feature_list,
)
from interplm.utils import get_device


def calculate_feature_statistics(
    sae: torch.nn.Module,
    esm_embds_dir: Path,
    n_shards: int | None = None,
    max_features_per_chunk: int = 640,
    max_tokens_per_chunk: int = 25_000,
) -> torch.Tensor:
    """
    Calculate maximum activation value for each SAE feature across all data.

    Args:
        sae: Sparse autoencoder model
        esm_embds_dir: Directory containing ESM embeddings
        n_shards: Number of data shards to process
        max_features_per_chunk: Maximum features to process at once
        max_tokens_per_chunk: Maximum tokens to process in one batch

    Returns:
        Tensor containing maximum activation value for each feature
    """
    device = get_device()
    num_features = sae.dict_size
    max_per_feat = torch.zeros(num_features, device=device)

    if n_shards is None:
        n_shards = len(list(esm_embds_dir.glob("shard_*.pt")))

    for shard in range(n_shards):
        # Load embeddings for current shard
        shard_path = esm_embds_dir / f"shard_{shard}.pt"
        esm_acts = torch.load(shard_path, map_location=device, weights_only=True)

        # Process features in chunks to manage memory
        for feature_list in split_up_feature_list(
            total_features=num_features, max_feature_chunk_size=max_features_per_chunk
        ):
            # Get SAE features for current chunk
            sae_feats = get_sae_feats_in_batches(
                sae=sae,
                device=device,
                esm_embds=esm_acts,
                chunk_size=max_tokens_per_chunk,
                feat_list=feature_list,
            )

            # Update maximum values for current feature subset
            max_per_feat[feature_list] = torch.max(
                max_per_feat[feature_list], torch.max(sae_feats, dim=0)[0]
            )

            # Clean up to manage memory
            del sae_feats
            torch.cuda.empty_cache()

    return max_per_feat


def create_normalized_model(
    sae: torch.nn.Module, max_per_feat: torch.Tensor
) -> torch.nn.Module:
    """
    Create a normalized version of the SAE model based on maximum feature values.

    Args:
        sae: Original SAE model
        max_per_feat: Maximum activation values per feature

    Returns:
        Normalized copy of the SAE model
    """

    with torch.no_grad():
        # Normalize encoder weights and bias
        sae.encoder.weight.div_(max_per_feat.unsqueeze(1))

        # Replace any inf values introduced by division by 0
        sae.encoder.weight[sae.encoder.weight.isinf()] = 0

        if sae.encoder.bias is not None:
            sae.encoder.bias.div_(max_per_feat)

            # Replace any inf values introduced by division by 0
            sae.encoder.bias[sae.encoder.bias.isinf()] = 0

        # Adjust decoder weights to maintain reconstruction
        sae.decoder.weight.mul_(max_per_feat.unsqueeze(0))

    return sae


def normalize_sae_features(
    sae_dir: Path, esm_embds_dir: Path, n_shards: Optional[int] = 1
) -> None:
    """
    Calculate feature statistics and create a normalized version of the SAE model.

    Args:
        sae_dir: Directory containing SAE model
        esm_embds_dir: Directory containing ESM embeddings
        n_shards: Number of data shards to process
    """
    # Setup paths
    ckpt_path = sae_dir / "ae.pt"
    feat_stat_cache = sae_dir / "feature_stats"
    norm_sae_path = ckpt_path.parent / f"{ckpt_path.stem}_normalized.pt"

    # Create cache directory
    feat_stat_cache.mkdir(parents=True, exist_ok=True)

    # Load model and calculate statistics
    print("Loading SAE model and calculating feature statistics...")
    sae = load_sae(ckpt_path)
    max_per_feat = calculate_feature_statistics(
        sae=sae, esm_embds_dir=esm_embds_dir, n_shards=n_shards
    )

    # Save statistics
    np.save(feat_stat_cache / "max.npy", max_per_feat.cpu().numpy())

    print("Creating normalized SAE model...")
    sae_normalized = create_normalized_model(sae, max_per_feat)
    torch.save(sae_normalized.state_dict(), norm_sae_path)
    print(f"Normalized model saved to {norm_sae_path}")


if __name__ == "__main__":
    from tap import tapify

    tapify(normalize_sae_features)
