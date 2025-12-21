from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from interplm.sae.dictionary import AutoEncoder
from interplm.sae.inference import get_sae_feats_in_batches
from interplm.utils import get_device


def get_random_sample_of_sae_feats(
    sae: AutoEncoder, esm_embds_dir: Path, n_shards: int = 0
):
    """
    Get a random sample of up to 1000 nonzero activations for each feature by scanning
    across n_shards shards of ESM embeddings.
    """
    device = get_device()

    nonzero_acts_per_feat = defaultdict(list)
    for shard in range(n_shards):
        esm_acts = torch.load(
            esm_embds_dir / f"shard_{shard}.pt", weights_only=True, map_location=device
        )

        # iterate through esm_acts and for each feat, add any nonzero acts to nonzero_per_feat
        for feat_chunk_list in np.array_split(range(sae.dict_size), 32):
            sae_feats = (
                get_sae_feats_in_batches(
                    sae=sae,
                    device=device,
                    esm_embds=esm_acts,
                    feat_list=feat_chunk_list,
                    chunk_size=10_000,
                )
                .cpu()
                .numpy()
            )

            for i, feature in enumerate(feat_chunk_list):
                # find the nonzero acts and add them to nonzero_acts_per_feat
                nonzero_for_feat = sae_feats[:, i][sae_feats[:, i] != 0]
                # if nonzero_per_feat > 1000, subsample to 1000
                if len(nonzero_for_feat) > 1_000:
                    nonzero_for_feat = np.random.choice(
                        nonzero_for_feat, 1_000, replace=False
                    )

                nonzero_acts_per_feat[feature] = nonzero_for_feat.tolist()

    return nonzero_acts_per_feat
