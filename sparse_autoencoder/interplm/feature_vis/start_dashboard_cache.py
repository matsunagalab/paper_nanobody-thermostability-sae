import pickle
from pathlib import Path

import pandas as pd

from interplm.constants import DASHBOARD_CACHE
from interplm.feature_vis.feature_activation_distribution import (
    get_random_sample_of_sae_feats,
)
from interplm.feature_vis.track_activation_per_protein import find_max_examples_per_feat
from interplm.sae.inference import load_sae


def add_layer_to_dashboard(
    sae_path: Path,
    esm_embeds_dir: Path,
    aa_metadata_dir: Path,
    n_shards: int, 
    esm_model_name: str,
    layer: int,
    concept_dir: Path | None = None,
):

    sae = load_sae(sae_path)

    per_protein_tracker = find_max_examples_per_feat(
        sae, esm_embeds_dir, aa_metadata_dir, n_shards
    )

    sae_feats = get_random_sample_of_sae_feats(sae=sae, esm_embds_dir=esm_embeds_dir)

    layer_cache = {}
    layer_cache["ESM_metadata"] = {"esm_model_name": esm_model_name, "layer": layer}
    layer_cache["Per_feature_max_examples"] = per_protein_tracker["max"]
    layer_cache["Per_feature_quantile_examples"] = per_protein_tracker["lower_quantile"]
    layer_cache["SAE"] = sae.to("cpu")
    layer_cache["SAE_features"] = sae_feats
    layer_cache["Per_feature_statistics"] = {
        "Per_prot_frequency_of_any_activation": per_protein_tracker[
            "pct_proteins_with_activation"
        ],
        "Per_prot_pct_activated_when_present": per_protein_tracker[
            "avg_pct_activated_when_present"
        ],
    }

    if concept_dir:
        concept_results = pd.read_csv(concept_dir / "heldout_all_top_pairings.csv")

    layer_cache["Sig_concepts_per_feature"] = concept_results

    #if DASHBOARD_CACHE.exists():
    if False:
        print(f"Adding layer {layer} to dashboard cache at {DASHBOARD_CACHE}")
        with open(DASHBOARD_CACHE, "rb") as f:
            cache = pickle.load(f)
    else:
        print(f"Creating new dashboard cache at {DASHBOARD_CACHE}")
        cache = {}

    cache[layer] = layer_cache

    DASHBOARD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving final results to dashboard cache at {DASHBOARD_CACHE}")
    with open(DASHBOARD_CACHE, "wb") as f:
        pickle.dump(cache, f)


if __name__ == "__main__":
    from tap import tapify

    tapify(add_layer_to_dashboard)
