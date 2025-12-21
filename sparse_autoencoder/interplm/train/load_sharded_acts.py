"""
Load activations from a sharded dataset. Lazily loads datasets one at a time as needed.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class LazySingleTokenDataset(Dataset):
    def __init__(self, filename: str, total_tokens: int, d_model: int):
        self.filename = filename
        self.total_tokens = total_tokens
        self.d_model = d_model
        self.tensor = None
        self.accessed_indices = set()

    def __len__(self):
        return self.total_tokens

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.tensor is None:
            self.tensor = torch.load(
                self.filename, map_location="cpu", weights_only=True)

        self.accessed_indices.add(idx)

        if len(self.accessed_indices) == self.total_tokens:
            # All indices have been accessed, unload the tensor
            result = self.tensor[idx].clone()
            self.tensor = None
            self.accessed_indices.clear()
            return result

        return self.tensor[idx]


class LazyMultiDirectoryTokenDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.datasets = []
        self.total_tokens = 0
        self.d_model = None
        self.cumulative_tokens = [0]

        print("Loading dataset metadata")
        subdirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        for subdir in tqdm(subdirs):
            dataset_info = self._load_dataset_info(subdir)
            if dataset_info is not None:
                self.datasets.append(dataset_info)
                self.total_tokens += dataset_info["total_tokens"]
                self.cumulative_tokens.append(self.total_tokens)
                if self.d_model is None:
                    self.d_model = dataset_info["d_model"]
                else:
                    assert (
                        self.d_model == dataset_info["d_model"]
                    ), "Inconsistent d_model across datasets"

    def _load_dataset_info(self, subdir: Path):
        metadata_path = subdir / "metadata.json"
        activations_path = subdir / "activations.pt"
        if not (metadata_path.exists() and activations_path.exists()):
            return None

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        dataset = LazySingleTokenDataset(
            str(activations_path),
            total_tokens=metadata["total_tokens"],
            d_model=metadata["d_model"],
        )
        return {
            "plm_name": metadata["model"],
            "total_tokens": metadata["total_tokens"],
            "d_model": metadata["d_model"],
            "dataset": dataset,
            "dtype": metadata["dtype"],
            "layer": metadata["layer"] if "layer" in metadata else 6,
        }

    def __len__(self):
        return self.total_tokens

    def __getitem__(self, idx: int) -> torch.Tensor:
        dataset_index = (
            next(
                i
                for i, cum_tokens in enumerate(self.cumulative_tokens)
                if cum_tokens > idx
            )
            - 1
        )
        local_idx = idx - self.cumulative_tokens[dataset_index]
        return self.datasets[dataset_index]["dataset"][local_idx]
