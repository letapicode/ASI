import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .dataset_versioner import DatasetVersioner


def _hash_params(model: nn.Module) -> str:
    vec = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy()
    return hashlib.sha256(vec.tobytes()).hexdigest()


class ModelVersionManager:
    """Record model hashes linked with dataset versions."""

    def __init__(self, root: str | Path, dataset_versioner: DatasetVersioner) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset_versioner = dataset_versioner
        self.log = self.root / "model_versions.json"

    def record(self, model: nn.Module, epoch: int) -> str:
        h = _hash_params(model)
        ds_file = self.dataset_versioner.root / "dataset_version.json"
        ds_hash = hashlib.sha256(ds_file.read_bytes()).hexdigest() if ds_file.exists() else ""
        entry = {"epoch": epoch, "model_hash": h, "dataset_hash": ds_hash}
        if self.log.exists():
            data = json.loads(self.log.read_text())
        else:
            data = []
        data.append(entry)
        self.log.write_text(json.dumps(data, indent=2))
        return h


__all__ = ["ModelVersionManager"]

