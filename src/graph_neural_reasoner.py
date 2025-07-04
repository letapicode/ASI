from __future__ import annotations
import random
from typing import Dict, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from .knowledge_graph_memory import KnowledgeGraphMemory


class GraphNeuralReasoner(nn.Module):
    """Predict missing edges from a knowledge graph using a tiny GCN."""

    def __init__(self, kg: KnowledgeGraphMemory, dim: int = 8) -> None:
        super().__init__()
        self.kg = kg
        entities = sorted({s for s, _, _ in kg.triples} | {o for _, _, o in kg.triples})
        self.index: Dict[str, int] = {e: i for i, e in enumerate(entities)}
        n = len(self.index)
        self.embed = nn.Embedding(n, dim)
        self.lin = nn.Linear(dim, dim)
        self.adj = torch.zeros(n, n)
        for s, _, o in kg.triples:
            self.adj[self.index[s], self.index[o]] = 1.0
        self.register_buffer("A", self.adj)

    def forward(self) -> torch.Tensor:
        h = self.embed.weight
        h = F.relu(self.A @ h)
        h = self.lin(h)
        return h

    def predict_link(self, src: str, dst: str) -> float:
        if src not in self.index or dst not in self.index:
            return 0.0
        with torch.no_grad():
            h = self.forward()
            a = h[self.index[src]]
            b = h[self.index[dst]]
            score = torch.sigmoid((a * b).sum())
            return float(score.item())

__all__ = ["GraphNeuralReasoner"]
