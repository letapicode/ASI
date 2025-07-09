from __future__ import annotations

"""GNN-based forecast scheduler for HPC clusters."""

from dataclasses import dataclass, field
from typing import Dict, List

from .hpc_base_scheduler import HPCBaseScheduler, ForecastStrategy

try:  # Optional heavy dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover - allow missing torch
    torch = None  # type: ignore
    nn = None  # type: ignore


if nn is not None:
    class SimpleGNN(nn.Module):
        """Very small GNN for forecasting carbon and price."""

        def __init__(self, input_dim: int, hidden_dim: int = 8) -> None:
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 2)

        def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            h = torch.relu(self.fc1(x))
            agg = torch.matmul(adj, h)
            agg = agg / (adj.sum(1, keepdim=True) + 1e-6)
            h = torch.relu(h + agg)
            out = self.fc2(h)
            return out
else:  # pragma: no cover - fallback when torch unavailable
    class SimpleGNN:  # type: ignore
        def __init__(self, *a, **kw) -> None:
            raise ImportError("torch is required for SimpleGNN")


class GNNStrategy(ForecastStrategy):
    """GNN-based forecasting across multiple clusters."""

    def __init__(self, hist_len: int = 4) -> None:
        self.hist_len = hist_len
        self._model: SimpleGNN | None = None

    def _ensure_model(self, dim: int) -> None:
        if torch is None or nn is None:
            raise ImportError("torch is required for GNNForecastScheduler")
        if self._model is None or self._model.fc1.in_features != dim:
            torch.manual_seed(0)
            self._model = SimpleGNN(dim)

    def forecast_scores(
        self,
        scheduler: HPCBaseScheduler,
        max_delay: float,
        clusters: Dict[str, HPCBaseScheduler] | None = None,
    ) -> List[float]:
        steps = max(int(max_delay // 3600) + 1, 1)
        if not clusters:
            clusters = {"self": scheduler}
        scheds = list(clusters.values())
        idx = scheds.index(scheduler)
        seq_len = min(
            self.hist_len,
            *[len(s.carbon_history) for s in scheds],
            *[len(s.cost_history) for s in scheds],
        )
        if seq_len == 0:
            carbon = scheduler.carbon_history[-1] if scheduler.carbon_history else 0.0
            cost = scheduler.cost_history[-1] if scheduler.cost_history else 0.0
            score = scheduler.carbon_weight * carbon + scheduler.cost_weight * cost
            return [score] * steps
        feats = []
        for s in scheds:
            ch = s.carbon_history[-seq_len:]
            co = s.cost_history[-seq_len:]
            ch = [0.0] * (seq_len - len(ch)) + list(ch)
            co = [0.0] * (seq_len - len(co)) + list(co)
            feats.append(ch + co)
        x = torch.tensor(feats, dtype=torch.float32)
        adj = torch.ones(len(scheds), len(scheds), dtype=torch.float32)
        adj -= torch.eye(len(scheds), dtype=torch.float32)
        self._ensure_model(seq_len * 2)
        with torch.no_grad():
            out = self._model(x, adj)
        carbon_pred = float(out[idx, 0].item())
        cost_pred = float(out[idx, 1].item())
        score = scheduler.carbon_weight * carbon_pred + scheduler.cost_weight * cost_pred
        return [score] * steps


@dataclass
class GNNForecastScheduler(HPCBaseScheduler):
    """Schedule jobs using a simple GNN forecast across clusters."""

    hist_len: int = 4

    def __post_init__(self) -> None:  # pragma: no cover - simple init
        self.strategy = GNNStrategy(self.hist_len)


__all__ = ["GNNForecastScheduler", "SimpleGNN", "GNNStrategy"]
