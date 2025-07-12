from __future__ import annotations

"""Forecasting strategies used by :mod:`hpc_base_scheduler`."""

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

try:  # pragma: no cover - optional heavy deps
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore
    nn = None  # type: ignore

from .hpc_schedulers import ForecastStrategy


# ---------------------------------------------------------------------------
def arima_forecast(series: List[float], steps: int = 1) -> List[float]:
    """Return an ARIMA forecast for ``series``."""
    if not series:
        return [0.0] * steps
    try:
        order = (1, 1, 0)
        if len(series) < 3:
            order = (0, 0, 0)
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(series, order=order)
        fit = model.fit()
        forecast = fit.forecast(steps=steps)
        return list(np.asarray(forecast, dtype=float))
    except Exception:
        return [float(series[-1])] * steps


# ---------------------------------------------------------------------------
class ArimaStrategy(ForecastStrategy):
    """ARIMA-based forecasting helper."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, int, int], List[float]] = {}

    def forecast_scores(
        self,
        scheduler: "HPCBaseScheduler",
        max_delay: float,
        clusters: Dict[str, "HPCBaseScheduler"] | None = None,
    ) -> List[float]:
        steps = max(int(max_delay // 3600) + 1, 1)
        key = (len(scheduler.carbon_history), len(scheduler.cost_history), steps)
        if key in self._cache:
            return list(self._cache[key])
        carbon_pred = arima_forecast(scheduler.carbon_history, steps=steps)
        cost_pred = arima_forecast(scheduler.cost_history, steps=steps)
        n = min(len(carbon_pred), len(cost_pred))
        scores = [
            scheduler.carbon_weight * carbon_pred[i]
            + scheduler.cost_weight * cost_pred[i]
            for i in range(n)
        ]
        self._cache.clear()
        self._cache[key] = list(scores)
        return scores


# ---------------------------------------------------------------------------
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

else:  # pragma: no cover - fallback

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
            raise ImportError("torch is required for GNNStrategy")
        if self._model is None or self._model.fc1.in_features != dim:
            torch.manual_seed(0)
            self._model = SimpleGNN(dim)

    def forecast_scores(
        self,
        scheduler: "HPCBaseScheduler",
        max_delay: float,
        clusters: Dict[str, "HPCBaseScheduler"] | None = None,
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
            out = self._model(x, adj)  # type: ignore[arg-type]
        carbon_pred = float(out[idx, 0].item())
        cost_pred = float(out[idx, 1].item())
        score = scheduler.carbon_weight * carbon_pred + scheduler.cost_weight * cost_pred
        return [score] * steps


# ---------------------------------------------------------------------------
if torch is not None:

    class _TrendRNN(nn.Module):  # pragma: no cover - simple model
        """Two-layer LSTM used for forecasting."""

        def __init__(self, input_size: int = 2, hidden_size: int = 16,
                     num_layers: int = 2, dropout: float = 0.1) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
            self.out = nn.Linear(hidden_size, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y, _ = self.lstm(x)
            return self.out(y[:, -1])

else:

    class _TrendRNN:  # pragma: no cover - torch absent
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required for _TrendRNN")


if nn is not None:

    class _TrendTransformer(nn.Module):
        """Tiny transformer forecasting carbon and price."""

        def __init__(self, hidden_size: int = 16, nhead: int = 2, num_layers: int = 2) -> None:
            super().__init__()
            self.in_proj = nn.Linear(2, hidden_size)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=nhead, batch_first=True
            )
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.pos_emb = nn.Parameter(torch.zeros(1, 32, hidden_size))
            self.out = nn.Linear(hidden_size, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            seq_len = x.size(1)
            pos = self.pos_emb[:, :seq_len]
            h = self.in_proj(x) + pos
            h = self.enc(h)
            return self.out(h[:, -1])

else:  # pragma: no cover - fallback when torch unavailable

    class _TrendTransformer:  # type: ignore
        def __init__(self, *a, **kw) -> None:
            raise ImportError("torch is required for _TrendTransformer")


__all__ = [
    "arima_forecast",
    "ArimaStrategy",
    "GNNStrategy",
    "_TrendRNN",
    "_TrendTransformer",
    "SimpleGNN",
]
