from __future__ import annotations

"""Schedule HPC jobs with transformer-based forecasts."""

from dataclasses import dataclass, field
from typing import List

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - allow missing torch
    torch = None  # type: ignore
    nn = None  # type: ignore


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
            raise ImportError("torch is required for TransformerForecastScheduler")


@dataclass
class TransformerForecastScheduler:
    """Forecast carbon and price with a tiny transformer."""

    carbon_history: List[float] = field(default_factory=list)
    cost_history: List[float] = field(default_factory=list)
    carbon_weight: float = 0.5
    cost_weight: float = 0.5
    backend: str = "slurm"
    hidden_size: int = 16
    nhead: int = 2
    num_layers: int = 2
    history_len: int = 4
    lr: float = 0.01
    epochs: int = 20
    _model: _TrendTransformer | None = field(default=None, init=False, repr=False)
    _predict_cache: dict[int, tuple[list[float], list[float]]] = field(
        default_factory=dict, init=False, repr=False
    )

    # --------------------------------------------------
    def __post_init__(self) -> None:
        if torch is None or nn is None:
            self._model = None
            return
        torch.manual_seed(0)
        self._model = _TrendTransformer(
            hidden_size=self.hidden_size, nhead=self.nhead, num_layers=self.num_layers
        )
        self.fit()
        self._predict_cache.clear()

    # --------------------------------------------------
    def fit(self) -> None:  # pragma: no cover - tiny dataset
        """Train the transformer on stored history."""
        if torch is None or self._model is None:
            return
        dataset: List[tuple[list[list[float]], list[float]]] = []
        ch, ph = self.carbon_history, self.cost_history
        n = min(len(ch), len(ph))
        for i in range(self.history_len, n):
            x = [[ch[j], ph[j]] for j in range(i - self.history_len, i)]
            y = [ch[i], ph[i]]
            dataset.append((x, y))
        if not dataset:
            return
        optim = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        for _ in range(self.epochs):
            for x, y in dataset:
                x_t = torch.tensor([x], dtype=torch.float32)
                y_t = torch.tensor(y, dtype=torch.float32)
                optim.zero_grad()
                pred = self._model(x_t)
                loss = loss_fn(pred, y_t)
                loss.backward()
                optim.step()
        self._predict_cache.clear()

    # --------------------------------------------------
    def add_history(self, carbon: float, cost: float) -> None:
        """Append a new observation and clear cached predictions."""
        self.carbon_history.append(carbon)
        self.cost_history.append(cost)
        self._predict_cache.clear()

    # --------------------------------------------------
    def _predict(self, steps: int) -> tuple[list[float], list[float]]:
        """Predict ``steps`` future values using the trained model."""
        if steps in self._predict_cache:
            return self._predict_cache[steps]

        ch, ph = self.carbon_history, self.cost_history
        n = min(len(ch), len(ph))
        if torch is None or self._model is None or n < self.history_len:
            last_c = ch[-1] if ch else 0.0
            last_p = ph[-1] if ph else 0.0
            preds = ([last_c] * steps, [last_p] * steps)
            self._predict_cache[steps] = preds
            return preds

        seq = [[ch[i], ph[i]] for i in range(n - self.history_len, n)]
        cur = torch.tensor([seq], dtype=torch.float32)
        preds_c: list[float] = []
        preds_p: list[float] = []
        with torch.no_grad():
            for _ in range(steps):
                out = self._model(cur)
                c, p = out[0].tolist()
                preds_c.append(float(c))
                preds_p.append(float(p))
                nxt = torch.tensor([[c, p]], dtype=torch.float32).unsqueeze(0)
                cur = torch.cat([cur[:, 1:], nxt], dim=1)

        preds = (preds_c, preds_p)
        self._predict_cache[steps] = preds
        return preds

    # --------------------------------------------------
    def forecast_scores(self, max_delay: float, clusters=None) -> List[float]:
        """Return combined score forecast for each hour."""
        steps = max(int(max_delay // 3600) + 1, 1)
        carbon_pred, cost_pred = self._predict(steps)
        n = min(len(carbon_pred), len(cost_pred))
        scores = [
            self.carbon_weight * carbon_pred[i] + self.cost_weight * cost_pred[i]
            for i in range(n)
        ]
        return scores

    # --------------------------------------------------
    def predict_slot(self, max_delay: float = 21600.0) -> int:
        """Return hour index with lowest forecast score."""
        scores = self.forecast_scores(max_delay)
        if not scores:
            return 0
        return int(min(range(len(scores)), key=lambda i: scores[i]))

    # --------------------------------------------------
    def submit_at_optimal_time(self, command, max_delay: float = 21600.0):
        from .hpc_schedulers import submit_job

        idx = self.predict_slot(max_delay)
        delay = idx * 3600.0
        if delay and delay <= max_delay:
            import time

            time.sleep(delay)
        return submit_job(command, backend=self.backend)


__all__ = ["TransformerForecastScheduler", "_TrendTransformer"]
