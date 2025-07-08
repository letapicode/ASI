from __future__ import annotations

"""Transformer-based forecast scheduler for HPC clusters."""

from dataclasses import dataclass, field
from typing import List

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore


if nn is not None:
    class _TinyEncoder(nn.Module):
        def __init__(self, dim: int = 2) -> None:
            super().__init__()
            layer = nn.TransformerEncoderLayer(d_model=dim, nhead=1)
            self.enc = nn.TransformerEncoder(layer, num_layers=1)
            self.out = nn.Linear(dim, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.enc(x)
            return self.out(h[-1])
else:  # pragma: no cover - fallback
    class _TinyEncoder:
        def __init__(self, *a, **kw) -> None:
            raise ImportError("torch is required for TransformerForecastScheduler")


@dataclass
class TransformerForecastScheduler:
    """Predict future carbon and price via a tiny Transformer."""

    carbon_history: List[float] = field(default_factory=list)
    cost_history: List[float] = field(default_factory=list)
    carbon_weight: float = 0.5
    cost_weight: float = 0.5
    backend: str = "slurm"
    hist_len: int = 4
    _model: _TinyEncoder | None = field(default=None, init=False, repr=False)

    # --------------------------------------------------
    def _ensure_model(self) -> None:
        if torch is None or nn is None:
            raise ImportError("torch is required for TransformerForecastScheduler")
        if self._model is None:
            self._model = _TinyEncoder(2)

    # --------------------------------------------------
    def forecast_scores(self, max_delay: float, clusters=None) -> List[float]:
        steps = max(int(max_delay // 3600) + 1, 1)
        n = min(len(self.carbon_history), len(self.cost_history))
        if torch is None or n < self.hist_len:
            carbon = self.carbon_history[-1] if self.carbon_history else 0.0
            cost = self.cost_history[-1] if self.cost_history else 0.0
            score = self.carbon_weight * carbon + self.cost_weight * cost
            return [score] * steps
        self._ensure_model()
        seq = [
            [self.carbon_history[i], self.cost_history[i]]
            for i in range(n - self.hist_len, n)
        ]
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            out = self._model(x)
        carbon_pred = float(out[0].item())
        cost_pred = float(out[1].item())
        score = self.carbon_weight * carbon_pred + self.cost_weight * cost_pred
        return [score] * steps

    # --------------------------------------------------
    def submit_at_optimal_time(
        self, command: List[str] | str, max_delay: float = 21600.0
    ) -> str:
        scores = self.forecast_scores(max_delay)
        delay = 0.0
        if scores:
            idx = int(min(range(len(scores)), key=lambda i: scores[i]))
            delay = idx * 3600.0
        if delay and delay <= max_delay:
            import time
            time.sleep(delay)
        from .hpc_scheduler import submit_job
        return submit_job(command, backend=self.backend)


__all__ = ["TransformerForecastScheduler"]
