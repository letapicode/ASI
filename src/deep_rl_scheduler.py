from __future__ import annotations

"""Deep RL scheduler with an optional RNN predictor."""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Tuple, Union

from .hpc_multi_scheduler import MultiClusterScheduler
from .hpc_schedulers import submit_job
from .forecast_strategies import _TrendRNN

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch may not be installed
    torch = None  # type: ignore
    nn = None  # type: ignore




@dataclass
class DeepRLScheduler(MultiClusterScheduler):
    """Predict cost and carbon trends with an RNN to choose the best cluster."""

    hidden_size: int = 16
    num_layers: int = 2
    history_len: int = 4
    lr: float = 0.01
    epochs: int = 20
    model: object | None = field(default=None, init=False)

    # --------------------------------------------------
    def update_history(self, cluster: str, carbon: float, cost: float) -> None:
        """Append a new (carbon, cost) sample to ``cluster``."""
        sched = self.clusters.get(cluster)
        if sched is None:
            raise KeyError(cluster)
        sched.carbon_history.append(carbon)
        sched.cost_history.append(cost)

    # --------------------------------------------------
    def refit(self) -> None:
        """Retrain the model on the updated history."""
        if self.model is None:
            return
        self.fit()

    def __post_init__(self) -> None:
        if torch is None:
            self.model = None
            return
        self.model = _TrendRNN(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )
        self.fit()

    # --------------------------------------------------
    def fit(self) -> None:  # pragma: no cover - tiny dataset
        """Train the RNN on traces from all clusters."""
        if torch is None:
            return
        dataset: List[Tuple[List[List[float]], List[float]]] = []
        for sched in self.clusters.values():
            ch, ph = sched.carbon_history, sched.cost_history
            n = min(len(ch), len(ph))
            for i in range(self.history_len, n):
                x = [[ch[j], ph[j]] for j in range(i - self.history_len, i)]
                y = [ch[i], ph[i]]
                dataset.append((x, y))
        if not dataset:
            return
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        for _ in range(self.epochs):
            for x, y in dataset:
                x_t = torch.tensor([x], dtype=torch.float32)
                y_t = torch.tensor(y, dtype=torch.float32)
                optim.zero_grad()
                pred = self.model(x_t)
                loss = loss_fn(pred, y_t)
                loss.backward()
                optim.step()

    # --------------------------------------------------
    def _predict(self, sched, steps: int) -> Tuple[List[float], List[float]]:
        ch, ph = sched.carbon_history, sched.cost_history
        n = min(len(ch), len(ph))
        if torch is None or self.model is None or n < self.history_len:
            last_c = ch[-1] if ch else 0.0
            last_p = ph[-1] if ph else 0.0
            return [last_c] * steps, [last_p] * steps

        seq = [[ch[i], ph[i]] for i in range(n - self.history_len, n)]
        cur = torch.tensor([seq], dtype=torch.float32)
        preds_c, preds_p = [], []
        with torch.no_grad():
            for _ in range(steps):
                out = self.model(cur)
                c, p = out[0].tolist()
                preds_c.append(float(c))
                preds_p.append(float(p))
                nxt = torch.tensor([[c, p]], dtype=torch.float32).unsqueeze(0)
                cur = torch.cat([cur[:, 1:], nxt], dim=1)
        return preds_c, preds_p

    # --------------------------------------------------
    def schedule_job(
        self, command: Union[str, List[str]], max_delay: float = 21600.0
    ) -> Tuple[str, str]:
        """Return chosen cluster name and job id after submission."""
        best_cluster = None
        best_backend = None
        best_score = float("inf")
        best_delay = 0.0

        steps = max(int(max_delay // 3600) + 1, 1)
        for name, sched in self.clusters.items():
            c_pred, p_pred = self._predict(sched, steps)
            n = min(len(c_pred), len(p_pred))
            if not n:
                continue
            scores = [
                sched.carbon_weight * c_pred[i] + sched.cost_weight * p_pred[i]
                for i in range(n)
            ]
            idx = int(min(range(n), key=lambda i: scores[i]))
            if scores[idx] < best_score:
                best_score = scores[idx]
                best_delay = idx * 3600.0
                best_cluster = name
                best_backend = sched.backend

        if best_cluster is None:
            raise ValueError("No forecasts available to choose a cluster")
        if best_delay and best_delay <= max_delay:
            time.sleep(best_delay)
        job_id = submit_job(command, backend=best_backend)
        return best_cluster, job_id


__all__ = [
    "DeepRLScheduler",
]
