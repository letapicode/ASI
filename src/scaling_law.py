import numpy as np
from itertools import combinations
from collections.abc import Iterable

class BreakpointScalingLaw:
    """Piecewise log-log scaling model supporting multiple breakpoints."""

    def __init__(self, break_compute: float | Iterable[float] | None = None):
        self.break_compute = break_compute
        # list of (slope, intercept) per segment
        self.params: list[tuple[float, float]] | None = None

    def _fit_segment(self, x, y):
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        return slope, intercept

    def fit(self, compute, loss):
        compute = np.asarray(compute, dtype=float)
        loss = np.asarray(loss, dtype=float)
        log_c = np.log10(compute)
        log_l = np.log10(loss)

        if isinstance(self.break_compute, Iterable) and not isinstance(
            self.break_compute, (float, int)
        ):
            num_breaks = len(list(self.break_compute))
        elif self.break_compute is None:
            num_breaks = 1
        else:
            num_breaks = 1

        best_err = float("inf")
        best_breaks = None
        best_params = None
        n = len(compute)
        indices = range(1, n)
        for breaks in combinations(indices, num_breaks):
            seg_params = []
            err = 0.0
            start = 0
            valid = True
            for b in list(breaks) + [n]:
                if b - start < 2:
                    valid = False
                    break
                slope, intercept = self._fit_segment(log_c[start:b], log_l[start:b])
                seg_params.append((slope, intercept))
                pred = slope * log_c[start:b] + intercept
                err += ((log_l[start:b] - pred) ** 2).mean()
                start = b
            if not valid:
                continue
            if err < best_err:
                best_err = err
                best_breaks = [compute[i] for i in breaks]
                best_params = seg_params

        if best_breaks is None:
            raise ValueError("Insufficient data to fit model")

        self.break_compute = best_breaks[0] if num_breaks == 1 else best_breaks
        self.params = best_params

    def predict(self, compute):
        if self.params is None:
            raise RuntimeError("Model is not fitted")
        compute = np.asarray(compute, dtype=float)
        log_c = np.log10(compute)

        breaks = self.break_compute
        if isinstance(breaks, Iterable) and not isinstance(breaks, (float, int)):
            bps = list(breaks)
        else:
            bps = [breaks]

        pred = np.empty_like(log_c)
        start_mask = compute < bps[0] if bps else np.full_like(compute, True, dtype=bool)
        slope, intercept = self.params[0]
        pred[start_mask] = slope * log_c[start_mask] + intercept
        prev_bp = bps[0] if bps else None
        for i in range(1, len(self.params)):
            slope, intercept = self.params[i]
            if i < len(bps):
                mask = (compute >= prev_bp) & (compute < bps[i])
                prev_bp = bps[i]
            else:
                mask = compute >= prev_bp
            pred[mask] = slope * log_c[mask] + intercept
        return 10 ** pred
