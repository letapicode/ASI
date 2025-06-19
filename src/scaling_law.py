import numpy as np

class BreakpointScalingLaw:
    """Simple log-log breakpoint model for loss vs. compute."""

    def __init__(self, break_compute=None):
        self.break_compute = break_compute
        self.params = None  # [(slope1, intercept1), (slope2, intercept2)]

    def _fit_segment(self, x, y):
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        return slope, intercept

    def fit(self, compute, loss):
        compute = np.asarray(compute, dtype=float)
        loss = np.asarray(loss, dtype=float)
        log_c = np.log10(compute)
        log_l = np.log10(loss)
        if self.break_compute is None:
            idx = len(compute) // 2
        else:
            idx = np.searchsorted(compute, self.break_compute)
            idx = max(1, min(idx, len(compute) - 1))
        self.break_compute = compute[idx]
        slope1, intercept1 = self._fit_segment(log_c[:idx], log_l[:idx])
        slope2, intercept2 = self._fit_segment(log_c[idx:], log_l[idx:])
        self.params = [(slope1, intercept1), (slope2, intercept2)]

    def predict(self, compute):
        if self.params is None:
            raise RuntimeError("Model is not fitted")
        compute = np.asarray(compute, dtype=float)
        log_c = np.log10(compute)
        slope1, intercept1 = self.params[0]
        slope2, intercept2 = self.params[1]
        break_mask = compute < self.break_compute
        pred = np.empty_like(log_c)
        pred[break_mask] = slope1 * log_c[break_mask] + intercept1
        pred[~break_mask] = slope2 * log_c[~break_mask] + intercept2
        return 10 ** pred
