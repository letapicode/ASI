import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class BreakpointModel:
    """Piecewise log-log scaling law with a breakpoint."""
    breakpoint: float
    slope1: float
    intercept1: float
    slope2: float
    intercept2: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        logx = np.log10(x)
        pred1 = self.intercept1 + self.slope1 * logx
        pred2 = self.intercept2 + self.slope2 * logx
        return np.where(x <= self.breakpoint, pred1, pred2)

def fit_breakpoint(x: np.ndarray, y: np.ndarray) -> BreakpointModel:
    """Fit piecewise linear model in log space via grid search."""
    logx = np.log10(x)
    best_err = float('inf')
    best_model = None
    for bp in np.unique(x):
        left = logx <= np.log10(bp)
        if left.sum() < 2 or (~left).sum() < 2:
            continue
        A1 = np.vstack([logx[left], np.ones(left.sum())]).T
        coef1, _, _, _ = np.linalg.lstsq(A1, y[left], rcond=None)
        pred1 = A1 @ coef1
        A2 = np.vstack([logx[~left], np.ones((~left).sum())]).T
        coef2, _, _, _ = np.linalg.lstsq(A2, y[~left], rcond=None)
        pred2 = A2 @ coef2
        err = ((y[left] - pred1) ** 2).mean() + ((y[~left] - pred2) ** 2).mean()
        if err < best_err:
            best_err = err
            best_model = BreakpointModel(bp, coef1[0], coef1[1], coef2[0], coef2[1])
    if best_model is None:
        raise ValueError("Insufficient data to fit model")
    return best_model

if __name__ == "__main__":
    # Example usage with dummy data
    params = np.array([1e7, 5e7, 1e8, 5e8, 1e9, 5e9])
    loss = np.array([2.0, 1.8, 1.6, 1.3, 1.25, 1.2])
    model = fit_breakpoint(params, loss)
    print("breakpoint:", model.breakpoint)
    print("predictions:", model.predict(params))
