import numpy as np
from typing import Sequence, Tuple


def calibrate_sensors(
    sim_readings: Sequence[np.ndarray],
    real_readings: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (scale, offset) mapping simulation to real sensors."""
    sim = np.vstack(sim_readings)
    real = np.vstack(real_readings)
    A = np.hstack([sim, np.ones((sim.shape[0], 1))])
    params, _, _, _ = np.linalg.lstsq(A, real, rcond=None)
    scale = params[:-1]
    offset = params[-1]
    return scale, offset


def adjust_actions(
    sim_actions: Sequence[np.ndarray],
    real_actions: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (scale, offset) mapping sim actions to real actuators."""
    sim = np.vstack(sim_actions)
    real = np.vstack(real_actions)
    A = np.hstack([sim, np.ones((sim.shape[0], 1))])
    params, _, _, _ = np.linalg.lstsq(A, real, rcond=None)
    scale = params[:-1]
    offset = params[-1]
    return scale, offset
