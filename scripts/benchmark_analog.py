import argparse
import torch

from asi.telemetry import TelemetryLogger
from asi.analog_backend import AnalogAccelerator


def run_cpu(a: torch.Tensor, b: torch.Tensor, steps: int) -> float:
    logger = TelemetryLogger(interval=0.05, carbon_tracker=True)
    logger.start()
    for _ in range(steps):
        _ = a @ b
    logger.stop()
    return logger.get_stats().get("energy_kwh", 0.0)


def run_analog(a: torch.Tensor, b: torch.Tensor, steps: int) -> float:
    accel = AnalogAccelerator()
    logger = TelemetryLogger(interval=0.05, carbon_tracker=True)
    logger.start()
    with accel:
        for _ in range(steps):
            _ = torch.matmul(a, b)
    logger.stop()
    return logger.get_stats().get("energy_kwh", 0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analog vs CPU energy benchmark")
    parser.add_argument("--dim", type=int, default=256, help="Matrix dimension")
    parser.add_argument("--steps", type=int, default=10, help="Number of matmuls")
    args = parser.parse_args()

    a = torch.randn(args.dim, args.dim)
    b = torch.randn(args.dim, args.dim)

    cpu = run_cpu(a, b, args.steps)
    analog = run_analog(a, b, args.steps)

    print(f"CPU energy_kwh: {cpu:.6f}")
    print(f"Analog energy_kwh: {analog:.6f}")

