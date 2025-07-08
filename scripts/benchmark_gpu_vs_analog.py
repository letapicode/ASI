import argparse
import time
import contextlib
from typing import Optional
import torch

from asi.telemetry import TelemetryLogger
from asi.analog_backend import AnalogAccelerator


class TinyModel(torch.nn.Module):
    """Simple two-layer perceptron used for inference benchmarking."""

    def __init__(self, dim: int = 128, hidden: int = 256) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _benchmark(model: torch.nn.Module, x: torch.Tensor, steps: int, ctx) -> tuple[float, float]:
    logger = TelemetryLogger(interval=0.05, carbon_tracker=True)
    logger.start()
    start = time.perf_counter()
    with ctx, torch.no_grad():
        for _ in range(steps):
            _ = model(x)
    if x.device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    logger.stop()
    stats = logger.get_stats()
    energy = float(stats.get("energy_kwh", 0.0))
    latency = (end - start) / steps
    return energy, latency


def run_gpu(
    model: torch.nn.Module,
    x: torch.Tensor,
    steps: int,
    amp: bool = False,
    use_compile: bool = False,
) -> tuple[float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available")
    device = torch.device("cuda")
    if use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    model = model.to(device)
    x = x.to(device)
    ctx = (
        torch.autocast(device_type="cuda")
        if amp and torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    return _benchmark(model, x, steps, ctx)


def run_analog(
    model: torch.nn.Module,
    x: torch.Tensor,
    steps: int,
    use_compile: bool = False,
) -> tuple[float, float]:
    if use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    return _benchmark(model, x, steps, AnalogAccelerator())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark GPU vs analog inference")
    parser.add_argument("--dim", type=int, default=128, help="Input dimension")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--amp", action="store_true", help="Enable autocast for GPU")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile if available")
    args = parser.parse_args()

    model = TinyModel(args.dim, args.hidden).eval()
    x = torch.randn(1, args.dim)

    if torch.cuda.is_available():
        gpu_energy, gpu_latency = run_gpu(model, x, args.steps, args.amp, args.compile)
        print(f"GPU energy_kwh: {gpu_energy:.6f}")
        print(f"GPU latency per step: {gpu_latency*1000:.3f} ms")
    else:
        print("GPU not available; skipping GPU benchmark")
        gpu_energy = None
        gpu_latency = None

    analog_energy, analog_latency = run_analog(model, x, args.steps, args.compile)
    print(f"Analog energy_kwh: {analog_energy:.6f}")
    print(f"Analog latency per step: {analog_latency*1000:.3f} ms")

    if gpu_energy is not None:
        ratio = analog_energy / gpu_energy if gpu_energy else float('inf')
        print(f"Energy ratio analog/gpu: {ratio:.2f}")
        print(f"Latency ratio analog/gpu: {analog_latency / gpu_latency:.2f}")

