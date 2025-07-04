import unittest
import time
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.hierarchical_memory import HierarchicalMemory
from src.memory_service import serve
from src.telemetry import TelemetryLogger

try:
    import grpc  # noqa: F401
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional
    _HAS_GRPC = False


class TestMemoryServerTelemetry(unittest.TestCase):
    def test_server_telemetry(self):
        if not _HAS_GRPC:
            self.skipTest("grpcio not available")
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        logger = TelemetryLogger(interval=0.1)
        server = serve(mem, "localhost:50210", telemetry=logger)
        time.sleep(0.2)
        stats = logger.get_stats()
        server.stop(0)
        self.assertIn("cpu", stats)


if __name__ == "__main__":
    unittest.main()
