import unittest
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asi.telemetry import TelemetryLogger, FineGrainedProfiler


class TestTelemetry(unittest.TestCase):
    def test_logger(self):
        logger = TelemetryLogger(interval=0.1)
        logger.start()
        time.sleep(0.2)
        logger.stop()
        stats = logger.get_stats()
        self.assertIn("cpu", stats)
        self.assertGreater(logger.get_carbon_intensity("EU"), 0.0)

    def test_profiler(self):
        vals = []
        def cb(cpu, gpu):
            vals.append(cpu)
        with FineGrainedProfiler(cb):
            sum(i for i in range(10000))
        self.assertTrue(vals and vals[0] > 0)


if __name__ == "__main__":
    unittest.main()
