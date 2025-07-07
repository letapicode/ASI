import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asi.telemetry import TelemetryLogger
from asi.telemetry_aggregator import TelemetryAggregator
from asi.fine_grained_profiler import FineGrainedProfiler


class TestFineGrainedProfiler(unittest.TestCase):
    def test_aggregated_metrics(self):
        agg = TelemetryAggregator()
        logger1 = TelemetryLogger()
        logger2 = TelemetryLogger()

        logger1.register_profiler({"cpu_time": 1.0, "gpu_mem": 5.0}, node_id="n1")
        logger2.register_profiler({"cpu_time": 2.0, "gpu_mem": 7.0}, node_id="n2")

        stats1 = {"node_id": "n1", **logger1.get_stats(), "net": 100, "energy_kwh": 1, "carbon_g": 2}
        stats2 = {"node_id": "n2", **logger2.get_stats(), "net": 200, "energy_kwh": 1.5, "carbon_g": 3}

        agg.ingest(stats1)
        agg.ingest(stats2)
        totals = agg.aggregate()

        self.assertEqual(totals["prof_cpu_time"], 3.0)
        self.assertEqual(totals["prof_gpu_mem"], 12.0)
        self.assertEqual(totals["net"], 300)
        self.assertEqual(totals["energy_kwh"], 2.5)
        self.assertEqual(totals["carbon_g"], 5)


if __name__ == "__main__":
    unittest.main()
