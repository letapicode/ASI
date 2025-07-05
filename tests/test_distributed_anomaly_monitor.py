import unittest
import importlib.machinery
import importlib.util
import types
import sys

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg

stub_hm = types.ModuleType('src.hierarchical_memory')
class _Mem:
    def get_stats(self):
        return {'hit_rate': 0.0}
stub_hm.MemoryServer = type('MS', (), {'__init__': lambda self,*a,**k: None, 'memory': _Mem(), 'telemetry': None})
stub_hm.HierarchicalMemory = type('HM', (), {})
sys.modules['src.hierarchical_memory'] = stub_hm

mods = ['risk_dashboard', 'risk_scoreboard', 'distributed_anomaly_monitor', 'training_anomaly_detector', 'memory_dashboard', 'telemetry', 'memory_service']
for name in mods:
    loader = importlib.machinery.SourceFileLoader(f'src.{name}', f'src/{name}.py')
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'src'
    sys.modules[f'src.{name}'] = mod
    loader.exec_module(mod)

DistributedAnomalyMonitor = sys.modules['src.distributed_anomaly_monitor'].DistributedAnomalyMonitor
RiskDashboard = sys.modules['src.risk_dashboard'].RiskDashboard
RiskScoreboard = sys.modules['src.risk_scoreboard'].RiskScoreboard
TrainingAnomalyDetector = sys.modules['src.training_anomaly_detector'].TrainingAnomalyDetector
TelemetryLogger = sys.modules['src.telemetry'].TelemetryLogger


class TestDistributedAnomalyMonitor(unittest.TestCase):
    def test_aggregate_cross_run(self):
        board = RiskScoreboard()
        server = type('Stub', (), {'memory': _Mem(), 'telemetry': TelemetryLogger(interval=0.1)})()
        dash = RiskDashboard(board, [server])
        detectors = {
            'n1': TrainingAnomalyDetector(window=2, threshold=1.1),
            'n2': TrainingAnomalyDetector(window=2, threshold=1.1),
        }
        mon = DistributedAnomalyMonitor(dash, detectors)
        mon.record('n1', 0, 1.0)
        mon.record('n1', 1, 2.0)
        mon.record('n2', 0, 1.0)
        mon.record('n2', 1, 2.0)
        agg = mon.aggregate()
        self.assertEqual(agg['per_node']['n1'], 1.0)
        self.assertEqual(agg['per_node']['n2'], 1.0)
        self.assertEqual(len(agg['cross_run_events']), 1)
        self.assertEqual(board.metrics.get('cross_run_anomalies'), 1.0)


if __name__ == '__main__':
    unittest.main()
