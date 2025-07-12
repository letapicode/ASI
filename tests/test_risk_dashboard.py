import unittest
import importlib.machinery
import importlib.util
import types
import sys
try:
    import torch  # pragma: no cover - optional heavy dep
except Exception:
    torch = types.ModuleType('torch')
sys.modules['torch'] = torch
np = types.SimpleNamespace(mean=lambda x: sum(x)/len(x) if x else 0.0,
                           corrcoef=lambda a,b: [[0,0],[0,0]],
                           array=lambda x: x,
                           ndarray=list)
sys.modules.setdefault('numpy', np)
sys.modules.setdefault('matplotlib', types.ModuleType('matplotlib'))
sys.modules.setdefault('matplotlib.pyplot', types.SimpleNamespace())
re_stub = types.ModuleType('asi.retrieval_explainer')
re_stub.RetrievalExplainer = type('RE', (), {})
rv_stub = types.ModuleType('asi.retrieval_visualizer')
rv_stub.RetrievalVisualizer = type('RV', (), {})
rts_stub = types.ModuleType('asi.retrieval_trust_scorer')
rts_stub.RetrievalTrustScorer = type('RTS', (), {'score': staticmethod(lambda *a, **k: 0.0)})
mtv_stub = types.ModuleType('asi.memory_timeline_viewer')
mtv_stub.MemoryTimelineViewer = type('MTV', (), {})
kgv_stub = types.ModuleType('asi.kg_visualizer')
kgv_stub.KGVisualizer = type('KGV', (), {})
sys.modules.update({
    'asi.retrieval_explainer': re_stub,
    'asi.retrieval_visualizer': rv_stub,
    'asi.retrieval_trust_scorer': rts_stub,
    'asi.memory_timeline_viewer': mtv_stub,
    'asi.kg_visualizer': kgv_stub,
})

src_pkg = types.ModuleType('src')
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src'] = src_pkg
stub_hm = types.ModuleType('src.hierarchical_memory')
class _Mem:
    def get_stats(self):
        return {'hit_rate': 0.0}
stub_hm.MemoryServer = type('MS', (), {'__init__': lambda self,*a,**k: None, 'memory': _Mem(), 'telemetry': None})
stub_hm.HierarchicalMemory = type('HM', (), {})
sys.modules['src.hierarchical_memory'] = stub_hm
for mod_name in [
    'dashboards',
    'risk_scoreboard',
    'telemetry',
    'memory_service',
]:
    loader = importlib.machinery.SourceFileLoader(f'src.{mod_name}', f'src/{mod_name}.py')
    spec = importlib.util.spec_from_loader(loader.name, loader)
    m = importlib.util.module_from_spec(spec)
    m.__package__ = 'src'
    sys.modules[f'src.{mod_name}'] = m
    loader.exec_module(m)
    if mod_name == 'hierarchical_memory' and 'MemoryServer' not in m.__dict__:
        m.MemoryServer = type('MS', (), {})

RiskDashboard = sys.modules['src.dashboards'].RiskDashboard
MemoryDashboard = sys.modules['src.dashboards'].MemoryDashboard
RiskScoreboard = sys.modules['src.risk_scoreboard'].RiskScoreboard
TelemetryLogger = sys.modules['src.telemetry'].TelemetryLogger

class TestRiskDashboard(unittest.TestCase):
    def test_aggregate(self):
        class Mem:
            def get_stats(self):
                return {'hit_rate': 0.0}
        mem = Mem()
        logger = TelemetryLogger(interval=0.1)
        class Stub:
            def __init__(self, m, t):
                self.memory = m
                self.telemetry = t
        server = Stub(mem, logger)
        board = RiskScoreboard()
        board.update(1, 1.0, 0.5)
        dash = RiskDashboard(board, [server])
        stats = dash.aggregate()
        self.assertIn('risk_score', stats)
        self.assertIn('hit_rate', stats)

if __name__ == '__main__':
    unittest.main()
