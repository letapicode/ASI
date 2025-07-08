import importlib.machinery
import importlib.util
import types
import sys
import unittest
try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = types.ModuleType('torch')
    class _T:
        def detach(self):
            return self
        def clone(self):
            return self
        def to(self, *a, **kw):
            return self
        def view(self, *a):
            return self
        def norm(self):
            return 0.0
        def __getitem__(self, idx):
            return _T()
        @property
        def dtype(self):
            return 0
        def item(self):
            return 0.6
        def __float__(self):
            return 0.6
    torch.randn = lambda *a, **kw: _T()
    torch.zeros = lambda *a, **kw: _T()
    torch.float32 = 0
    torch.softmax = lambda tensor, dim=0: tensor
    torch.tensor = lambda data, dtype=None: _T()
    torch.nn = types.SimpleNamespace(Module=object)
    utils_mod = types.ModuleType('torch.utils')
    data_mod = types.ModuleType('torch.utils.data')
    data_mod.Dataset = object
    data_mod.DataLoader = object
    utils_mod.data = data_mod
    torch.utils = utils_mod
    sys.modules['torch.utils'] = utils_mod
    sys.modules['torch.utils.data'] = data_mod
sys.modules['torch'] = torch
sys.modules['asi.loihi_backend'] = types.SimpleNamespace(
    LoihiConfig=object,
    configure_loihi=lambda *a, **kw: None,
    _HAS_LOIHI=False,
)
sys.modules['requests'] = types.SimpleNamespace(get=lambda *a, **kw: None)
sys.modules['asi.privacy_guard'] = types.SimpleNamespace(PrivacyGuard=object)

pkg = types.ModuleType('asi')
pkg.__path__ = ['src']
sys.modules['asi'] = pkg


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = name.rpartition('.')[0]
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

TelemetryLogger = _load('asi.telemetry', 'src/telemetry.py').TelemetryLogger
CognitiveLoadMonitor = _load('asi.cognitive_load_monitor', 'src/cognitive_load_monitor.py').CognitiveLoadMonitor
self_play_env = _load('asi.self_play_env', 'src/self_play_env.py')
robot_skill_transfer = _load('asi.robot_skill_transfer', 'src/robot_skill_transfer.py')
adaptive_curriculum = _load('asi.adaptive_curriculum', 'src/adaptive_curriculum.py')

PrioritizedReplayBuffer = self_play_env.PrioritizedReplayBuffer
VideoPolicyDataset = robot_skill_transfer.VideoPolicyDataset
AdaptiveCurriculum = adaptive_curriculum.AdaptiveCurriculum


class DummyLogger(TelemetryLogger):
    def __init__(self):
        super().__init__(interval=0.01)

    def start(self):
        pass

    def stop(self):
        pass


class TestCognitiveLoadMonitor(unittest.TestCase):
    def test_metrics(self):
        logger = DummyLogger()
        monitor = CognitiveLoadMonitor(telemetry=logger, pause_threshold=1.0)
        monitor.log_input('a', timestamp=0.0)
        monitor.log_input('b', timestamp=1.0)
        monitor.log_correction()
        m = monitor.get_metrics()
        self.assertAlmostEqual(m['avg_pause'], 1.0, delta=1e-6)
        self.assertAlmostEqual(m['correction_rate'], 0.5, delta=1e-6)
        self.assertIn('avg_pause', logger.metrics)

    def test_curriculum_bias(self):
        frames = [torch.randn(3, 2, 2) for _ in range(2)]
        actions = [0, 1]
        curated = VideoPolicyDataset(frames, actions)
        buf = PrioritizedReplayBuffer(2)
        for f, a in zip(frames, actions):
            buf.add(f, a, 1.0)
        logger = DummyLogger()
        monitor = CognitiveLoadMonitor(telemetry=logger, pause_threshold=1.0)
        monitor.log_input('a', timestamp=0.0)
        monitor.log_input('b', timestamp=2.0)
        ac = AdaptiveCurriculum(curated, buf, load_monitor=monitor)
        p = ac._probs()
        self.assertGreater(p[0].item(), 0.5)

    def test_callbacks(self):
        logger = DummyLogger()
        loads = []
        monitor = CognitiveLoadMonitor(telemetry=logger, pause_threshold=1.0)
        monitor.add_callback(loads.append)
        monitor.log_input('a', timestamp=0.0)
        monitor.log_input('b', timestamp=1.5)
        self.assertTrue(loads and isinstance(loads[-1], float))

    def test_stream_metrics(self):
        logger = DummyLogger()
        monitor = CognitiveLoadMonitor(telemetry=logger, pause_threshold=1.0)
        monitor.log_input('a', timestamp=0.0)
        monitor.log_input('b', timestamp=1.0)
        metrics = list(monitor.stream_metrics())
        self.assertTrue(metrics)
        self.assertAlmostEqual(metrics[-1]['avg_pause'], 1.0, delta=1e-6)


if __name__ == '__main__':
    unittest.main()
