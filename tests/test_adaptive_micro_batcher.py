import importlib.machinery
import importlib.util
import types
import sys
import unittest

pkg = types.ModuleType('asi')
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
AdaptiveMicroBatcher = _load('asi.adaptive_micro_batcher', 'src/adaptive_micro_batcher.py').AdaptiveMicroBatcher


class DummyLogger(TelemetryLogger):
    def __init__(self, seq):
        super().__init__(interval=0.01)
        self.seq = list(seq)

    def start(self):
        pass

    def stop(self):
        pass

    def get_stats(self):
        val = self.seq.pop(0) if self.seq else 0.0
        return {"gpu_mem": val}


class TestAdaptiveMicroBatcher(unittest.TestCase):
    def test_adjustment(self):
        logger = DummyLogger([80, 80, 20, 20])
        batcher = AdaptiveMicroBatcher(4, min_size=1, max_size=8, telemetry=logger, high_mem=0.75, low_mem=0.25)
        sizes = [batcher.tick() for _ in range(4)]
        self.assertEqual(sizes[1], 1)
        self.assertGreaterEqual(sizes[-1], 2)

    def test_micro_batches(self):
        logger = DummyLogger([90, 10])
        batcher = AdaptiveMicroBatcher(4, min_size=2, max_size=4, telemetry=logger, high_mem=0.8, low_mem=0.2)
        data = list(range(6))
        batches = list(batcher.micro_batches(data))
        self.assertEqual(len(batches[0]), 4)
        self.assertGreaterEqual(len(batches[1]), 2)


if __name__ == '__main__':
    unittest.main()

