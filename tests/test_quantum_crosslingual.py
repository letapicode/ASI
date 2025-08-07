import importlib.machinery
import importlib.util
import sys
import types
import unittest
import time
import numpy as np

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'asi'
    loader.exec_module(mod)
    sys.modules[name] = mod
    setattr(pkg, name.split('.')[-1], mod)
    return mod


qs = load('asi.quantum_sampling', 'src/quantum_sampling.py')
clm = load('asi.cross_lingual_memory', 'src/cross_lingual_memory.py')
di = load('asi.data_ingest', 'src/data_ingest.py')

CrossLingualMemory = clm.CrossLingualMemory
CrossLingualTranslator = di.CrossLingualTranslator


class TestQuantumCrossLingual(unittest.TestCase):
    def test_quantum_retrieval(self):
        rng = np.random.default_rng(0)
        tr = CrossLingualTranslator(['es'])
        mem = CrossLingualMemory(dim=8, compressed_dim=4, capacity=100, translator=tr)
        data = rng.normal(size=(40, 8)).astype(np.float32)
        mem.add(data, metadata=list(range(len(data))))
        queries = data[:10] + rng.normal(scale=0.01, size=(10, 8)).astype(np.float32)

        start = time.perf_counter()
        classical_hits = 0
        for i, q in enumerate(queries):
            _, meta = mem.search(q, k=1)
            if meta and meta[0] == i:
                classical_hits += 1
        classical_time = time.perf_counter() - start

        start = time.perf_counter()
        quantum_hits = 0
        for i, q in enumerate(queries):
            _, meta = mem.search(q, k=1, quantum=True)
            if meta and meta[0] == i:
                quantum_hits += 1
        quantum_time = time.perf_counter() - start

        self.assertGreaterEqual(quantum_hits, classical_hits - 2)
        self.assertLessEqual(quantum_time, classical_time * 2)


if __name__ == '__main__':
    unittest.main()
