import importlib.machinery
import importlib.util
import sys
import time
import types
import unittest
import torch

pkg = types.ModuleType('asi')
pkg.__path__ = ['src']
pkg.__spec__ = importlib.machinery.ModuleSpec('asi', None, is_package=True)
sys.modules['asi'] = pkg

# stub cryptography dependency required by encrypted_vector_store
crypto = types.ModuleType('cryptography')
haz = types.ModuleType('cryptography.hazmat')
prim = types.ModuleType('cryptography.hazmat.primitives')
ci = types.ModuleType('cryptography.hazmat.primitives.ciphers')
aead = types.ModuleType('cryptography.hazmat.primitives.ciphers.aead')
class AESGCM:
    pass
aead.AESGCM = AESGCM
ci.aead = aead
prim.ciphers = ci
haz.primitives = prim
crypto.hazmat = haz
sys.modules['cryptography'] = crypto
sys.modules['cryptography.hazmat'] = haz
sys.modules['cryptography.hazmat.primitives'] = prim
sys.modules['cryptography.hazmat.primitives.ciphers'] = ci
sys.modules['cryptography.hazmat.primitives.ciphers.aead'] = aead
dummy_di = types.ModuleType('asi.data_ingest')
dummy_di.CrossLingualTranslator = None
dummy_di.__spec__ = importlib.machinery.ModuleSpec('asi.data_ingest', None)
sys.modules['asi.data_ingest'] = dummy_di
sys.modules['requests'] = types.ModuleType('requests')
psutil_stub = types.SimpleNamespace(
    cpu_percent=lambda interval=None: 0.0,
    virtual_memory=lambda: types.SimpleNamespace(percent=0.0),
    net_io_counters=lambda: types.SimpleNamespace(bytes_sent=0, bytes_recv=0),
)
sys.modules['psutil'] = psutil_stub

def _load(name: str, path: str):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

hier = _load('asi.hierarchical_memory', 'src/hierarchical_memory.py')
RetrievalPolicy = _load('asi.retrieval_rl', 'src/retrieval_rl.py').RetrievalPolicy
RetrievalPolicyUpdater = _load('asi.retrieval_policy_updater', 'src/retrieval_policy_updater.py').RetrievalPolicyUpdater
TelemetryLogger = _load('asi.telemetry', 'src/telemetry.py').TelemetryLogger


class TestRetrievalPolicyUpdater(unittest.TestCase):
    def test_policy_improves(self):
        policy = RetrievalPolicy(epsilon=0.0, alpha=1.0)
        mem = hier.HierarchicalMemory(
            dim=2,
            compressed_dim=2,
            capacity=10,
            retrieval_policy=policy,
            encryption_key=b'0'*16,
        )
        good = torch.tensor([[0.5, 0.0]])
        bad = torch.tensor([[0.8, 0.0]])
        mem.add(torch.cat([bad, good]), metadata=['bad', 'good'])
        q = torch.tensor([1.0, 0.0])
        _, meta_before = mem.search(q, k=2)
        self.assertEqual(meta_before[0], 'bad')

        logs = [('good', True, 0.01), ('bad', False, 0.01)]

        def load_logs():
            nonlocal logs
            out, logs = logs, []
            return out

        logger = TelemetryLogger(interval=0.1)
        updater = RetrievalPolicyUpdater(policy, load_logs, interval=0.05,
                                          telemetry=logger)
        updater.start()
        time.sleep(0.1)
        updater.stop()

        _, meta_after = mem.search(q, k=2)
        self.assertEqual(meta_after[0], 'good')
        events = logger.get_events()
        self.assertTrue(any(e['metric'] == 'recall_improvement' for e in events))


if __name__ == '__main__':
    unittest.main()
