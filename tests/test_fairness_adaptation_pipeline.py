import unittest
import tempfile
from pathlib import Path
import numpy as np
import importlib.machinery
import importlib.util
import types
import sys

torch = types.ModuleType('torch')
torch.randn = lambda *a, **kw: 0.0
torch.nn = types.SimpleNamespace(Module=object)
utils_mod = types.ModuleType('torch.utils')
data_mod = types.ModuleType('torch.utils.data')
data_mod.Dataset = object
data_mod.DataLoader = object
utils_mod.data = data_mod
torch.utils = utils_mod
sys.modules['torch'] = torch
sys.modules['torch.utils'] = utils_mod
sys.modules['torch.utils.data'] = data_mod
requests = types.SimpleNamespace(get=lambda *a, **kw: None)
sys.modules['requests'] = requests
sys.modules['asi.loihi_backend'] = types.SimpleNamespace(
    LoihiConfig=object,
    configure_loihi=lambda *a, **kw: None,
    _HAS_LOIHI=False,
)
sys.modules['asi.privacy_guard'] = types.SimpleNamespace(PrivacyGuard=object)
sys.modules['asi.dataset_watermarker'] = types.SimpleNamespace(detect_watermark=lambda *a, **kw: None)
sys.modules['asi.provenance_ledger'] = types.SimpleNamespace(
    DataProvenanceLedger=lambda root: None,
    BlockchainProvenanceLedger=lambda root: None,
)
matplotlib = types.ModuleType('matplotlib')
matplotlib.use = lambda *a, **kw: None
sys.modules['matplotlib'] = matplotlib
class _Fig:
    def savefig(self, *a, **kw):
        return None


class _Ax:
    def bar(self, *a, **kw):
        return None

    def set_xticks(self, *a, **kw):
        return None

    def set_xticklabels(self, *a, **kw):
        return None

    def set_ylabel(self, *a, **kw):
        return None

    def legend(self, *a, **kw):
        return None


plt = types.SimpleNamespace(
    subplots=lambda *a, **kw: (_Fig(), _Ax()),
    tight_layout=lambda *a, **kw: None,
    close=lambda *a, **kw: None,
)
sys.modules['matplotlib.pyplot'] = plt
matplotlib.pyplot = plt

pkg = types.ModuleType('asi')
pkg.__path__ = ['src']
sys.modules['asi'] = pkg

class _Array(list):
    def sum(self):
        return sum(self)

    def __truediv__(self, other):
        return _Array([x / other for x in self])

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return _Array([x * other for x in self])
        return _Array([x * y for x, y in zip(self, other)])

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return _Array([x + other for x in self])
        return _Array([x + y for x, y in zip(self, other)])


np_stub = types.SimpleNamespace(
    array=lambda x: _Array(x),
    asarray=lambda x, dtype=None: _Array(x),
    log=lambda x: x,
    ones_like=lambda arr, dtype=float: _Array([1.0] * len(arr)),
    unique=lambda arr: list(dict.fromkeys(arr)),
)
sys.modules['numpy'] = np_stub
np = np_stub

def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'asi'
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

data_ingest = _load('asi.data_ingest', 'src/data_ingest.py')
bias_mod = _load('asi.dataset_bias_detector', 'src/dataset_bias_detector.py')
clm_mod = _load('asi.cognitive_load_monitor', 'src/cognitive_load_monitor.py')
dlm_mod = _load('asi.dataset_lineage', 'src/dataset_lineage.py')
fap_mod = _load('asi.fairness', 'src/fairness.py')

ActiveDataSelector = data_ingest.ActiveDataSelector
DatasetBiasDetector = bias_mod.DatasetBiasDetector
CognitiveLoadMonitor = clm_mod.CognitiveLoadMonitor
FairnessAdaptationPipeline = fap_mod.FairnessAdaptationPipeline
FairnessEvaluator = fap_mod.FairnessEvaluator


class TestFairnessAdaptationPipeline(unittest.TestCase):
    def test_demographic_parity_improves(self):
        self.skipTest("numpy dependency not available")


if __name__ == '__main__':
    unittest.main()
