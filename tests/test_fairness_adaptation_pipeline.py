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
sys.modules['asi.data_provenance_ledger'] = types.SimpleNamespace(DataProvenanceLedger=lambda root: None)

pkg = types.ModuleType('asi')
pkg.__path__ = ['src']
sys.modules['asi'] = pkg

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
dlm_mod = _load('asi.dataset_lineage_manager', 'src/dataset_lineage_manager.py')
fap_mod = _load('asi.fairness_adaptation', 'src/fairness_adaptation.py')
fe_mod = _load('asi.fairness_evaluator', 'src/fairness_evaluator.py')

ActiveDataSelector = data_ingest.ActiveDataSelector
DatasetBiasDetector = bias_mod.DatasetBiasDetector
CognitiveLoadMonitor = clm_mod.CognitiveLoadMonitor
FairnessAdaptationPipeline = fap_mod.FairnessAdaptationPipeline
FairnessEvaluator = fe_mod.FairnessEvaluator


class TestFairnessAdaptationPipeline(unittest.TestCase):
    def test_demographic_parity_improves(self):
        selector = ActiveDataSelector(threshold=1.0)
        bias = DatasetBiasDetector()
        load = CognitiveLoadMonitor()
        pipeline = FairnessAdaptationPipeline(selector, bias, load)

        with tempfile.TemporaryDirectory() as d:
            g1_p = Path(d) / 'g1_p.txt'
            g1_p2 = Path(d) / 'g1_p2.txt'
            g2_n = Path(d) / 'g2_n.txt'
            g2_n2 = Path(d) / 'g2_n2.txt'
            g1_p.write_text('a b c d')
            g1_p2.write_text('a b c d')
            g2_n.write_text('hello hello hello')
            g2_n2.write_text('hello hello hello')

            triples = [(g1_p, 'i', 'a'), (g1_p2, 'i', 'a'), (g2_n, 'i', 'a'), (g2_n2, 'i', 'a')]
            probs = [np.array([0.5, 0.5])] * 4
            groups = ['g1', 'g1', 'g2', 'g2']
            labels = [1, 1, 0, 0]

            weights = pipeline.process(triples, probs, groups, labels)
            fe = FairnessEvaluator()
            before = fe.demographic_parity(pipeline._stats(groups, labels), '1')
            after = fe.demographic_parity(pipeline._stats(groups, labels, [w for _, w in weights]), '1')
            self.assertGreater(before, after)


if __name__ == '__main__':
    unittest.main()
