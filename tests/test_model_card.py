import json
import tempfile
import unittest
from pathlib import Path
import importlib.machinery
import importlib.util
import types
import sys

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg

loader = importlib.machinery.SourceFileLoader('src.dataset_lineage', 'src/dataset_lineage.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
dlm = importlib.util.module_from_spec(spec)
dlm.__package__ = 'src'
sys.modules['src.dataset_lineage'] = dlm
loader.exec_module(dlm)
DatasetLineageManager = dlm.DatasetLineageManager

loader = importlib.machinery.SourceFileLoader('src.telemetry', 'src/telemetry.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
tel = importlib.util.module_from_spec(spec)
tel.__package__ = 'src'
sys.modules['src.telemetry'] = tel
loader.exec_module(tel)
TelemetryLogger = tel.TelemetryLogger

loader = importlib.machinery.SourceFileLoader('src.model_card', 'src/model_card.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mc = importlib.util.module_from_spec(spec)
mc.__package__ = 'src'
sys.modules['src.model_card'] = mc
loader.exec_module(mc)
ModelCardGenerator = mc.ModelCardGenerator


class TestModelCard(unittest.TestCase):
    def test_collect(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = DatasetLineageManager(tmp)
            p = Path(tmp) / 'a.txt'
            p.write_text('x')
            mgr.record([p], [p], note='step')
            tel_logger = TelemetryLogger()
            gen = ModelCardGenerator(mgr, tel_logger, {"acc": 1.0})
            card = gen.collect()
            self.assertIn('dataset_lineage', card)
            self.assertIn('evaluation', card)
            md = gen.to_markdown(card)
            self.assertIn('Model Card', md)


if __name__ == '__main__':
    unittest.main()
