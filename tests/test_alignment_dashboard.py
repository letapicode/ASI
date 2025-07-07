import http.client
import json
import unittest
import importlib.machinery
import importlib.util
import types
import sys

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'asi'
    loader.exec_module(mod)
    sys.modules[name] = mod
    setattr(pkg, name.split('.')[-1], mod)
    return mod

AlignmentDashboard = _load('asi.alignment_dashboard', 'src/alignment_dashboard.py').AlignmentDashboard


class TestAlignmentDashboard(unittest.TestCase):
    def test_server_and_record(self):
        dash = AlignmentDashboard()
        dash.record(True, ["ok"], ["bad"])
        dash.start(port=0)
        port = dash.port
        conn = http.client.HTTPConnection("localhost", port)
        conn.request("GET", "/stats")
        data = json.loads(conn.getresponse().read())
        self.assertIn("pass_rate", data)
        self.assertEqual(data["flagged_examples"], ["ok"])
        self.assertEqual(data["normative_violations"], ["bad"])
        dash.stop()


if __name__ == "__main__":
    unittest.main()
