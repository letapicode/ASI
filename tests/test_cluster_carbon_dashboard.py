import unittest
import json
import http.client
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
    loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

ClusterCarbonDashboard = _load('asi.cluster_carbon_dashboard', 'src/cluster_carbon_dashboard.py').ClusterCarbonDashboard


class TestClusterCarbonDashboard(unittest.TestCase):
    def test_aggregate(self):
        dash = ClusterCarbonDashboard()
        dash.start(port=0)
        port = dash.port
        conn = http.client.HTTPConnection('localhost', port)
        data1 = json.dumps({'node_id': 'n1', 'energy_kwh': 1.0, 'carbon_g': 2.0})
        conn.request('POST', '/update', body=data1)
        conn.getresponse().read()
        data2 = json.dumps({'node_id': 'n2', 'energy_kwh': 0.5, 'carbon_g': 1.0})
        conn.request('POST', '/update', body=data2)
        conn.getresponse().read()
        conn.request('GET', '/stats')
        resp = conn.getresponse()
        stats = json.loads(resp.read())
        dash.stop()
        self.assertEqual(stats['total']['energy_kwh'], 1.5)
        self.assertEqual(stats['nodes']['n1']['carbon_g'], 2.0)


if __name__ == '__main__':
    unittest.main()
