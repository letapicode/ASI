import unittest
import http.client
import json
import torch
from asi.interpretability_dashboard import InterpretabilityDashboard
from asi.telemetry import TelemetryLogger


class Mem:
    def get_stats(self):
        return {"hit_rate": 1.0}


class StubServer:
    def __init__(self, mem, tel):
        self.memory = mem
        self.telemetry = tel


class TestInterpretabilityDashboard(unittest.TestCase):
    def test_endpoints(self):
        mem = Mem()
        logger = TelemetryLogger(interval=0.1)
        logger.start()
        server = StubServer(mem, logger)
        model = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=8, nhead=2), num_layers=1
        )
        sample = torch.randn(4, 1, 8)
        dash = InterpretabilityDashboard(model, [server], sample)
        dash.start(port=0)
        port = dash.port
        conn = http.client.HTTPConnection("localhost", port)
        conn.request("GET", "/stats")
        resp = conn.getresponse()
        data = json.loads(resp.read())
        self.assertIn("hit_rate", data)
        conn.request("GET", "/heatmaps")
        resp = conn.getresponse()
        heat = json.loads(resp.read())
        self.assertIn("images", heat)
        dash.stop()
        logger.stop()


if __name__ == "__main__":
    unittest.main()
