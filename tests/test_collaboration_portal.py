import unittest
import http.client
import json
import time

from asi.collaboration_portal import CollaborationPortal
from asi.telemetry import TelemetryLogger
from asi.reasoning_history import ReasoningHistoryLogger


class TestCollaborationPortal(unittest.TestCase):
    def test_endpoints(self):
        tel = TelemetryLogger(interval=0.1)
        tel.start()
        rh = ReasoningHistoryLogger()
        rh.log("start")
        portal = CollaborationPortal(["t1"], tel, rh)
        portal.start(port=0)
        port = portal.port
        assert port is not None

        conn = http.client.HTTPConnection("localhost", port)
        conn.request("GET", "/tasks")
        tasks = json.loads(conn.getresponse().read())
        self.assertIn("t1", tasks)

        time.sleep(0.2)
        conn.request("GET", "/metrics")
        metrics = json.loads(conn.getresponse().read())
        self.assertIn("cpu", metrics)

        conn.request("GET", "/logs")
        logs = json.loads(conn.getresponse().read())
        self.assertEqual(logs[0][1], "start")

        portal.stop()
        tel.stop()


if __name__ == "__main__":  # pragma: no cover - test helper
    unittest.main()
