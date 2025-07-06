import unittest
import importlib.machinery
import importlib.util
import types
import sys
import http.client
import json
import time

pkg = types.ModuleType("asi")
pkg.__path__ = ["src"]
sys.modules["asi"] = pkg


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "asi"
    sys.modules[name] = mod
    loader.exec_module(mod)
    setattr(pkg, name.split(".")[-1], mod)
    return mod


TelemetryLogger = load("asi.telemetry", "src/telemetry.py").TelemetryLogger
ReasoningHistoryLogger = load(
    "asi.reasoning_history", "src/reasoning_history.py"
).ReasoningHistoryLogger
sys.modules["asi.telemetry"]._HAS_PROM = False


class CrossLingualTranslator:
    def __init__(self, languages):
        self.languages = list(languages)

    def translate(self, text, lang):
        if lang not in self.languages:
            raise ValueError("unsupported language")
        return f"[{lang}] {text}"

    def translate_all(self, text):
        return {l: self.translate(text, l) for l in self.languages}


dummy_tr = types.ModuleType("asi.data_ingest")
dummy_tr.CrossLingualTranslator = CrossLingualTranslator
sys.modules["asi.data_ingest"] = dummy_tr

CollaborationPortal = load(
    "asi.collaboration_portal", "src/collaboration_portal.py"
).CollaborationPortal


class TestCollaborationPortal(unittest.TestCase):
    def test_endpoints(self):
        tel = TelemetryLogger(interval=0.1)
        tel.start()
        rh = ReasoningHistoryLogger()
        rh.log("start")
        tr = CrossLingualTranslator(["es"])
        portal = CollaborationPortal(["t1"], tel, rh, translator=tr)
        portal.start(port=0)
        port = portal.port
        assert port is not None

        conn = http.client.HTTPConnection("localhost", port)
        conn.request("GET", "/tasks?lang=es")
        tasks = json.loads(conn.getresponse().read())
        self.assertEqual(tasks[0], "[es] t1")

        time.sleep(0.2)
        conn.request("GET", "/metrics?lang=es")
        metrics = json.loads(conn.getresponse().read())
        self.assertTrue(any(k.startswith("[es]") for k in metrics))

        conn.request("GET", "/logs", headers={"Accept-Language": "es"})
        logs = json.loads(conn.getresponse().read())
        self.assertTrue(logs[0][1].startswith("[es]"))

        portal.stop()
        tel.stop()


if __name__ == "__main__":  # pragma: no cover - test helper
    unittest.main()
