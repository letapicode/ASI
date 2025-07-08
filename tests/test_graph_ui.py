import unittest
import importlib.machinery
import importlib.util
import types
import sys
import json
import http.client

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

GraphOfThought = _load('asi.graph_of_thought', 'src/graph_of_thought.py').GraphOfThought
ReasoningHistoryLogger = _load('asi.reasoning_history', 'src/reasoning_history.py').ReasoningHistoryLogger
_load('asi.nl_graph_editor', 'src/nl_graph_editor.py')
GraphUI = _load('asi.graph_ui', 'src/graph_ui.py').GraphUI
TelemetryLogger = _load('asi.telemetry', 'src/telemetry.py').TelemetryLogger
CognitiveLoadMonitor = _load('asi.cognitive_load_monitor', 'src/cognitive_load_monitor.py').CognitiveLoadMonitor

class CrossLingualTranslator:
    def __init__(self, languages):
        self.languages = list(languages)

    def translate(self, text, lang):
        if lang not in self.languages:
            raise ValueError('unsupported language')
        return f'[{lang}] {text}'

    def translate_all(self, text):
        return {l: self.translate(text, l) for l in self.languages}

dummy_tr = types.ModuleType('asi.data_ingest')
dummy_tr.CrossLingualTranslator = CrossLingualTranslator
sys.modules['asi.data_ingest'] = dummy_tr

CrossLingualReasoningGraph = _load('asi.cross_lingual_graph', 'src/cross_lingual_graph.py').CrossLingualReasoningGraph


class DummyLogger(TelemetryLogger):
    def __init__(self):
        super().__init__(interval=0.01)

    def start(self):
        pass

    def stop(self):
        pass


class TestGraphUI(unittest.TestCase):
    def test_endpoints(self):
        g = GraphOfThought()
        a = g.add_step('start')
        b = g.add_step('finish')
        g.connect(a, b)
        logger = ReasoningHistoryLogger()
        ui = GraphUI(g, logger)
        ui.start(port=0)
        port = ui.port
        conn = http.client.HTTPConnection('localhost', port)
        # add node
        body = json.dumps({'text': 'mid'})
        conn.request('POST', '/graph/node', body, {'Content-Type': 'application/json'})
        resp = conn.getresponse()
        node_id = json.loads(resp.read())['id']
        conn.request('POST', '/graph/edge', json.dumps({'src': a, 'dst': node_id}),
                     {'Content-Type': 'application/json'})
        conn.getresponse().read()
        conn.request('POST', '/graph/nl_edit', json.dumps({'command': 'add node extra'}),
                     {'Content-Type': 'application/json'})
        conn.getresponse().read()
        conn.request('POST', '/graph/recompute')
        summary = json.loads(conn.getresponse().read())['summary']
        self.assertIn('start', summary)
        conn.request('GET', '/graph/data')
        resp = conn.getresponse()
        data = json.loads(resp.read())
        self.assertEqual(len(data['nodes']), 4)
        conn.request('GET', '/history')
        resp = conn.getresponse()
        hist = json.loads(resp.read())
        self.assertGreaterEqual(len(hist), 2)
        ui.stop()

    def test_high_load(self):
        g = GraphOfThought()
        g.add_step('long text ' * 10)
        logger = ReasoningHistoryLogger()
        tele = DummyLogger()
        monitor = CognitiveLoadMonitor(telemetry=tele, pause_threshold=1.0)
        ui = GraphUI(g, logger, load_monitor=monitor, throttle_threshold=0.5, update_interval=0.1, telemetry=tele)
        ui.start(port=0)
        monitor.log_input('a', timestamp=0.0)
        monitor.log_input('b', timestamp=2.0)
        conn = http.client.HTTPConnection('localhost', ui.port)
        conn.request('GET', '/graph/data')
        data = json.loads(conn.getresponse().read())
        self.assertTrue(data['nodes'][0]['text'].endswith('...'))
        ui.stop()

    def test_search(self):
        tr = CrossLingualTranslator(['es'])
        g = CrossLingualReasoningGraph(translator=tr)
        g.add_step('hello', lang='en')
        logger = ReasoningHistoryLogger()
        ui = GraphUI(g, logger)
        ui.start(port=0)
        conn = http.client.HTTPConnection('localhost', ui.port)
        conn.request('GET', '/graph/search?query=hello&lang=es')
        res = json.loads(conn.getresponse().read())
        self.assertEqual(res[0]['text'], '[es] hello')
        ui.stop()


if __name__ == '__main__':
    unittest.main()
