import unittest
import importlib.machinery
import importlib.util
import types
import sys

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'asi'
    loader.exec_module(mod)
    sys.modules[name] = mod
    setattr(pkg, name.split('.')[-1], mod)
    return mod

GraphOfThought = load('asi.graph_of_thought', 'src/graph_of_thought.py').GraphOfThought
KnowledgeGraphMemory = load('asi.knowledge_graph_memory', 'src/knowledge_graph_memory.py').KnowledgeGraphMemory
ReasoningHistoryLogger = load('asi.reasoning_history', 'src/reasoning_history.py').ReasoningHistoryLogger
bridge = load('asi.reasoning_kb_bridge', 'src/reasoning_kb_bridge.py')

graph_to_triples = bridge.graph_to_triples
HistoryKGExporter = bridge.HistoryKGExporter
get_following_steps = bridge.get_following_steps
get_step_metadata = bridge.get_step_metadata


class TestReasoningKGBride(unittest.TestCase):
    def test_graph_conversion_and_query(self):
        g = GraphOfThought()
        a = g.add_step('start', metadata={'timestamp': 1.0})
        b = g.add_step('end', metadata={'timestamp': 2.0})
        g.connect(a, b)
        kg = KnowledgeGraphMemory()
        kg.add_triples(graph_to_triples(g))
        next_steps = get_following_steps(kg, 'start')
        self.assertEqual(next_steps, ['end'])
        meta = get_step_metadata(kg, 'start', 'node_id')
        self.assertEqual(meta, [str(a)])

    def test_history_exporter(self):
        logger = ReasoningHistoryLogger()
        logger.log('first')
        kg = KnowledgeGraphMemory()
        exporter = HistoryKGExporter(logger, kg, interval=0.01)
        exporter.export_once()
        triples = kg.query_triples(subject='reasoning')
        self.assertEqual(len(triples), 1)
        exporter.stop()


if __name__ == '__main__':
    unittest.main()
