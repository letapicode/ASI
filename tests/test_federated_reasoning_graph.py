import importlib.machinery
import importlib.util
import types
import sys
import time
import unittest

# Stub heavy deps
torch = types.ModuleType('torch')
sys.modules['torch'] = torch

# Stub GraphOfThought to avoid dataclass dependencies
class SimpleNode:
    def __init__(self, id_, text, metadata=None):
        self.id = id_
        self.text = text
        self.metadata = metadata or {}

class GraphOfThought:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self._next = 0
    def add_step(self, text, metadata=None, node_id=None):
        if node_id is None:
            node_id = self._next
            self._next += 1
        self.nodes[node_id] = SimpleNode(node_id, text, metadata)
        self.edges.setdefault(node_id, [])
        return node_id
    def connect(self, src, dst):
        self.edges.setdefault(src, []).append(dst)

mod = types.ModuleType('graph_of_thought')
mod.GraphOfThought = GraphOfThought
sys.modules['asi.graph_of_thought'] = mod

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
setattr(pkg, 'graph_of_thought', mod)


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'asi'
    loader.exec_module(mod)
    sys.modules[name] = mod
    setattr(pkg, name.split('.')[-1], mod)
    return mod

load('asi.reasoning_graph_pb2', 'src/reasoning_graph_pb2.py')
sys.modules['reasoning_graph_pb2'] = sys.modules['asi.reasoning_graph_pb2']
load('asi.reasoning_graph_pb2_grpc', 'src/reasoning_graph_pb2_grpc.py')
sys.modules['reasoning_graph_pb2_grpc'] = sys.modules['asi.reasoning_graph_pb2_grpc']
frg = load('asi.federated_reasoning_graph', 'src/federated_reasoning_graph.py')
FederatedReasoningGraph = frg.FederatedReasoningGraph
push_node_remote = frg.push_node_remote


class TestFederatedReasoningGraph(unittest.TestCase):
    def test_convergence(self):
        try:
            import grpc  # noqa: F401
        except Exception:
            self.skipTest('grpcio not available')

        g1 = GraphOfThought()
        g2 = GraphOfThought()
        s1 = FederatedReasoningGraph(g1, 'localhost:50900', peers=['localhost:50901'])
        s2 = FederatedReasoningGraph(g2, 'localhost:50901', peers=['localhost:50900'])
        s1.start()
        s2.start()

        push_node_remote('localhost:50900', 'A', key='a')
        push_node_remote('localhost:50901', 'B', edges=['a'], key='b')
        time.sleep(0.2)

        s1.stop(0)
        s2.stop(0)

        self.assertEqual({n.text for n in g1.nodes.values()}, {n.text for n in g2.nodes.values()})
        b1 = s1.id_map['b']
        b2 = s2.id_map['b']
        a1 = s1.id_map['a']
        a2 = s2.id_map['a']
        self.assertEqual(g1.edges[b1], [a1])
        self.assertEqual(g2.edges[b2], [a2])


if __name__ == '__main__':
    unittest.main()
