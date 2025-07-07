import unittest
import importlib.machinery
import importlib.util
import types
import sys
try:
    import torch
except Exception:  # pragma: no cover - optional heavy dep
    torch = None

# Set up a minimal asi package and load required modules manually
pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

# Provide minimal stubs for dependencies
gnr_mod = types.ModuleType('asi.graph_neural_reasoner')
class DummyGNR:
    def predict_link(self, src: str, dst: str) -> float:
        return 0.0
gnr_mod.GraphNeuralReasoner = DummyGNR
sys.modules['asi.graph_neural_reasoner'] = gnr_mod

if torch is None:
    torch = types.ModuleType('torch')
    torch.nn = types.SimpleNamespace(Module=object)
    torch.tensor = lambda *a, **kw: None
    torch.zeros = lambda *a, **kw: None
    torch.long = 0
    torch.Tensor = object
sys.modules['torch'] = torch

# stub out analogical_retrieval used by graph_of_thought
ar_mod = types.ModuleType('asi.analogical_retrieval')
def analogy_search(*a, **kw):
    return ([], [])
ar_mod.analogy_search = analogy_search
sys.modules['asi.analogical_retrieval'] = ar_mod
rh_mod = types.ModuleType('asi.reasoning_history')
class DummyLogger:
    def log(self, *a, **kw):
        pass
    def get_history(self):
        return []
rh_mod.ReasoningHistoryLogger = DummyLogger
sys.modules['asi.reasoning_history'] = rh_mod
tc_mod = types.ModuleType('asi.transformer_circuit_analyzer')
class DummyTCA:
    def head_importance(self, *a, **kw):
        class Dummy(list):
            def tolist(self):
                return []
        return Dummy()
tc_mod.TransformerCircuitAnalyzer = DummyTCA
sys.modules['asi.transformer_circuit_analyzer'] = tc_mod


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'asi'
    sys.modules[name] = mod
    setattr(pkg, name.split('.')[-1], mod)
    loader.exec_module(mod)
    return mod

kg_mod = _load('asi.knowledge_graph_memory', 'src/knowledge_graph_memory.py')
KnowledgeGraphMemory = kg_mod.KnowledgeGraphMemory
TimedTriple = kg_mod.TimedTriple

csm_mod = types.ModuleType('asi.context_summary_memory')
class DummyCSM:
    pass
csm_mod.ContextSummaryMemory = DummyCSM
sys.modules['asi.context_summary_memory'] = csm_mod

got_mod = _load('asi.graph_of_thought', 'src/graph_of_thought.py')
GraphOfThought = got_mod.GraphOfThought

import torch
import types

wm_mod = types.ModuleType('asi.world_model_rl')

class DummyWM(torch.nn.Module):
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        return state + 1, 0.0
    __call__ = forward

def rollout_policy(model: DummyWM, policy, init_state: torch.Tensor, steps: int = 1):
    state = init_state
    states = []
    rewards = []
    for _ in range(steps):
        action = policy(state)
        state, r = model(state, action)
        states.append(state)
        rewards.append(float(r))
    return states, rewards

wm_mod.WorldModel = DummyWM
wm_mod.rollout_policy = rollout_policy
sys.modules['asi.world_model_rl'] = wm_mod
WorldModel = DummyWM

tr_mod = _load('asi.temporal_reasoner', 'src/temporal_reasoner.py')
TemporalReasoner = tr_mod.TemporalReasoner

hp_mod = _load('asi.hierarchical_planner', 'src/hierarchical_planner.py')
HierarchicalPlanner = hp_mod.HierarchicalPlanner


class TestTemporalReasoner(unittest.TestCase):
    def test_infer_order(self):
        kg = KnowledgeGraphMemory()
        kg.add_triples([
            TimedTriple('e', 'r', 'a', 2.0),
            TimedTriple('e', 'r', 'b', 1.0),
        ])
        reasoner = TemporalReasoner(kg)
        ordered = reasoner.infer_order([('e', 'r', 'a'), ('e', 'r', 'b')])
        self.assertEqual([t.object for t in ordered], ['b', 'a'])

    def test_planner_ordering(self):
        kg = KnowledgeGraphMemory()
        kg.add_triples([
            TimedTriple('s', 'r', 'start', 0.0),
            TimedTriple('m1', 'r', 'mid1', 3.0),
            TimedTriple('m2', 'r', 'mid2', 1.0),
            TimedTriple('g', 'r', 'goal', 2.0),
        ])
        reasoner = TemporalReasoner(kg)

        g = GraphOfThought()
        g.add_step('start', metadata={'triple': ('s', 'r', 'start')}, node_id=0)
        g.add_step('step1', metadata={'triple': ('m1', 'r', 'mid1')}, node_id=1)
        g.add_step('step2', metadata={'triple': ('m2', 'r', 'mid2')}, node_id=2)
        g.add_step('done', metadata={'triple': ('g', 'r', 'goal')}, node_id=3)
        g.connect(0, 1)
        g.connect(1, 2)
        g.connect(2, 3)

        wm = WorldModel()
        planner = HierarchicalPlanner(
            g,
            wm,
            lambda s: 0,
            temporal_reasoner=reasoner,
        )
        path, states, rewards = planner.compose_plan(
            0, lambda n: n.id == 3, 0, rollout_steps=1, use_temporal=True
        )
        self.assertEqual(path, [0, 2, 1, 3])
        self.assertEqual(len(states), len(path) + 1)
        self.assertEqual(len(rewards), len(path))

    def test_compress(self):
        kg = KnowledgeGraphMemory()
        reasoner = TemporalReasoner(kg)

        g = GraphOfThought()
        g.add_step('start', node_id=0, timestamp=0.0)
        g.add_step('a', node_id=1, timestamp=1.0)
        g.add_step('b', node_id=2, timestamp=1.0)
        g.add_step('end', node_id=3, timestamp=2.0)
        g.connect(0, 1)
        g.connect(1, 2)
        g.connect(2, 3)
        ordered = reasoner.order_nodes_by_time(g, [0, 1, 2, 3], compress=True)
        self.assertEqual(ordered, [0, 1, 3])


if __name__ == '__main__':
    unittest.main()
