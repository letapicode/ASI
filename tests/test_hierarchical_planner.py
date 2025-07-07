import unittest
import importlib.machinery
import importlib.util
import types
import sys
try:
    import torch
except Exception:  # pragma: no cover - optional heavy dep
    torch = None

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

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

# stub analogical retrieval
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
csm_mod = types.ModuleType('asi.context_summary_memory')
class DummyCSM:
    pass
csm_mod.ContextSummaryMemory = DummyCSM
sys.modules['asi.context_summary_memory'] = csm_mod

def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'asi'
    sys.modules[name] = mod
    setattr(pkg, name.split('.')[-1], mod)
    loader.exec_module(mod)
    return mod

got_mod = _load('asi.graph_of_thought', 'src/graph_of_thought.py')
GraphOfThought = got_mod.GraphOfThought

kg_mod = _load('asi.knowledge_graph_memory', 'src/knowledge_graph_memory.py')
KnowledgeGraphMemory = kg_mod.KnowledgeGraphMemory
TimedTriple = kg_mod.TimedTriple

wm_mod = types.ModuleType('asi.world_model_rl')

class DummyWM(torch.nn.Module):
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        return state, 0.0
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

class TestHierarchicalPlanner(unittest.TestCase):
    def test_compose_plan(self):
        model = WorldModel()
        graph = GraphOfThought()
        graph.add_step("start", node_id=0)
        graph.add_step("goal", node_id=1)
        graph.connect(0, 1)
        planner = HierarchicalPlanner(graph, model, lambda s: 0)
        path, states, rewards = planner.compose_plan(0, lambda n: n.id == 1, 0)
        self.assertEqual(path, [0, 1])
        self.assertEqual(len(states), len(path) + 1)
        self.assertEqual(len(rewards), len(path))

if __name__ == "__main__":
    unittest.main()
