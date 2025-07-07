import importlib.machinery
import importlib.util
import types
import sys

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

sys.modules['torch'] = types.SimpleNamespace(
    full=lambda shape, val, **kw: [val] * shape[0],
    zeros=lambda dim, device=None: [0.0] * dim,
)

class DummyEnv:
    def __init__(self, state_dim, device=None):
        self.state = [0.0] * state_dim
    def reset(self):
        self.state = [0.0] * len(self.state)
        return self.state
    def step(self, action):
        self.state = [s + float(a) for s, a in zip(self.state, action)]
        return types.SimpleNamespace(observation=self.state, reward=-1.0, done=False)

class DummyAgent:
    actions = ('a', 'b', 'c')
    def select_action(self, state):
        return 'a'
    def update(self, s, a, r, ns):
        pass
    def train(self, entries):
        pass

class DummyCoordinator:
    def __init__(self, agents):
        self.agents = agents
        self.log = []
    async def schedule_round(self, tasks, apply_fn=None, reward_fn=None):
        for t in tasks:
            for name in self.agents:
                if apply_fn:
                    await apply_fn(name, 'a')
                if reward_fn:
                    reward_fn(name, 'a')
                self.log.append((name, t, 'a', 0.0))
    def train_agents(self):
        pass

class DummyDashboard:
    def __init__(self, coord):
        self.coord = coord
    def start(self, port=0):
        pass
    def stop(self):
        pass
    def aggregate(self):
        return {'assignments': self.coord.log}

sys.modules['asi.meta_rl_refactor'] = types.SimpleNamespace(MetaRLRefactorAgent=DummyAgent)
sys.modules['asi.self_play_env'] = types.SimpleNamespace(SimpleEnv=DummyEnv)
sys.modules['asi.multi_agent_coordinator'] = types.SimpleNamespace(MultiAgentCoordinator=DummyCoordinator)
sys.modules['asi.multi_agent_dashboard'] = types.SimpleNamespace(MultiAgentDashboard=DummyDashboard)

def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

msp = _load('asi.multi_agent_self_play', 'src/multi_agent_self_play.py')
MultiAgentSelfPlayConfig = msp.MultiAgentSelfPlayConfig
run_multi_agent_self_play = msp.run_multi_agent_self_play

class TestMultiAgentSelfPlay:
    def test_run(self):
        cfg = MultiAgentSelfPlayConfig(num_agents=2, steps=2, env_state_dim=3)
        dash = run_multi_agent_self_play(cfg)
        data = dash.aggregate()
        assert 'assignments' in data
        assert len(data['assignments']) > 0

