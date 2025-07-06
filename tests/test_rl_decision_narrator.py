import importlib.machinery
import importlib.util
import sys
import torch
import unittest

# dynamic imports
loader = importlib.machinery.SourceFileLoader('rl_decision_narrator', 'src/rl_decision_narrator.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
rl_narrator = importlib.util.module_from_spec(spec)
loader.exec_module(rl_narrator)
RLDecisionNarrator = rl_narrator.RLDecisionNarrator

loader = importlib.machinery.SourceFileLoader('meta_rl_refactor', 'src/meta_rl_refactor.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
meta_mod = importlib.util.module_from_spec(spec)
loader.exec_module(meta_mod)
MetaRLRefactorAgent = meta_mod.MetaRLRefactorAgent

loader = importlib.machinery.SourceFileLoader('wmrl', 'src/world_model_rl.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
wmrl = importlib.util.module_from_spec(spec)
loader.exec_module(wmrl)
RLBridgeConfig = wmrl.RLBridgeConfig
TransitionDataset = wmrl.TransitionDataset
train_world_model = wmrl.train_world_model
rollout_policy = wmrl.rollout_policy

ReasoningHistoryLogger = importlib.import_module('asi.reasoning_history').ReasoningHistoryLogger


class TestRLDecisionNarrator(unittest.TestCase):
    def test_meta_agent_logs(self):
        logger = ReasoningHistoryLogger()
        narrator = RLDecisionNarrator(logger)
        agent = MetaRLRefactorAgent(epsilon=0.0, narrator=narrator)
        agent.update('s1', 'replace', 1.0, 's2')
        agent.select_action('s1')
        hist = logger.get_history()
        self.assertEqual(len(hist), 1)
        self.assertIn('replace', hist[0][1])

    def test_rollout_logs(self):
        cfg = RLBridgeConfig(state_dim=2, action_dim=2, epochs=1, batch_size=1)
        transitions = []
        for _ in range(3):
            s = torch.randn(2)
            a = torch.randint(0, 2, (1,)).item()
            ns = torch.randn(2)
            r = torch.randn(())
            transitions.append((s, a, ns, r))
        dataset = TransitionDataset(transitions)
        model = train_world_model(cfg, dataset)
        logger = ReasoningHistoryLogger()
        narrator = RLDecisionNarrator(logger)
        init_state = torch.zeros(2)
        policy = lambda state: torch.zeros((), dtype=torch.long)
        rollout_policy(model, policy, init_state, steps=2, narrator=narrator)
        self.assertEqual(len(logger.get_history()), 2)


if __name__ == '__main__':
    unittest.main()
