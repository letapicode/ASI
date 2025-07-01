import unittest
import subprocess
import tempfile
import sys
import os
import importlib.machinery
import importlib.util

loader = importlib.machinery.SourceFileLoader(
    'meta_rl_refactor', 'src/meta_rl_refactor.py'
)
spec = importlib.util.spec_from_loader(loader.name, loader)
meta_rl_refactor = importlib.util.module_from_spec(spec)
loader.exec_module(meta_rl_refactor)
MetaRLRefactorAgent = meta_rl_refactor.MetaRLRefactorAgent
QAEHyperparamSearch = meta_rl_refactor.QAEHyperparamSearch



class TestMetaRLRefactorAgent(unittest.TestCase):
    def test_update_and_select(self):
        agent = MetaRLRefactorAgent(epsilon=0.0)
        s1 = "s1"
        s2 = "s2"
        agent.update(s1, "replace", 1.0, s2)
        self.assertIn((s1, "replace"), agent.q)
        self.assertGreater(agent.q[(s1, "replace")], 0.0)
        action = agent.select_action(s1)
        self.assertEqual(action, "replace")

    def test_tune_epsilon(self):
        calls = []

        def eval_func(eps: float) -> bool:
            calls.append(eps)
            return eps > 0.2

        agent = MetaRLRefactorAgent(epsilon=0.5)
        best, prob = agent.tune_epsilon([0.1, 0.3, 0.5], eval_func, shots=5)
        self.assertIn(best, [0.3, 0.5])
        self.assertGreaterEqual(prob, 0.0)
        self.assertEqual(agent.epsilon, best)

    def test_train_with_searcher(self):
        class DummySearch:
            def __init__(self) -> None:
                self.calls = 0
                self.eval_func = lambda x: True
                self.param_space = []

            def search(self, *args, **kwargs):
                self.calls += 1
                return 0.5, 1.0

        searcher = DummySearch()
        agent = MetaRLRefactorAgent(searcher=searcher)
        entries = [("replace", 1.0)]
        agent.train(entries, epsilon_space=[0.1, 0.5], eval_func=lambda e: True)
        self.assertEqual(searcher.calls, 1)
        self.assertEqual(agent.epsilon, 0.5)


class TestMetaRLRefactorCLI(unittest.TestCase):
    def test_cli_runs(self):
        data = "\n".join([
            "replace,1",
            "refactor,0",
            "rollback,-1",
        ])
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(data)
            fname = f.name
        try:
            proc = subprocess.run(
                [sys.executable, "src/meta_rl_refactor.py", fname],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0)
            self.assertIn("Best action", proc.stdout)
        finally:
            os.unlink(fname)


if __name__ == "__main__":
    unittest.main()
