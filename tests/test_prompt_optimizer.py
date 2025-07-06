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


up = load('asi.user_preferences', 'src/user_preferences.py')
po = load('asi.prompt_optimizer', 'src/prompt_optimizer.py')
PromptOptimizer = po.PromptOptimizer
UserPreferences = up.UserPreferences

class TestPromptOptimizer(unittest.TestCase):
    def test_optimize(self):
        def scorer(p: str) -> float:
            return -len(p)
        opt = PromptOptimizer(scorer, "hello")
        res = opt.optimize(steps=5)
        self.assertIsInstance(res, str)
        self.assertTrue(len(opt.history) >= 1)

    def test_emotion_bias(self):
        prefs = UserPreferences(dim=8)
        prefs.update_user_text("u", "good", feedback=1.0)

        def scorer(_: str) -> float:
            return 0.0

        opt = PromptOptimizer(
            scorer, "hi", user_preferences=prefs, user_id="u"
        )
        pos = opt._score("I love this")
        neg = opt._score("I hate this")
        self.assertGreater(pos, neg)

if __name__ == '__main__':
    unittest.main()
