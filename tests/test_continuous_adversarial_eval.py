import unittest
import importlib.machinery
import importlib.util
import types
import sys
import tempfile

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
sys.modules['scripts'] = types.ModuleType('scripts')


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = name.rpartition('.')[0]
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

adv_mod = _load('asi.adversarial_robustness', 'src/adversarial_robustness.py')
ce_mod = _load('scripts.continuous_eval', 'scripts/continuous_eval.py')
AdversarialRobustnessScheduler = adv_mod.AdversarialRobustnessScheduler
load_adv_config = ce_mod.load_adv_config


class TestContinuousAdversarialEval(unittest.TestCase):
    def test_scheduler_once(self):
        model = lambda s: float(len(s))
        prompts = [('hi', ['hi', 'h'])]
        sched = AdversarialRobustnessScheduler(model, prompts, interval=0.01)
        score = sched.run_once()
        self.assertEqual(score, 1.0)

    def test_load_config(self):
        with tempfile.NamedTemporaryFile('w+', delete=False) as f:
            f.write('interval: 2\nprompts:\n  - prompt: a\n    candidates: [b]\n')
            name = f.name
        prompts, interval = load_adv_config(name)
        self.assertEqual(interval, 2.0)
        self.assertEqual(prompts, [('a', ['b'])])


if __name__ == '__main__':
    unittest.main()
