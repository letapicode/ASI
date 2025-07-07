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


ed = load('asi.emotion_detector', 'src/emotion_detector.py')
detect_emotion = ed.detect_emotion

class CrossLingualTranslator:
    def __init__(self, languages):
        self.languages = list(languages)

    def translate(self, text, lang):
        if lang not in self.languages:
            raise ValueError('unsupported language')
        return f'[{lang}] {text}'

    def translate_all(self, text):
        return {l: self.translate(text, l) for l in self.languages}

dummy = types.ModuleType('asi.data_ingest')
dummy.CrossLingualTranslator = CrossLingualTranslator
sys.modules['asi.data_ingest'] = dummy
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

    def test_multilingual_emotion_handling(self):
        prefs = UserPreferences(dim=8)
        prefs.set_language("u", "es")
        prefs.set_emotion("u", "positive")

        def scorer(_: str) -> float:
            return 0.0

        tr = CrossLingualTranslator(["es"])
        opt = PromptOptimizer(
            scorer,
            "hola",
            user_preferences=prefs,
            user_id="u",
        )
        res = opt.optimize(steps=2, translator=tr)
        self.assertTrue(res.startswith("[es]"))
        self.assertEqual(prefs.get_emotion("u"), detect_emotion(res))

    def test_history_adjustment(self):
        prefs = UserPreferences(dim=4)
        for e in ["negative", "negative", "positive"]:
            prefs.set_emotion("u", e)

        def scorer(_: str) -> float:
            return 0.0

        opt = PromptOptimizer(
            scorer,
            "hi",
            user_preferences=prefs,
            user_id="u",
        )
        res = opt.optimize(steps=1)
        self.assertTrue(res.endswith(":)"))

if __name__ == '__main__':
    unittest.main()
