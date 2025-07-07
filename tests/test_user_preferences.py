import unittest
import numpy as np
import importlib.machinery
import importlib.util
import types
import sys


class CrossLingualTranslator:
    pass

dummy = types.ModuleType('asi.data_ingest')
dummy.CrossLingualTranslator = CrossLingualTranslator
sys.modules['asi.data_ingest'] = dummy

from asi.user_preferences import UserPreferences
from asi.prompt_optimizer import PromptOptimizer


class TestUserPreferences(unittest.TestCase):
    def test_update_stats(self):
        prefs = UserPreferences(dim=8)
        prefs.update_user_text("u", "hello world", feedback=1.0)
        prefs.update_user_text("u", "bad", feedback=-1.0)
        vec = prefs.get_vector("u")
        pos, neg = prefs.get_stats("u")
        self.assertEqual((pos, neg), (1, 1))
        self.assertEqual(vec.shape, (8,))
        self.assertTrue(vec.any())

    def test_personalized_score(self):
        prefs = UserPreferences(dim=8)
        prefs.update_user_text("u", "hello", feedback=1.0)

        def scorer(p: str) -> float:
            return 0.0

        opt = PromptOptimizer(scorer, "base", user_preferences=prefs, user_id="u")
        s1 = opt._score("hello")
        s2 = opt._score("bye")
        self.assertGreater(s1, s2)

    def test_emotion_history(self):
        prefs = UserPreferences(dim=4, history_size=3)
        for e in ["positive", "negative", "negative", "positive"]:
            prefs.set_emotion("u", e)
        hist = prefs.get_emotion_history("u")
        self.assertEqual(len(hist), 3)
        self.assertEqual(hist[-1], "positive")


if __name__ == "__main__":
    unittest.main()
