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


emotion = load('asi.emotion_detector', 'src/emotion_detector.py')
detect_emotion = emotion.detect_emotion

class TestEmotionDetector(unittest.TestCase):
    def test_detect_emotion(self):
        self.assertEqual(detect_emotion('I love this!'), 'positive')
        self.assertEqual(detect_emotion('I hate this!'), 'negative')
        self.assertEqual(detect_emotion('This is a book.'), 'neutral')

if __name__ == '__main__':
    unittest.main()
