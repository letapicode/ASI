import unittest
import importlib.machinery
import importlib.util
import sys
import types
from unittest.mock import patch

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

di = load('asi.data_ingest', 'src/data_ingest.py')
clm = load('asi.cross_lingual_memory', 'src/cross_lingual_memory.py')

CrossLingualMemory = clm.CrossLingualMemory
CrossLingualTranslator = di.CrossLingualTranslator
CrossLingualSpeechTranslator = di.CrossLingualSpeechTranslator


class TestCrossLingualSpeech(unittest.TestCase):
    def test_audio_query(self):
        tr = CrossLingualTranslator(["es"])
        speech = CrossLingualSpeechTranslator(tr)
        mem = CrossLingualMemory(dim=4, compressed_dim=2, capacity=10,
                                 translator=tr, speech_translator=speech)
        mem.add("hello")
        with patch.object(speech, 'transcribe', return_value='hello'):
            vecs, meta = mem.search('dummy.wav', k=1)
        self.assertEqual(len(meta), 1)
        self.assertIn(meta[0], ["hello", "[es] hello"])


if __name__ == '__main__':
    unittest.main()
