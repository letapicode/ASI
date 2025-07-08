import importlib.machinery
import importlib.util
import sys
import types
from unittest.mock import patch
import unittest

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
pkg.__path__ = ['src']


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'asi'
    sys.modules[name] = mod
    loader.exec_module(mod)
    setattr(pkg, name.split('.')[-1], mod)
    return mod

di = load('asi.data_ingest', 'src/data_ingest.py')
GraphOfThought = load('asi.graph_of_thought', 'src/graph_of_thought.py').GraphOfThought
clt = load('asi.cross_lingual_voice_chat', 'src/cross_lingual_voice_chat.py')

CrossLingualVoiceChat = clt.CrossLingualVoiceChat
CrossLingualTranslator = di.CrossLingualTranslator
CrossLingualSpeechTranslator = di.CrossLingualSpeechTranslator


class TestCrossLingualVoiceChat(unittest.TestCase):
    def test_chat(self):
        tr = CrossLingualTranslator(['es'])
        speech = CrossLingualSpeechTranslator(tr)
        chat = CrossLingualVoiceChat(speech, GraphOfThought())
        with patch.object(speech, 'transcribe', return_value='hello'):
            res = chat.chat('dummy.wav')
        self.assertIn('text', res)
        self.assertTrue(chat.graph.nodes)


if __name__ == '__main__':
    unittest.main()
