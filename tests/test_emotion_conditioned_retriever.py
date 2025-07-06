import importlib.machinery
import importlib.util
import types
import sys
import unittest

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

edet = load('asi.emotion_detector', 'src/emotion_detector.py')
ed = load('asi.emotion_conditioned_retriever', 'src/emotion_conditioned_retriever.py')

EmotionConditionedRetriever = ed.EmotionConditionedRetriever
_embed_text = ed._embed_text

def make_memory(texts):
    vecs = [_embed_text(t, 8) for t in texts]
    class Dummy:
        def __init__(self, vs, ms):
            import numpy as np
            self.vs = np.stack(vs)
            self.meta = list(ms)
        def search(self, q, k=5, return_scores=False):
            import numpy as np
            sims = self.vs @ q / (np.linalg.norm(self.vs, axis=1) * np.linalg.norm(q))
            idx = np.argsort(sims)[::-1][:k]
            vecs = self.vs[idx]
            metas = [self.meta[i] for i in idx]
            if return_scores:
                return vecs, metas, sims[idx].tolist()
            return vecs, metas
    return Dummy(vecs, texts)


class TestEmotionConditionedRetriever(unittest.TestCase):
    def test_rank_by_emotion(self):
        mem = make_memory(['I love it', 'I hate it'])
        retriever = EmotionConditionedRetriever(mem, dim=8)
        vecs, meta = retriever.search_with_emotion('I love this', k=2)
        self.assertEqual(meta[0], 'I love it')
        vecs2, meta2 = retriever.search_with_emotion('I hate everything', k=2)
        self.assertEqual(meta2[0], 'I hate it')


if __name__ == '__main__':
    unittest.main()
