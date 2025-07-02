import unittest
import numpy as np

from asi.embedding_visualizer import EmbeddingVisualizer


class TestEmbeddingVisualizer(unittest.TestCase):
    def test_reduce_and_html(self):
        emb = np.random.randn(10, 5)
        vis = EmbeddingVisualizer(emb)
        reduced = vis.reduce('tsne', n_iter=250)
        self.assertEqual(reduced.shape, (10, 2))
        html = vis.to_html()
        self.assertIn('<html', html.lower())

    def test_server_lifecycle(self):
        emb = np.random.randn(5, 3)
        vis = EmbeddingVisualizer(emb)
        vis.serve(port=0)  # choose random free port
        self.assertIsNotNone(vis._server)
        vis.stop()
        self.assertIsNone(vis._server)


if __name__ == '__main__':
    unittest.main()

