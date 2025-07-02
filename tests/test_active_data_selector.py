import unittest
import numpy as np
from asi.data_ingest import ActiveDataSelector


class TestActiveDataSelector(unittest.TestCase):
    def test_select(self):
        selector = ActiveDataSelector(threshold=0.5)
        triples = [("t", "i", "a"), ("t2", "i2", "a2")]
        probs = [np.array([0.5, 0.5]), np.array([0.9, 0.1])]
        kept = selector.select(triples, probs)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0][0], "t")


if __name__ == "__main__":
    unittest.main()
