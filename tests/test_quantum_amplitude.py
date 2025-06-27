import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from quantum_hpo import amplitude_estimate


class TestAmplitudeEstimate(unittest.TestCase):
    def test_raises_on_non_positive_shots(self):
        with self.assertRaises(ValueError):
            amplitude_estimate(lambda: True, shots=0)
        with self.assertRaises(ValueError):
            amplitude_estimate(lambda: True, shots=-5)


if __name__ == "__main__":
    unittest.main()
