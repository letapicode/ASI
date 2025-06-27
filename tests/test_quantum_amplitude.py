import os
import sys
import unittest
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from quantum_hpo import amplitude_estimate, amplitude_estimate_bayesian


class TestAmplitudeEstimate(unittest.TestCase):
    def test_raises_on_non_positive_shots(self):
        with self.assertRaises(ValueError):
            amplitude_estimate(lambda: True, shots=0)
        with self.assertRaises(ValueError):
            amplitude_estimate(lambda: True, shots=-5)

    def test_half_probability(self):
        random.seed(0)
        oracle = lambda: random.random() < 0.5
        prob = amplitude_estimate(oracle, shots=1000)
        self.assertAlmostEqual(prob, 0.5, delta=0.05)

    def test_bayesian_matches_frequency(self):
        random.seed(1)
        oracle = lambda: random.random() < 0.3
        est_std = amplitude_estimate(oracle, shots=200)
        est_bayes = amplitude_estimate_bayesian(oracle, shots=200)
        self.assertAlmostEqual(est_bayes, est_std, delta=0.05)


if __name__ == "__main__":
    unittest.main()
