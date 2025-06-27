import unittest
import random
import importlib.machinery
import importlib.util

loader = importlib.machinery.SourceFileLoader('quantum_hpo', 'src/quantum_hpo.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
quantum_hpo = importlib.util.module_from_spec(spec)
loader.exec_module(quantum_hpo)
amplitude_estimate = quantum_hpo.amplitude_estimate


class TestAmplitudeEstimate(unittest.TestCase):
    def test_half_probability(self):
        random.seed(0)
        oracle = lambda: random.random() < 0.5
        prob = amplitude_estimate(oracle, shots=1000)
        self.assertAlmostEqual(prob, 0.5, delta=0.05)


if __name__ == '__main__':
    unittest.main()
