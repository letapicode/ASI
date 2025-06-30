import unittest
import numpy as np

from asi.embodied_calibration import calibrate_sensors, adjust_actions


class TestEmbodiedCalibration(unittest.TestCase):
    def test_calibrate(self):
        sim = [np.ones((2, 2))]
        real = [np.full((2, 2), 2.0)]
        scale, offset = calibrate_sensors(sim, real)
        self.assertEqual(scale.shape[0], 2)
        self.assertEqual(offset.shape[0], 2)

    def test_adjust_actions(self):
        sim = [np.ones((2, 2))]
        real = [np.full((2, 2), 3.0)]
        scale, offset = adjust_actions(sim, real)
        self.assertEqual(scale.shape[0], 2)
        self.assertEqual(offset.shape[0], 2)


if __name__ == '__main__':
    unittest.main()
