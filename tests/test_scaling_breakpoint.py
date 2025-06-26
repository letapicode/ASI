import numpy as np
import unittest

from asi.scaling_breakpoint import BreakpointModel, fit_breakpoint

class TestScalingBreakpoint(unittest.TestCase):
    def test_fit_breakpoint_basic(self):
        x = np.array([1, 2, 4, 8, 16], dtype=float)
        logx = np.log10(x)
        slope1, intercept1 = 1.0, 0.5
        slope2, intercept2 = 2.0, 0.1
        y = np.where(x <= 4, intercept1 + slope1 * logx, intercept2 + slope2 * logx)

        model = fit_breakpoint(x, y)

        self.assertAlmostEqual(model.breakpoint, 4.0, places=7)
        self.assertAlmostEqual(model.slope1, slope1, places=7)
        self.assertAlmostEqual(model.intercept1, intercept1, places=7)
        self.assertAlmostEqual(model.slope2, slope2, places=7)
        self.assertAlmostEqual(model.intercept2, intercept2, places=7)

    def test_predict_piecewise(self):
        model = BreakpointModel(breakpoint=4.0, slope1=1.0, intercept1=0.5, slope2=2.0, intercept2=0.1)
        x = np.array([1.0, 10.0])
        preds = model.predict(x)
        expected = np.array([0.5, 0.1 + 2.0 * np.log10(10.0)])
        np.testing.assert_allclose(preds, expected)

    def test_fit_breakpoint_error(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            fit_breakpoint(x, y)

if __name__ == '__main__':
    unittest.main()
