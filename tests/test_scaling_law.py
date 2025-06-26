import os
import importlib.util
import unittest
import numpy as np

MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "scaling_breakpoint.py")
spec = importlib.util.spec_from_file_location("scaling_breakpoint", MODULE_PATH)
sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb)
BreakpointModel = sb.BreakpointModel
fit_breakpoint = sb.fit_breakpoint


class TestBreakpointModel(unittest.TestCase):
    def test_fit_and_predict(self):
        compute = np.logspace(0, 2, 20)
        break_idx = 10
        break_compute = compute[break_idx]
        slope1, intercept1 = -0.6, 2.0
        slope2, intercept2 = -0.3, 1.0
        log_c = np.log10(compute)
        log_loss = np.where(
            compute < break_compute,
            slope1 * log_c + intercept1,
            slope2 * log_c + intercept2,
        )
        model = fit_breakpoint(compute, log_loss)

        self.assertAlmostEqual(model.breakpoint, break_compute, places=7)
        np.testing.assert_allclose(model.slope1, slope1, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(model.intercept1, intercept1, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(model.slope2, slope2, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(model.intercept2, intercept2, rtol=1e-6, atol=1e-6)

        preds = model.predict(compute)
        np.testing.assert_allclose(preds, log_loss, rtol=1e-5, atol=1e-8)

    def test_predict_before_fit(self):
        model = BreakpointModel(breakpoint=10.0, slope1=1.0, intercept1=0.0, slope2=1.0, intercept2=0.0)
        preds = model.predict(np.array([1.0, 100.0]))
        self.assertEqual(preds.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
