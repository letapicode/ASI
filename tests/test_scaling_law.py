import unittest
import numpy as np

from asi.scaling_law import BreakpointScalingLaw


class TestBreakpointScalingLaw(unittest.TestCase):
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
        loss = 10 ** log_loss

        model = BreakpointScalingLaw(break_compute=break_compute)
        model.fit(compute, loss)

        self.assertAlmostEqual(model.break_compute, break_compute, places=7)
        params = model.params
        np.testing.assert_allclose(params[0][0], slope1, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(params[0][1], intercept1, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(params[1][0], slope2, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(params[1][1], intercept2, rtol=1e-6, atol=1e-6)

        preds = model.predict(compute)
        np.testing.assert_allclose(preds, loss, rtol=1e-5, atol=1e-8)

    def test_predict_before_fit(self):
        model = BreakpointScalingLaw()
        with self.assertRaises(RuntimeError):
            model.predict([1, 10])

    def test_two_breakpoints(self):
        compute = np.logspace(0, 3, 30)
        b1, b2 = 10, 20
        breaks = [compute[b1], compute[b2]]
        slopes = [-0.5, -0.3, -0.1]
        intercepts = [2.0, 1.5, 1.0]
        log_c = np.log10(compute)
        log_loss = np.empty_like(log_c)
        log_loss[:b1] = slopes[0] * log_c[:b1] + intercepts[0]
        log_loss[b1:b2] = slopes[1] * log_c[b1:b2] + intercepts[1]
        log_loss[b2:] = slopes[2] * log_c[b2:] + intercepts[2]
        loss = 10 ** log_loss

        model = BreakpointScalingLaw(break_compute=breaks)
        model.fit(compute, loss)

        self.assertEqual(len(model.params), 3)
        np.testing.assert_allclose(model.break_compute, breaks, rtol=1e-6, atol=1e-6)
        for (s, i), s_exp, i_exp in zip(model.params, slopes, intercepts):
            np.testing.assert_allclose(s, s_exp, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(i, i_exp, rtol=1e-6, atol=1e-6)

        preds = model.predict(compute)
        np.testing.assert_allclose(preds, loss, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
