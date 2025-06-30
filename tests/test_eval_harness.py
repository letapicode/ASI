import unittest

from asi import eval_harness


class TestEvalHarness(unittest.TestCase):
    def test_parse_plan_metrics(self):
        metrics = eval_harness.parse_plan_metrics('docs/Plan.md')
        self.assertIn('moe_load_balance_std', metrics)
        self.assertAlmostEqual(metrics['moe_load_balance_std'], 0.02, places=2)

    def test_run_and_summarize(self):
        results = eval_harness.run_evaluations()
        self.assertIn('moe_load_balance_std', results)
        self.assertTrue(results['moe_load_balance_std'].passed)
        summary = eval_harness.summarize_results(results)
        self.assertIn('Passed 1/1 metrics', summary)
        self.assertIn('moe_load_balance_std', summary)


if __name__ == '__main__':
    unittest.main()
