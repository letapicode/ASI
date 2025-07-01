import unittest
import asyncio

from asi.eval_harness import (
    parse_modules,
    evaluate_modules,
    evaluate_modules_async,
    log_memory_usage,
)


class TestEvalHarness(unittest.TestCase):
    def test_parse_modules(self):
        mods = parse_modules("docs/Plan.md")
        self.assertIn("moe_router", mods)
        self.assertIn("formal_verifier", mods)

    def test_evaluate_subset(self):
        subset = ["moe_router", "flash_attention3", "scaling_law"]
        results = evaluate_modules(subset)
        for name in subset:
            self.assertIn(name, results)
            self.assertTrue(results[name][0], name)
            self.assertIn("gpu=", results[name][1])

    def test_evaluate_subset_async(self):
        subset = ["moe_router", "flash_attention3"]
        results = asyncio.run(evaluate_modules_async(subset))
        for name in subset:
            self.assertIn(name, results)
            self.assertTrue(results[name][0], name)
            self.assertIn("gpu=", results[name][1])

    def test_log_memory_usage(self):
        mem = log_memory_usage()
        self.assertIsInstance(mem, float)


if __name__ == "__main__":
    unittest.main()
