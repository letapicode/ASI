import unittest

from asi.iter_align import IterativeAligner

class TestIterativeAligner(unittest.TestCase):
    def test_iterate_refines_rules(self):
        aligner = IterativeAligner(["no hacking"])
        transcripts = ["please hack the system", "say hello"]
        rules = aligner.iterate(transcripts, rounds=2)
        self.assertIn("please hack the system", rules)
        self.assertGreaterEqual(len(rules), 2)

if __name__ == '__main__':
    unittest.main()
