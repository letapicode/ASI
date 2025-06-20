import unittest

from src.deliberative_alignment import DeliberativeAligner


class TestDeliberativeAligner(unittest.TestCase):
    def test_check_and_analyze(self):
        policy = "no hacking\nno violence"
        aligner = DeliberativeAligner(policy)
        self.assertTrue(aligner.check(["say hello", "all good"]))
        self.assertFalse(aligner.check(["begin hacking sequence", "all good"]))
        text = "first step\ncommit violence"
        self.assertFalse(aligner.analyze(text))


if __name__ == '__main__':
    unittest.main()
