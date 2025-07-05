import unittest
from asi.enclave_runner import EnclaveRunner


def double(arr):
    return [a * 2 for a in arr]


class TestEnclaveRunner(unittest.TestCase):
    def test_run(self):
        runner = EnclaveRunner()
        out = runner.run(double, [1, 2])
        self.assertEqual(out, [2, 4])


if __name__ == "__main__":
    unittest.main()
