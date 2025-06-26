import unittest

from asi.paper_to_code import transpile


class TestPaperToCode(unittest.TestCase):
    def test_basic_transpile(self):
        latex = r"""
        \begin{algorithm}
        \State x = 0
        \For{i in 1..N}{
        \State x = x + i
        }
        \Return{x}
        \end{algorithm}
        """
        expected = "\n".join([
            "x = 0",
            "for i in 1..N:",
            "    x = x + i",
            "return x",
        ])
        self.assertEqual(transpile(latex).strip(), expected)


if __name__ == "__main__":
    unittest.main()
