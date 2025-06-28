import unittest
import importlib.machinery
import importlib.util

loader = importlib.machinery.SourceFileLoader('paper_to_code', 'src/paper_to_code.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
paper_to_code = importlib.util.module_from_spec(spec)
loader.exec_module(paper_to_code)
transpile = paper_to_code.transpile


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

    def test_function_and_math(self):
        latex = r"""
        \begin{algorithm}
        \Function{Add}{a, b}
        \State result = \(a + b\)
        \Return{result}
        \EndFunction
        \end{algorithm}
        """
        expected = "\n".join([
            "def Add(a, b):",
            "    result = a + b",
            "    return result",
        ])
        self.assertEqual(transpile(latex).strip(), expected)

    def test_repeat_until(self):
        latex = r"""
        \begin{algorithm}
        \State x = 0
        \Repeat
        \State x = x + 1
        \Until{x > 3}
        \end{algorithm}
        """
        expected = "\n".join([
            "x = 0",
            "while True:",
            "    x = x + 1",
            "    if x > 3:",
            "        break",
        ])
        self.assertEqual(transpile(latex).strip(), expected)


if __name__ == "__main__":
    unittest.main()
