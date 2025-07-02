import unittest
from asi.data_ingest import CrossLingualTranslator


class TestCrossLingualTranslator(unittest.TestCase):
    def test_translate_all(self):
        tr = CrossLingualTranslator(["en", "fr"])
        res = tr.translate_all("hello")
        self.assertEqual(res["en"], "[en] hello")
        self.assertIn("fr", res)


if __name__ == "__main__":
    unittest.main()
