import unittest
import importlib.machinery
import importlib.util
import types
import sys

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg

loader = importlib.machinery.SourceFileLoader('src.risk_scoreboard', 'src/risk_scoreboard.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
rs = importlib.util.module_from_spec(spec)
rs.__package__ = 'src'
sys.modules['src.risk_scoreboard'] = rs
loader.exec_module(rs)
RiskScoreboard = rs.RiskScoreboard


class TestRiskScoreboard(unittest.TestCase):
    def test_update(self):
        board = RiskScoreboard()
        risk = board.update(1, 10.0, 0.5)
        self.assertAlmostEqual(risk, 1 * 10 + 1.0 - 0.5)
        self.assertIn('risk_score', board.get_metrics())


if __name__ == '__main__':
    unittest.main()
