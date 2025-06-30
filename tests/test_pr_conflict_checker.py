import unittest
import importlib.util
import importlib.machinery
import types
from unittest.mock import patch
import io
import sys

pkg = types.ModuleType('asi')
pkg.__path__ = ['src']
sys.modules.setdefault('asi', pkg)
spec = importlib.util.spec_from_file_location('asi.pr_conflict_checker', 'src/pr_conflict_checker.py')
pr_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr_check)

BenchResult = pr_check.BenchResult
check_pr_conflicts = pr_check.check_pr_conflicts


def fake_list_open_prs(repo, token=None):
    return [
        {"number": 1, "title": "Bug"},
        {"number": 2, "title": "Feature"},
    ]


def fake_run(cmd, capture_output=True, text=True):
    class Res:
        def __init__(self, code=0, out="", err=""):
            self.returncode = code
            self.stdout = out
            self.stderr = err
    if cmd[1] == "fetch":
        return Res()
    if cmd[1] == "merge-tree":
        if cmd[-1] == "pr/1":
            return Res(0, "<<<<<<< HEAD\nconflict\n=======\n>>>>>>\n")
        return Res()
    return Res()


def fake_check_output(cmd, text=True):
    return "base\n"


class TestPRConflictChecker(unittest.TestCase):
    def test_check_pr_conflicts(self):
        with patch.object(pr_check, 'list_open_prs', fake_list_open_prs), \
             patch.object(pr_check.subprocess, 'run', fake_run), \
             patch.object(pr_check.subprocess, 'check_output', fake_check_output):
            results = check_pr_conflicts('owner/repo')
        self.assertIn('#1 Bug', results)
        self.assertFalse(results['#1 Bug'].passed)
        self.assertIn('#2 Feature', results)
        self.assertTrue(results['#2 Feature'].passed)

    def test_main_outputs_summary(self):
        with patch.object(pr_check, 'list_open_prs', fake_list_open_prs), \
             patch.object(pr_check.subprocess, 'run', fake_run), \
             patch.object(pr_check.subprocess, 'check_output', fake_check_output):
            buf = io.StringIO()
            with patch.object(sys, 'stdout', buf), \
                 patch.object(sys, 'argv', ['pr_conflict_checker', 'owner/repo']):
                pr_check.main()
            out = buf.getvalue()
        self.assertIn('Passed 1/2 modules', out)
        self.assertIn('#1 Bug', out)


if __name__ == '__main__':
    unittest.main()
