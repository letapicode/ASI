import os
import tempfile
import subprocess
from unittest.mock import patch
import unittest

import importlib.machinery
import importlib.util
import types
import sys

loader2 = importlib.machinery.SourceFileLoader('asi.autobench', 'src/autobench.py')
spec2 = importlib.util.spec_from_loader(loader2.name, loader2)
ab = importlib.util.module_from_spec(spec2)
sys.modules['asi.autobench'] = ab
loader2.exec_module(ab)
summarize_results = ab.summarize_results

loader_pm = importlib.machinery.SourceFileLoader('asi.pull_request_monitor', 'src/pull_request_monitor.py')
spec_pm = importlib.util.spec_from_loader(loader_pm.name, loader_pm)
pm = importlib.util.module_from_spec(spec_pm)
sys.modules['asi.pull_request_monitor'] = pm
loader_pm.exec_module(pm)

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

loader = importlib.machinery.SourceFileLoader('asi.pr_conflict_checker', 'src/pr_conflict_checker.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
prcc = importlib.util.module_from_spec(spec)
sys.modules['asi.pr_conflict_checker'] = prcc
loader.exec_module(prcc)
check_pr_conflicts = prcc.check_pr_conflicts


def git(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


class TestPRConflictChecker(unittest.TestCase):
    def setUp(self):
        self.remote = tempfile.mkdtemp()
        git(["git", "init", "--bare", self.remote], cwd=self.remote)
        self.repo = tempfile.mkdtemp()
        git(["git", "init"], cwd=self.repo)
        git(["git", "config", "user.email", "a@b.c"], cwd=self.repo)
        git(["git", "config", "user.name", "test"], cwd=self.repo)
        git(["git", "remote", "add", "origin", self.remote], cwd=self.repo)
        (Path := os.path.join)
        with open(Path(self.repo, "file.txt"), "w") as f:
            f.write("base\n")
        git(["git", "add", "file.txt"], cwd=self.repo)
        git(["git", "commit", "-m", "init"], cwd=self.repo)
        git(["git", "branch", "-M", "main"], cwd=self.repo)
        git(["git", "push", "origin", "main"], cwd=self.repo)
        self.first_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.repo).decode().strip()
        git(["git", "checkout", "-b", "feature_clean"], cwd=self.repo)
        with open(Path(self.repo, "another.txt"), "w") as f:
            f.write("hi\n")
        git(["git", "add", "another.txt"], cwd=self.repo)
        git(["git", "commit", "-m", "clean"], cwd=self.repo)
        git(["git", "push", "origin", "feature_clean:refs/pull/1/head"], cwd=self.repo)
        git(["git", "checkout", "main"], cwd=self.repo)
        with open(Path(self.repo, "file.txt"), "w") as f:
            f.write("mainline\n")
        git(["git", "add", "file.txt"], cwd=self.repo)
        git(["git", "commit", "-m", "main"], cwd=self.repo)
        git(["git", "push", "origin", "main"], cwd=self.repo)
        git(["git", "checkout", self.first_commit, "-b", "feature_conflict"], cwd=self.repo)
        with open(Path(self.repo, "file.txt"), "w") as f:
            f.write("branch\n")
        git(["git", "add", "file.txt"], cwd=self.repo)
        git(["git", "commit", "-m", "conflict"], cwd=self.repo)
        git(["git", "push", "origin", "feature_conflict:refs/pull/2/head"], cwd=self.repo)

    def tearDown(self):
        subprocess.run(["rm", "-rf", self.repo, self.remote])

    def test_check_pr_conflicts(self):
        prs = [{"number": 1, "title": "clean"}, {"number": 2, "title": "conflict"}]
        with patch("asi.pr_conflict_checker.list_open_prs", return_value=prs):
            results = check_pr_conflicts("dummy/repo", repo_path=self.repo)
        self.assertTrue(results["PR 1"].passed)
        self.assertFalse(results["PR 2"].passed)
        summary = summarize_results(results)
        self.assertIn("Passed 1/2 modules", summary)
        self.assertIn("PR 1: PASS", summary)
        self.assertIn("PR 2: FAIL", summary)


if __name__ == "__main__":
    unittest.main()
