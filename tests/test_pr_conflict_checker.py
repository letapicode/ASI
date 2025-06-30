import os
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from asi.pr_conflict_checker import check_pr_conflicts
from autobench import summarize_results


def git(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


class TestPRConflictChecker(unittest.TestCase):
    def setUp(self):
        self.remote = tempfile.mkdtemp()
        git(['git', 'init', '--bare', self.remote], cwd=self.remote)
        self.repo = tempfile.mkdtemp()
        git(['git', 'init'], cwd=self.repo)
        git(['git', 'config', 'user.email', 'a@b.c'], cwd=self.repo)
        git(['git', 'config', 'user.name', 'test'], cwd=self.repo)
        git(['git', 'remote', 'add', 'origin', self.remote], cwd=self.repo)
        with open(os.path.join(self.repo, 'file.txt'), 'w') as f:
            f.write('base\n')
        git(['git', 'add', 'file.txt'], cwd=self.repo)
        git(['git', 'commit', '-m', 'init'], cwd=self.repo)
        git(['git', 'branch', '-M', 'main'], cwd=self.repo)
        git(['git', 'push', 'origin', 'main'], cwd=self.repo)
        self.first_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.repo).decode().strip()
        git(['git', 'checkout', '-b', 'feature_no_conflict'], cwd=self.repo)
        with open(os.path.join(self.repo, 'another.txt'), 'w') as f:
            f.write('hi\n')
        git(['git', 'add', 'another.txt'], cwd=self.repo)
        git(['git', 'commit', '-m', 'feature_no_conflict'], cwd=self.repo)
        git(['git', 'push', 'origin', 'feature_no_conflict:refs/pull/1/head'], cwd=self.repo)
        git(['git', 'checkout', 'main'], cwd=self.repo)
        with open(os.path.join(self.repo, 'file.txt'), 'w') as f:
            f.write('mainline\n')
        git(['git', 'add', 'file.txt'], cwd=self.repo)
        git(['git', 'commit', '-m', 'main_change'], cwd=self.repo)
        git(['git', 'push', 'origin', 'main'], cwd=self.repo)
        git(['git', 'checkout', self.first_commit, '-b', 'feature_conflict'], cwd=self.repo)
        with open(os.path.join(self.repo, 'file.txt'), 'w') as f:
            f.write('pr_branch\n')
        git(['git', 'add', 'file.txt'], cwd=self.repo)
        git(['git', 'commit', '-m', 'feature_conflict'], cwd=self.repo)
        git(['git', 'push', 'origin', 'feature_conflict:refs/pull/2/head'], cwd=self.repo)

    def tearDown(self):
        subprocess.run(['rm', '-rf', self.repo, self.remote])

    def test_check_pr_conflicts(self):
        prs = [{'number': 1, 'title': 'clean'}, {'number': 2, 'title': 'conflict'}]
        with patch('asi.pr_conflict_checker.list_open_prs', return_value=prs):
            results = check_pr_conflicts('dummy/repo', repo_path=self.repo)
        self.assertTrue(results['PR 1'].passed)
        self.assertFalse(results['PR 2'].passed)
        summary = summarize_results(results)
        self.assertIn('Passed 1/2 modules', summary)
        self.assertIn('PR 1: PASS', summary)
        self.assertIn('PR 2: FAIL', summary)


if __name__ == '__main__':
    unittest.main()
