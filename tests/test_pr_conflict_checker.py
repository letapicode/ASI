import unittest
from unittest.mock import patch
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    'pr_conflict_checker', os.path.join(os.path.dirname(__file__), '..', 'src', 'pr_conflict_checker.py')
)
prcc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prcc)

list_open_prs = prcc.list_open_prs
pr_has_conflict = prcc.pr_has_conflict
summarize_conflicts = prcc.summarize_conflicts


class TestPRConflictChecker(unittest.TestCase):
    def test_summarize_conflicts(self):
        prs = [
            {'number': 1, 'title': 'Fix bug'},
            {'number': 2, 'title': 'Add feature'},
        ]
        conflicts = {1: True, 2: False}
        summary = summarize_conflicts(prs, conflicts)
        self.assertIn('Conflicts in 1/2 PRs', summary)
        self.assertIn('#1 Fix bug: CONFLICT', summary)
        self.assertIn('#2 Add feature: NO CONFLICT', summary)

    def test_check_all_prs(self):
        prs = [
            {'number': 1, 'title': 'Fix bug'},
            {'number': 2, 'title': 'Add feature'},
        ]
        with patch.object(prcc, 'list_open_prs', return_value=prs), \
             patch.object(prcc, 'fetch_pr'), \
             patch.object(prcc.subprocess, 'check_output', return_value=b'base'), \
             patch.object(prcc.subprocess, 'run') as mock_run:
            mock_run.return_value.stdout = '<<<<<\n'
            conflicts = prcc.check_all_prs('owner/repo')
        self.assertEqual(conflicts[1], True)
        self.assertEqual(conflicts[2], True)


if __name__ == '__main__':
    unittest.main()
