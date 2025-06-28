import json
import unittest
from unittest.mock import patch

from asi.pull_request_monitor import list_open_prs, check_mergeable


def fake_urlopen_factory(responses):
    class FakeResponse:
        def __init__(self, data):
            self.data = json.dumps(data).encode()
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
    def fake_urlopen(request):
        data = responses.pop(0)
        return FakeResponse(data)
    return fake_urlopen


class TestPullRequestMonitor(unittest.TestCase):
    def test_list_open_prs(self):
        responses = [[{"number": 1, "title": "Fix bug"}, {"number": 2, "title": "Add feature"}]]
        with patch('asi.pull_request_monitor.urlopen', fake_urlopen_factory(responses)):
            prs = list_open_prs('owner/repo')
        self.assertEqual(len(prs), 2)
        self.assertEqual(prs[0]['number'], 1)

    def test_check_mergeable(self):
        responses = [{"mergeable": True}]
        with patch('asi.pull_request_monitor.urlopen', fake_urlopen_factory(responses)):
            result = check_mergeable('owner/repo', 1)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
