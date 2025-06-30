import json
import unittest
from unittest.mock import patch

import asyncio
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "pull_request_monitor",
    os.path.join(os.path.dirname(__file__), "..", "src", "pull_request_monitor.py"),
)
prmon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prmon)

list_open_prs = prmon.list_open_prs
check_mergeable = prmon.check_mergeable
list_open_prs_async = prmon.list_open_prs_async
check_mergeable_async = prmon.check_mergeable_async


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


def fake_async_api_factory(responses, calls=None):
    async def fake_api(path, token=None, session=None):
        if calls is not None:
            calls.append(session)
        return responses.pop(0)
    return fake_api


class TestPullRequestMonitor(unittest.TestCase):
    def test_list_open_prs(self):
        responses = [[{"number": 1, "title": "Fix bug"}, {"number": 2, "title": "Add feature"}]]
        with patch.object(prmon, 'urlopen', fake_urlopen_factory(responses)):
            prs = list_open_prs('owner/repo')
        self.assertEqual(len(prs), 2)
        self.assertEqual(prs[0]['number'], 1)

    def test_check_mergeable(self):
        responses = [{"mergeable": True}]
        with patch.object(prmon, 'urlopen', fake_urlopen_factory(responses)):
            result = check_mergeable('owner/repo', 1)
        self.assertTrue(result)

    def test_list_open_prs_async(self):
        responses = [[{"number": 1, "title": "Bug"}, {"number": 2, "title": "Feature"}]]
        calls = []
        session = object()
        with patch.object(prmon, '_github_api_async', fake_async_api_factory(responses, calls)):
            prs = asyncio.run(list_open_prs_async('owner/repo', session=session))
        self.assertEqual(len(prs), 2)
        self.assertEqual(prs[1]['title'], 'Feature')
        self.assertIs(calls[0], session)

    def test_check_mergeable_async(self):
        responses = [{"mergeable": False}]
        calls = []
        session = object()
        with patch.object(prmon, '_github_api_async', fake_async_api_factory(responses, calls)):
            result = asyncio.run(check_mergeable_async('owner/repo', 1, session=session))
        self.assertFalse(result)
        self.assertIs(calls[0], session)


if __name__ == '__main__':
    unittest.main()
