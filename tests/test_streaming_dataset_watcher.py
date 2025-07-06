import importlib.machinery
import importlib.util
import sqlite3
import sys
import tempfile
import types
from pathlib import Path
import unittest

src_pkg = types.ModuleType('src')
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src'] = src_pkg

for name in ['dataset_discovery', 'dataset_summarizer', 'streaming_dataset_watcher']:
    loader = importlib.machinery.SourceFileLoader(f'src.{name}', f'src/{name}.py')
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'src'
    sys.modules[f'src.{name}'] = mod
    loader.exec_module(mod)
    setattr(src_pkg, name, mod)

StreamingDatasetWatcher = sys.modules['src.streaming_dataset_watcher'].StreamingDatasetWatcher


class TestStreamingDatasetWatcher(unittest.TestCase):
    def test_poll_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'ds'
            data.mkdir()
            (data / 'a.txt').write_text('hello world')
            feed = root / 'feed.xml'
            feed.write_text(
                f"<rss><channel><item><title>ds</title><link>{data.as_uri()}</link><license>MIT</license></item></channel></rss>"
            )
            db = root / 'db.sqlite'
            watcher = StreamingDatasetWatcher({feed.as_uri(): 'local'}, db)
            added = watcher.poll_once()
            self.assertEqual(len(added), 1)
            conn = sqlite3.connect(db)
            cur = conn.execute('SELECT name, source, url FROM datasets')
            row = cur.fetchone()
            conn.close()
            self.assertEqual(row[0], 'ds')
            self.assertEqual(row[1], 'local')
            self.assertEqual(row[2], data.as_uri())


if __name__ == '__main__':
    unittest.main()
