import unittest
import tempfile
import subprocess
import sys
import os
from pathlib import Path


class TestTrainInfiniteContext(unittest.TestCase):
    def test_checkpoint_and_memory_created(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "train_infinite_context.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            dataset = data_dir / "tinyshakespeare.txt"
            dataset.write_text("abcdefghij" * 10)
            ckpt_dir = tmp_path / "ckpt"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root)
            result = subprocess.run(
                [sys.executable, str(script), "--epochs", "1", "--checkpoint-dir", str(ckpt_dir)],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            ckpt_step = ckpt_dir / "step1"
            self.assertTrue((ckpt_step / "model.pt").exists())
            self.assertTrue((ckpt_step / "memory" / "compressor.pt").exists())


if __name__ == "__main__":
    unittest.main()
