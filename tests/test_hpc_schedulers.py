import importlib.machinery
import importlib.util
import types
import sys
from unittest.mock import patch
import unittest

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
pkg.__path__ = ['src']

loader = importlib.machinery.SourceFileLoader('asi.hpc_schedulers', 'src/hpc_schedulers.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = mod
loader.exec_module(mod)
submit_job = mod.submit_job
monitor_job = mod.monitor_job
cancel_job = mod.cancel_job


class TestHPCScheduler(unittest.TestCase):
    def test_submit_job_slurm(self):
        with patch('subprocess.run') as run:
            run.return_value = types.SimpleNamespace(returncode=0, stdout='123', stderr='')
            job_id = submit_job(['job.sh'], backend='slurm')
            run.assert_called_with(['sbatch', 'job.sh'], capture_output=True, text=True)
            self.assertEqual(job_id, '123')

    def test_monitor_job_k8s(self):
        with patch('subprocess.run') as run:
            run.return_value = types.SimpleNamespace(returncode=0, stdout='1', stderr='')
            status = monitor_job('job1', backend='kubernetes')
            run.assert_called_with(['kubectl', 'get', 'job', 'job1', '-o', 'jsonpath={.status.succeeded}'], capture_output=True, text=True)
            self.assertEqual(status, '1')

    def test_cancel_job(self):
        with patch('subprocess.run') as run:
            run.return_value = types.SimpleNamespace(returncode=0, stdout='', stderr='')
            cancel_job('55', backend='slurm')
            run.assert_called_with(['scancel', '55'], capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
