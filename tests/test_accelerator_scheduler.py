import unittest
import importlib.machinery
import importlib.util
import types
import sys
import time

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg

loader = importlib.machinery.SourceFileLoader('src.schedulers', 'src/schedulers.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = 'src'
sys.modules[loader.name] = mod
loader.exec_module(mod)
AcceleratorScheduler = mod.AcceleratorScheduler


class TestAcceleratorScheduler(unittest.TestCase):
    def test_multi_device(self):
        mod.psutil.cpu_percent = lambda interval=None: 0.0
        if hasattr(mod, 'torch') and mod.torch is not None:
            mod.torch.cuda.is_available = lambda: True
            mod.torch.cuda.memory_allocated = lambda: 0
            class Props:
                total_memory = 1
            mod.torch.cuda.get_device_properties = lambda idx: Props()
        mod._HAS_XLA = True
        class XM:
            @staticmethod
            def get_memory_info(dev):
                return {'kb_total': 1, 'kb_free': 1}
        mod.xm = XM()

        sched = AcceleratorScheduler(check_interval=0.01)
        ran = []
        sched.add(lambda: ran.append('cpu'), 'cpu')
        sched.add(lambda: ran.append('gpu'), 'gpu')
        sched.add(lambda: ran.append('tpu'), 'tpu')
        time.sleep(0.05)
        self.assertEqual(set(ran), {'cpu', 'gpu', 'tpu'})
        util = sched.get_utilization()
        self.assertIsInstance(util, dict)
        self.assertIn('cpu', util)


if __name__ == '__main__':
    unittest.main()
