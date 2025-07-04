import unittest
import importlib.machinery
import importlib.util
import types
import sys

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg

loader = importlib.machinery.SourceFileLoader('src.telemetry', 'src/telemetry.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
tel = importlib.util.module_from_spec(spec)
tel.__package__ = 'src'
sys.modules['src.telemetry'] = tel
loader.exec_module(tel)

loader = importlib.machinery.SourceFileLoader('src.resource_broker', 'src/resource_broker.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
rb = importlib.util.module_from_spec(spec)
rb.__package__ = 'src'
sys.modules['src.resource_broker'] = rb
loader.exec_module(rb)
ResourceBroker = rb.ResourceBroker


class TestResourceBroker(unittest.TestCase):
    def test_allocate_and_decide(self):
        broker = ResourceBroker()
        broker.register_cluster('a', 2)
        broker.register_cluster('b', 1)
        c = broker.allocate('job')
        self.assertIn(c, {'a', 'b'})
        decision = broker.scale_decision({'cpu': 90})
        self.assertEqual(decision, 'scale_up')


if __name__ == '__main__':
    unittest.main()
