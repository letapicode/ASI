import unittest
import importlib.machinery
import importlib.util
import types
import sys
import os

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
sys.modules['asi.fpga_backend'] = types.SimpleNamespace(_HAS_FPGA=False, cl=None)
sys.modules['asi.loihi_backend'] = types.SimpleNamespace(_HAS_LOIHI=False)
sys.modules['asi.analog_backend'] = types.SimpleNamespace(
    _HAS_ANALOG=True,
    analogsim=types.SimpleNamespace()
)


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = name.rpartition('.')[0]
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

hardware_detect = _load('asi.hardware_detect', 'src/hardware_detect.py')


class TestHardwareDetect(unittest.TestCase):
    def tearDown(self):
        os.environ.pop('ASI_ANALOG_DEVICES', None)

    def test_env_override(self):
        os.environ['ASI_ANALOG_DEVICES'] = 'd0,d1'
        self.assertEqual(hardware_detect.list_analog(), ['d0', 'd1'])

    def test_backend_list_devices(self):
        hardware_detect.analog_backend.analogsim = types.SimpleNamespace(
            list_devices=lambda: ['a0', 'a1']
        )
        self.assertEqual(hardware_detect.list_analog(), ['a0', 'a1'])

    def test_device_count(self):
        hardware_detect.analog_backend.analogsim = types.SimpleNamespace(
            device_count=lambda: 2
        )
        self.assertEqual(hardware_detect.list_analog(), ['analog0', 'analog1'])


if __name__ == '__main__':
    unittest.main()
