import unittest
import tempfile
import os

from importlib.machinery import SourceFileLoader
from importlib.util import spec_from_loader, module_from_spec


def _load(name: str, path: str):
    loader = SourceFileLoader(name, path)
    spec = spec_from_loader(name, loader)
    mod = module_from_spec(spec)
    loader.exec_module(mod)
    return mod


micro_export = _load('micro_export', 'src/micro_export.py')
export_to_tflite_micro = micro_export.export_to_tflite_micro
export_to_microtvm = micro_export.export_to_microtvm


class TestMicroExport(unittest.TestCase):
    def test_conversions_create_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            onnx = os.path.join(tmp, 'model.onnx')
            with open(onnx, 'wb') as f:
                f.write(b'dummy')
            tf_path = os.path.join(tmp, 'model.tflite')
            tvm_path = os.path.join(tmp, 'model.tar')
            export_to_tflite_micro(onnx, tf_path)
            export_to_microtvm(onnx, tvm_path)
            self.assertTrue(os.path.exists(tf_path))
            self.assertTrue(os.path.exists(tvm_path))
            self.assertGreater(os.path.getsize(tf_path), 0)
            self.assertGreater(os.path.getsize(tvm_path), 0)


if __name__ == '__main__':
    unittest.main()

