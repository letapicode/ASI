import unittest
import importlib.machinery
import importlib.util
import types
import sys
import numpy as np

# create minimal asi package
pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = name.rpartition('.')[0]
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

VectorStore = _load('asi.vector_store', 'src/vector_store.py').VectorStore
_load('memory_pb2', 'src/memory_pb2.py')
_load('asi.memory_pb2', 'src/memory_pb2.py')
_load('fhe_memory_pb2', 'src/fhe_memory_pb2.py')
_load('asi.fhe_memory_pb2', 'src/fhe_memory_pb2.py')
_load('fhe_memory_pb2_grpc', 'src/fhe_memory_pb2_grpc.py')
_load('asi.fhe_memory_pb2_grpc', 'src/fhe_memory_pb2_grpc.py')
fhe_mod = _load('asi.fhe_memory_server', 'src/fhe_memory_server.py')
FHEMemoryServer = getattr(fhe_mod, 'FHEMemoryServer', None)
FHEMemoryClient = getattr(fhe_mod, 'FHEMemoryClient', None)

try:
    import grpc  # noqa: F401
    import tenseal as ts
    _HAVE_DEPS = True
except Exception:  # pragma: no cover - optional
    _HAVE_DEPS = False


class TestFHEMemoryServer(unittest.TestCase):
    def test_encrypted_roundtrip(self):
        if not _HAVE_DEPS or FHEMemoryServer is None:
            self.skipTest('grpcio or tenseal not available')

        store = VectorStore(dim=2)
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        ctx.generate_galois_keys()
        ctx.global_scale = 2**40

        server = FHEMemoryServer(store, ctx, 'localhost:50910')
        server.start()

        client = FHEMemoryClient('localhost:50910', ctx)
        vec = np.array([1.0, 0.0], dtype=np.float32)
        client.add(vec, metadata='a')
        out, meta = client.search(vec, k=1)
        client.close()
        server.stop(0)

        exp_vec, exp_meta = store.search(vec, k=1)
        np.testing.assert_allclose(out, exp_vec, atol=1e-5)
        self.assertEqual(meta, ['a'])


if __name__ == '__main__':
    unittest.main()
