import os
import sys
import tempfile
import unittest
import json
from pathlib import Path
from dataclasses import asdict
import importlib.machinery
import importlib.util
import types

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PrivateFormat,
    PublicFormat,
    NoEncryption,
)

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")

src_pkg = types.ModuleType("src")
src_pkg.__path__ = [SRC_DIR]
src_pkg.__spec__ = importlib.machinery.ModuleSpec("src", None, is_package=True)
sys.modules["src"] = src_pkg

def _load(name: str, path: str):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "src"
    sys.modules[name] = mod
    short = name.split(".")[-1]
    sys.modules[short] = mod
    loader.exec_module(mod)
    return mod

_load("src.lineage_pb2", os.path.join(SRC_DIR, "lineage_pb2.py"))
_load("src.lineage_pb2_grpc", os.path.join(SRC_DIR, "lineage_pb2_grpc.py"))

dl_mod = _load(
    "src.dataset_lineage", os.path.join(SRC_DIR, "dataset_lineage.py")
)
DatasetLineageClient = dl_mod.DatasetLineageClient
DatasetLineageManager = dl_mod.DatasetLineageManager
BlockchainProvenanceLedger = _load(
    "src.blockchain_provenance_ledger", os.path.join(SRC_DIR, "blockchain_provenance_ledger.py")
).BlockchainProvenanceLedger

DatasetLineageServer = dl_mod.DatasetLineageServer


class TestDatasetLineageServer(unittest.TestCase):
    def test_signature_and_convergence(self):
        try:
            import grpc  # noqa: F401
        except Exception:
            self.skipTest("grpcio not available")

        priv = ed25519.Ed25519PrivateKey.generate()
        pub = priv.public_key()
        priv_b = priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        pub_b = pub.public_bytes(Encoding.Raw, PublicFormat.Raw)

        with tempfile.TemporaryDirectory() as tmp:
            root1 = Path(tmp) / "s1"
            root1.mkdir()
            root2 = Path(tmp) / "s2"
            root2.mkdir()

            server = DatasetLineageServer(root1, priv_b, address="localhost:50910")
            server.start()

            client = DatasetLineageClient("localhost:50910")
            inp = root1 / "inp.txt"
            outp = root1 / "out.txt"
            inp.write_text("a")
            outp.write_text("b")
            client.add_entry([inp], [outp], note="step")

            client.close()
            server.stop(0)

            # verify signature
            records = [json.dumps(asdict(s), sort_keys=True) for s in server.manager.steps]
            ledger = server.manager.ledger
            pub.verify(bytes.fromhex(ledger.entries[0]["sig"]), records[0].encode())

            # replicate ledger
            mgr2 = DatasetLineageManager(root2)
            mgr2.ledger = BlockchainProvenanceLedger(root2)
            for rec, entry in zip(records, ledger.entries):
                mgr2.ledger.append(rec, signature=entry.get("sig"))
            self.assertEqual(ledger.entries, mgr2.ledger.entries)


if __name__ == "__main__":
    unittest.main()
