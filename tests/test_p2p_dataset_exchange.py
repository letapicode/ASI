import unittest
import importlib.machinery
import importlib.util
import types
import sys
import tempfile
import json
from pathlib import Path

pkg = types.ModuleType("src")
pkg.__path__ = ["src"]
pkg.__spec__ = importlib.machinery.ModuleSpec("src", None, is_package=True)
sys.modules["src"] = pkg

loader = importlib.machinery.SourceFileLoader(
    "src.p2p_dataset_exchange", "src/p2p_dataset_exchange.py"
)
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = "src"
sys.modules[loader.name] = mod
loader.exec_module(mod)
P2PDatasetExchange = mod.P2PDatasetExchange
InMemoryDHT = mod.InMemoryDHT

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PrivateFormat,
    PublicFormat,
    NoEncryption,
)


class TestP2PDatasetExchange(unittest.TestCase):
    def test_chunk_transfer_and_verify(self):
        priv = ed25519.Ed25519PrivateKey.generate()
        pub = priv.public_key()
        priv_b = priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        pub_b = pub.public_bytes(Encoding.Raw, PublicFormat.Raw)
        key = P2PDatasetExchange.generate_key()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            dht = InMemoryDHT()
            src = Path(tmp) / "src"
            src.mkdir()
            (src / "a.txt").write_text("hello")

            ex_push = P2PDatasetExchange(root, dht, key, signing_key=priv_b)
            ex_push.push(src, "ds", chunk_size=32)

            dst = Path(tmp) / "dst"
            ex_pull = P2PDatasetExchange(root, dht, key, verify_key=pub_b)
            ex_pull.pull("ds", dst)

            self.assertEqual((dst / "a.txt").read_text(), "hello")

    def test_detect_tampered_chunk(self):
        priv = ed25519.Ed25519PrivateKey.generate()
        pub = priv.public_key()
        priv_b = priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        pub_b = pub.public_bytes(Encoding.Raw, PublicFormat.Raw)
        key = P2PDatasetExchange.generate_key()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            dht = InMemoryDHT()
            src = Path(tmp) / "src"
            src.mkdir()
            (src / "b.txt").write_text("data")

            ex_push = P2PDatasetExchange(root, dht, key, signing_key=priv_b)
            ex_push.push(src, "ds2", chunk_size=16)

            meta = json.loads((root / "dataset_metadata.json").read_text())[0]
            bad_hash = meta["chunks"][0]
            dht.store[bad_hash] = b"bad" + dht.store[bad_hash][3:]

            dst = Path(tmp) / "dst"
            ex_pull = P2PDatasetExchange(root, dht, key, verify_key=pub_b)
            with self.assertRaises(ValueError):
                ex_pull.pull("ds2", dst)


if __name__ == "__main__":
    unittest.main()
