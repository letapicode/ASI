import unittest
import importlib.machinery
import importlib.util
import types
import sys
import tempfile
from pathlib import Path

pkg = types.ModuleType("src")
pkg.__path__ = ["src"]
pkg.__spec__ = importlib.machinery.ModuleSpec("src", None, is_package=True)
sys.modules["src"] = pkg

loader = importlib.machinery.SourceFileLoader(
    "src.secure_dataset_exchange", "src/secure_dataset_exchange.py"
)
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = "src"
sys.modules[loader.name] = mod
loader.exec_module(mod)
SecureDatasetExchange = mod.SecureDatasetExchange

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PrivateFormat,
    PublicFormat,
    NoEncryption,
)


class TestSecureDatasetExchange(unittest.TestCase):
    def test_roundtrip_with_signature(self):
        priv = ed25519.Ed25519PrivateKey.generate()
        pub = priv.public_key()
        priv_b = priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        pub_b = pub.public_bytes(Encoding.Raw, PublicFormat.Raw)
        key = SecureDatasetExchange.generate_key()

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            out = Path(tmp) / "pkg.bin"
            dst = Path(tmp) / "dst"
            src.mkdir()
            (src / "a.txt").write_text("hello")

            ex = SecureDatasetExchange(key, signing_key=priv_b)
            ex.push(src, out)

            ex2 = SecureDatasetExchange(key, verify_key=pub_b)
            ex2.pull(out, dst)

            self.assertEqual((dst / "a.txt").read_text(), "hello")

    def test_invalid_signature(self):
        priv1 = ed25519.Ed25519PrivateKey.generate()
        priv2 = ed25519.Ed25519PrivateKey.generate()
        pub2 = priv2.public_key()
        key = SecureDatasetExchange.generate_key()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "s"
            src.mkdir()
            (src / "b.txt").write_text("data")
            pkg = Path(tmp) / "p.bin"
            ex = SecureDatasetExchange(
                key,
                signing_key=priv1.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
            )
            ex.push(src, pkg)
            ex_bad = SecureDatasetExchange(
                key,
                verify_key=pub2.public_bytes(Encoding.Raw, PublicFormat.Raw),
            )
            with self.assertRaises(ValueError):
                ex_bad.pull(pkg, Path(tmp) / "o")


if __name__ == "__main__":
    unittest.main()
