from __future__ import annotations

import io
import os
import struct
import tarfile
from pathlib import Path
from typing import Iterable

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PrivateFormat,
    PublicFormat,
    NoEncryption,
)
from cryptography.exceptions import InvalidSignature


class SecureDatasetExchange:
    """Encrypt and sign datasets for transfer between nodes."""

    def __init__(
        self,
        key: bytes,
        signing_key: bytes | None = None,
        verify_key: bytes | None = None,
    ) -> None:
        if len(key) not in (16, 24, 32):
            raise ValueError("key must be 16, 24, or 32 bytes")
        self.key = key
        self._signing_key = signing_key
        self._verify_key = verify_key

    # ------------------------------------------------------------------
    def _encrypt(self, data: bytes) -> bytes:
        aes = AESGCM(self.key)
        nonce = os.urandom(12)
        enc = aes.encrypt(nonce, data, None)
        return nonce + enc

    def _decrypt(self, blob: bytes) -> bytes:
        aes = AESGCM(self.key)
        nonce, enc = blob[:12], blob[12:]
        return aes.decrypt(nonce, enc, None)

    def _sign(self, data: bytes) -> bytes:
        if self._signing_key is None:
            raise ValueError("signing key required")
        priv = ed25519.Ed25519PrivateKey.from_private_bytes(self._signing_key)
        return priv.sign(data)

    def _verify(self, data: bytes, signature: bytes) -> None:
        if self._verify_key is None:
            raise ValueError("verify key required")
        pub = ed25519.Ed25519PublicKey.from_public_bytes(self._verify_key)
        try:
            pub.verify(signature, data)
        except InvalidSignature as e:
            raise ValueError("invalid signature") from e

    # ------------------------------------------------------------------
    def push(self, directory: str | Path, out_file: str | Path) -> Path:
        """Package ``directory`` into ``out_file`` with encryption and signature."""
        root = Path(directory)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for p in root.rglob("*"):
                tar.add(p, arcname=str(p.relative_to(root)))
        data = buf.getvalue()
        enc = self._encrypt(data)
        sig = self._sign(enc) if self._signing_key is not None else b""
        with open(out_file, "wb") as fh:
            fh.write(struct.pack("!H", len(sig)))
            if sig:
                fh.write(sig)
            fh.write(enc)
        return Path(out_file)

    def pull(self, package: str | Path, dest_dir: str | Path) -> None:
        """Extract ``package`` into ``dest_dir`` after decryption and verification."""
        dest = Path(dest_dir)
        with open(package, "rb") as fh:
            sig_len = struct.unpack("!H", fh.read(2))[0]
            sig = fh.read(sig_len) if sig_len else b""
            enc = fh.read()
        if sig_len and self._verify_key is not None:
            self._verify(enc, sig)
        data = self._decrypt(enc)
        buf = io.BytesIO(data)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            tar.extractall(dest)

    # ------------------------------------------------------------------
    @staticmethod
    def generate_key() -> bytes:
        return AESGCM.generate_key(bit_length=128)

    @staticmethod
    def generate_signing_key() -> tuple[bytes, bytes]:
        priv = ed25519.Ed25519PrivateKey.generate()
        pub = priv.public_key()
        return (
            priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
            pub.public_bytes(Encoding.Raw, PublicFormat.Raw),
        )


__all__ = ["SecureDatasetExchange"]
