from __future__ import annotations

import argparse
import io
import os
import struct
import tarfile
import hashlib
import json
from dataclasses import dataclass
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


@dataclass
class DatasetIntegrityProof:
    """Hash-based proof of dataset integrity optionally signed."""

    digest: str
    signature: bytes | None = None

    @classmethod
    def generate(
        cls, data: bytes, *, signing_key: bytes | None = None
    ) -> "DatasetIntegrityProof":
        digest = hashlib.sha256(data).hexdigest()
        sig = None
        if signing_key is not None:
            priv = ed25519.Ed25519PrivateKey.from_private_bytes(signing_key)
            sig = priv.sign(bytes.fromhex(digest))
        return cls(digest, sig)

    def verify(self, data: bytes, *, verify_key: bytes | None = None) -> bool:
        if hashlib.sha256(data).hexdigest() != self.digest:
            return False
        if verify_key is not None:
            if self.signature is None:
                return False
            pub = ed25519.Ed25519PublicKey.from_public_bytes(verify_key)
            try:
                pub.verify(self.signature, bytes.fromhex(self.digest))
            except InvalidSignature:
                return False
        return True

    def to_json(self) -> str:
        return json.dumps(
            {
                "digest": self.digest,
                "signature": self.signature.hex() if self.signature else None,
            }
        )

    @classmethod
    def from_json(cls, text: str) -> "DatasetIntegrityProof":
        obj = json.loads(text)
        sig = bytes.fromhex(obj["signature"]) if obj.get("signature") else None
        return cls(obj["digest"], sig)


class SecureDatasetExchange:
    """Encrypt and sign datasets for transfer between nodes."""

    def __init__(
        self,
        key: bytes,
        signing_key: bytes | None = None,
        verify_key: bytes | None = None,
        *,
        require_proof: bool = False,
    ) -> None:
        if len(key) not in (16, 24, 32):
            raise ValueError("key must be 16, 24, or 32 bytes")
        self.key = key
        self._signing_key = signing_key
        self._verify_key = verify_key
        self.require_proof = require_proof

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
    def push(
        self, directory: str | Path, out_file: str | Path, *, with_proof: bool = False
    ) -> Path | tuple[Path, DatasetIntegrityProof]:
        """Package ``directory`` into ``out_file`` with encryption and signature."""
        root = Path(directory)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for p in root.rglob("*"):
                tar.add(p, arcname=str(p.relative_to(root)))
        data = buf.getvalue()
        proof = DatasetIntegrityProof.generate(data, signing_key=self._signing_key)
        enc = self._encrypt(data)
        sig = self._sign(enc) if self._signing_key is not None else b""
        with open(out_file, "wb") as fh:
            fh.write(struct.pack("!H", len(sig)))
            if sig:
                fh.write(sig)
            fh.write(enc)
        path = Path(out_file)
        return (path, proof) if with_proof else path

    def pull(
        self,
        package: str | Path,
        dest_dir: str | Path,
        *,
        proof: DatasetIntegrityProof | None = None,
    ) -> None:
        """Extract ``package`` into ``dest_dir`` after decryption and verification."""
        dest = Path(dest_dir)
        with open(package, "rb") as fh:
            sig_len = struct.unpack("!H", fh.read(2))[0]
            sig = fh.read(sig_len) if sig_len else b""
            enc = fh.read()
        if sig_len and self._verify_key is not None:
            self._verify(enc, sig)
        data = self._decrypt(enc)
        if self.require_proof:
            if proof is None or not proof.verify(data, verify_key=self._verify_key):
                raise ValueError("invalid dataset proof")
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


__all__ = ["SecureDatasetExchange", "DatasetIntegrityProof"]


def _load_key(hex_str: str | None) -> bytes | None:
    return bytes.fromhex(hex_str) if hex_str else None


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Secure dataset exchange")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_push = sub.add_parser("push", help="Encrypt and sign a dataset")
    p_push.add_argument("directory", help="Dataset directory")
    p_push.add_argument("package", help="Output package path")
    p_push.add_argument("--key", required=True, help="AES key in hex")
    p_push.add_argument("--sign-key", help="Ed25519 private key in hex")
    p_push.add_argument("--proof-out", help="Write integrity proof to file")

    p_pull = sub.add_parser("pull", help="Decrypt and verify a dataset")
    p_pull.add_argument("package", help="Input package path")
    p_pull.add_argument("directory", help="Destination directory")
    p_pull.add_argument("--key", required=True, help="AES key in hex")
    p_pull.add_argument("--verify-key", help="Ed25519 public key in hex")
    p_pull.add_argument("--proof-in", help="Read integrity proof from file")

    args = parser.parse_args(argv)

    key = bytes.fromhex(args.key)
    if args.cmd == "push":
        sign_key = _load_key(args.sign_key)
        ex = SecureDatasetExchange(key, signing_key=sign_key)
        path, proof = ex.push(
            Path(args.directory), Path(args.package), with_proof=bool(args.proof_out)
        )
        if args.proof_out:
            Path(args.proof_out).write_text(proof.to_json())
    else:
        verify_key = _load_key(args.verify_key)
        ex = SecureDatasetExchange(key, verify_key=verify_key, require_proof=bool(args.proof_in))
        proof = None
        if args.proof_in:
            proof = DatasetIntegrityProof.from_json(Path(args.proof_in).read_text())
        ex.pull(Path(args.package), Path(args.directory), proof=proof)


if __name__ == "__main__":
    main()
