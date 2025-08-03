from __future__ import annotations

import io
import os
import json
import tarfile
import hashlib
from pathlib import Path
from typing import Iterable, List, Dict

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature

from .provenance_ledger import BlockchainProvenanceLedger


class InMemoryDHT:
    """Very small DHT implementation backed by a dictionary."""

    def __init__(self) -> None:
        self.store: Dict[str, bytes] = {}

    def put(self, key: str, value: bytes) -> None:
        self.store[key] = value

    def get(self, key: str) -> bytes | None:
        return self.store.get(key)


class FileDHT:
    """Store DHT keys as files under a directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, value: bytes) -> None:
        with open(self.root / key, "wb") as fh:
            fh.write(value)

    def get(self, key: str) -> bytes | None:
        p = self.root / key
        return p.read_bytes() if p.exists() else None


class P2PDatasetExchange:
    """Share encrypted dataset chunks via a simple DHT."""

    def __init__(
        self,
        root: str | Path,
        dht: InMemoryDHT | FileDHT,
        key: bytes,
        *,
        signing_key: bytes | None = None,
        verify_key: bytes | None = None,
    ) -> None:
        if len(key) not in (16, 24, 32):
            raise ValueError("key must be 16, 24, or 32 bytes")
        self.root = Path(root)
        self.dht = dht
        self.key = key
        self.signing_key = signing_key
        self.verify_key = verify_key
        self.ledger = BlockchainProvenanceLedger(self.root)
        self.meta_path = self.root / "dataset_metadata.json"
        if self.meta_path.exists():
            self.metadata: List[Dict[str, object]] = json.loads(self.meta_path.read_text())
        else:
            self.metadata = []

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
        if self.signing_key is None:
            raise ValueError("signing key required")
        priv = ed25519.Ed25519PrivateKey.from_private_bytes(self.signing_key)
        return priv.sign(data)

    def _verify(self, data: bytes, sig: bytes) -> None:
        if self.verify_key is None:
            raise ValueError("verify key required")
        pub = ed25519.Ed25519PublicKey.from_public_bytes(self.verify_key)
        try:
            pub.verify(sig, data)
        except InvalidSignature as e:
            raise ValueError("invalid signature") from e

    # ------------------------------------------------------------------
    def push(self, directory: str | Path, dataset_id: str, *, chunk_size: int = 1_048_576) -> None:
        """Encrypt ``directory`` and store it in the DHT as chunks."""
        root = Path(directory)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for p in root.rglob("*"):
                tar.add(p, arcname=str(p.relative_to(root)))
        enc = self._encrypt(buf.getvalue())

        chunks: List[str] = []
        for i in range(0, len(enc), chunk_size):
            part = enc[i : i + chunk_size]
            h = hashlib.sha256(part).hexdigest()
            self.dht.put(h, part)
            chunks.append(h)

        meta = {"id": dataset_id, "chunks": chunks}
        meta_str = json.dumps(meta, sort_keys=True)
        self.metadata.append(meta)
        self.meta_path.write_text(json.dumps(self.metadata, indent=2))
        sig_hex = None
        if self.signing_key is not None:
            sig_hex = self._sign(meta_str.encode()).hex()
        self.ledger.append(meta_str, signature=sig_hex)
        self.dht.put(f"{dataset_id}:meta", meta_str.encode())

    def seed(self) -> None:  # pragma: no cover - optional server
        """Placeholder for starting a network seed."""
        pass

    def pull(self, dataset_id: str, dest_dir: str | Path) -> None:
        """Retrieve ``dataset_id`` from the DHT and extract to ``dest_dir``."""
        if self.meta_path.exists():
            self.metadata = json.loads(self.meta_path.read_text())
        records = [json.dumps(m, sort_keys=True) for m in self.metadata]
        if not self.ledger.verify(records):
            raise ValueError("ledger verification failed")
        idx = next((i for i, m in enumerate(self.metadata) if m["id"] == dataset_id), None)
        if idx is None:
            raise ValueError("dataset not found")
        meta = self.metadata[idx]
        entry = self.ledger.entries[idx]
        sig_hex = entry.get("sig")
        if sig_hex and self.verify_key is not None:
            self._verify(records[idx].encode(), bytes.fromhex(sig_hex))

        parts = bytearray()
        for h in meta["chunks"]:
            chunk = self.dht.get(h)
            if chunk is None:
                raise ValueError(f"missing chunk {h}")
            if hashlib.sha256(chunk).hexdigest() != h:
                raise ValueError("chunk hash mismatch")
            parts.extend(chunk)
        data = self._decrypt(bytes(parts))
        buf = io.BytesIO(data)
        dest = Path(dest_dir)
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


__all__ = ["P2PDatasetExchange", "InMemoryDHT", "FileDHT"]
