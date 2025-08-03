from __future__ import annotations

import hashlib
import json
from concurrent import futures
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .dataset_watermarker import detect_watermark
from .provenance_ledger import DataProvenanceLedger, BlockchainProvenanceLedger

try:  # pragma: no cover - optional dependency
    import grpc  # type: ignore
    from . import lineage_pb2, lineage_pb2_grpc
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class LineageStep:
    note: str
    inputs: List[str]
    outputs: Dict[str, Dict[str, str | None]]
    fairness_before: Dict[str, float] | None = None
    fairness_after: Dict[str, float] | None = None


class DatasetLineageManager:
    """Record dataset transformations and resulting file hashes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.log_path = self.root / "dataset_lineage.json"
        self.ledger = DataProvenanceLedger(self.root)
        if self.log_path.exists():
            data = json.loads(self.log_path.read_text())
            self.steps: List[LineageStep] = [LineageStep(**d) for d in data]
        else:
            self.steps = []

    def record(
        self,
        inputs: Iterable[str | Path],
        outputs: Iterable[str | Path],
        note: str = "",
        fairness_before: Dict[str, float] | None = None,
        fairness_after: Dict[str, float] | None = None,
    ) -> None:
        out_hashes: Dict[str, Dict[str, str | None]] = {}
        for p in outputs:
            path = Path(p)
            out_hashes[str(path)] = {
                "hash": _hash_file(path),
                "watermark_id": detect_watermark(path),
            }
        step = LineageStep(
            note,
            [str(p) for p in inputs],
            out_hashes,
            fairness_before,
            fairness_after,
        )
        self.steps.append(step)
        self.log_path.write_text(json.dumps([asdict(s) for s in self.steps], indent=2))
        rec = json.dumps(asdict(step), sort_keys=True)
        self.ledger.append(rec)

    def load(self) -> List[LineageStep]:
        """Reload steps from the log file."""
        if self.log_path.exists():
            data = json.loads(self.log_path.read_text())
            self.steps = [LineageStep(**d) for d in data]
        return self.steps


if _HAS_GRPC:

    class DatasetLineageClient:
        """Client for :class:`DatasetLineageServer`."""

        def __init__(self, address: str) -> None:
            self.address = address
            self.channel = grpc.insecure_channel(address)
            self.stub = lineage_pb2_grpc.DatasetLineageServiceStub(self.channel)

        def add_entry(
            self, inputs: Iterable[str | Path], outputs: Iterable[str | Path], note: str = ""
        ) -> None:
            req = lineage_pb2.AddEntryRequest(
                note=note, inputs=[str(p) for p in inputs], outputs=[str(p) for p in outputs]
            )
            self.stub.AddEntry(req)

        def get_entries(self) -> List[lineage_pb2.Entry]:
            reply = self.stub.GetEntries(lineage_pb2.GetEntriesRequest())
            return list(reply.entries)

        def close(self) -> None:
            self.channel.close()

    class DatasetLineageServer(lineage_pb2_grpc.DatasetLineageServiceServicer):
        """gRPC server exposing :class:`DatasetLineageManager`."""

        def __init__(
            self,
            root: str | Path,
            signing_key: bytes,
            address: str = "localhost:50900",
            max_workers: int = 4,
        ) -> None:
            self.manager = DatasetLineageManager(root)
            self.signing_key = signing_key
            self.address = address
            self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
            lineage_pb2_grpc.add_DatasetLineageServiceServicer_to_server(self, self.server)
            self.server.add_insecure_port(address)
            self.verify_key = (
                ed25519.Ed25519PrivateKey.from_private_bytes(signing_key)
                .public_key()
                .public_bytes(Encoding.Raw, PublicFormat.Raw)
            )
            self.manager.ledger = BlockchainProvenanceLedger(root)
            orig_append = self.manager.ledger.append

            def signed_append(record: str, signature: str | None = None) -> None:
                priv = ed25519.Ed25519PrivateKey.from_private_bytes(self.signing_key)
                sig = priv.sign(record.encode()).hex()
                orig_append(record, signature=sig)

            self.manager.ledger.append = signed_append  # type: ignore[assignment]

        def AddEntry(self, request: lineage_pb2.AddEntryRequest, context) -> lineage_pb2.AddEntryReply:  # noqa: N802
            self.manager.record(request.inputs, request.outputs, note=request.note)
            return lineage_pb2.AddEntryReply(ok=True)

        def GetEntries(self, request: lineage_pb2.GetEntriesRequest, context) -> lineage_pb2.GetEntriesReply:  # noqa: N802
            self.manager.load()
            entries = []
            for step in self.manager.steps:
                outs = [
                    lineage_pb2.OutputEntry(
                        path=p,
                        hash=info.get("hash", ""),
                        watermark_id="" if info.get("watermark_id") is None else str(info.get("watermark_id")),
                    )
                    for p, info in step.outputs.items()
                ]
                entries.append(lineage_pb2.Entry(note=step.note, inputs=step.inputs, outputs=outs))
            return lineage_pb2.GetEntriesReply(entries=entries)

        def start(self) -> None:
            self.server.start()

        def stop(self, grace: float = 0) -> None:
            self.server.stop(grace)

else:  # pragma: no cover - optional dependency

    class DatasetLineageClient:  # type: ignore[dead-code]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ImportError("grpcio is required for DatasetLineageClient")

    class DatasetLineageServer:  # type: ignore[dead-code]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ImportError("grpcio is required for DatasetLineageServer")


__all__ = [
    "DatasetLineageManager",
    "LineageStep",
    "DatasetLineageClient",
    "DatasetLineageServer",
]
