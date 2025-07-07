from __future__ import annotations
from concurrent import futures
from pathlib import Path

from .dataset_lineage_manager import DatasetLineageManager
from .blockchain_provenance_ledger import BlockchainProvenanceLedger

try:
    import grpc  # type: ignore
    from . import lineage_pb2, lineage_pb2_grpc
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PublicFormat,
    )
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False


if _HAS_GRPC:

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

            self.manager.ledger.append = signed_append  # type: ignore

        # --------------------------------------------------
        def AddEntry(self, request: lineage_pb2.AddEntryRequest, context) -> lineage_pb2.AddEntryReply:  # noqa: N802
            self.manager.record(request.inputs, request.outputs, note=request.note)
            return lineage_pb2.AddEntryReply(ok=True)

        def GetEntries(
            self, request: lineage_pb2.GetEntriesRequest, context
        ) -> lineage_pb2.GetEntriesReply:  # noqa: N802
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
                entries.append(
                    lineage_pb2.Entry(note=step.note, inputs=step.inputs, outputs=outs)
                )
            return lineage_pb2.GetEntriesReply(entries=entries)

        def start(self) -> None:
            self.server.start()

        def stop(self, grace: float = 0) -> None:
            self.server.stop(grace)


    __all__ = ["DatasetLineageServer"]
else:  # pragma: no cover - optional dependency
    DatasetLineageServer = None  # type: ignore
    __all__ = ["DatasetLineageServer"]
