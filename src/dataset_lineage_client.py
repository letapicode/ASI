from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

try:
    import grpc  # type: ignore
    from . import lineage_pb2, lineage_pb2_grpc
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False


class DatasetLineageClient:
    """Client for :class:`DatasetLineageServer`."""

    def __init__(self, address: str) -> None:
        if not _HAS_GRPC:
            raise ImportError("grpcio is required for DatasetLineageClient")
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


__all__ = ["DatasetLineageClient"]
