from __future__ import annotations
from concurrent import futures

try:
    import grpc  # type: ignore
    from . import scheduler_pb2, scheduler_pb2_grpc
    _HAS_GRPC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GRPC = False


if _HAS_GRPC:

    class ScheduleServer(scheduler_pb2_grpc.ScheduleServiceServicer):
        """Minimal gRPC server for RL scheduler coordination."""

        def __init__(self, address: str = "localhost:51050", max_workers: int = 2) -> None:
            self.address = address
            self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
            scheduler_pb2_grpc.add_ScheduleServiceServicer_to_server(self, self.server)
            self.server.add_insecure_port(address)
            self.last_decision: scheduler_pb2.ScheduleDecision | None = None

        def Propose(
            self, request: scheduler_pb2.ScheduleProposal, context
        ) -> scheduler_pb2.ScheduleReply:  # noqa: N802
            start_ts = request.forecast[0].ts if request.forecast else 0
            return scheduler_pb2.ScheduleReply(accept=True, start_ts=start_ts, cluster="default")

        def Accept(
            self, request: scheduler_pb2.ScheduleDecision, context
        ) -> scheduler_pb2.Ack:  # noqa: N802
            self.last_decision = request
            return scheduler_pb2.Ack(ok=True)

        def start(self) -> None:
            self.server.start()

        def stop(self, grace: float = 0) -> None:
            self.server.stop(grace)


    def propose_remote(address: str, proposal: scheduler_pb2.ScheduleProposal) -> scheduler_pb2.ScheduleReply:
        with grpc.insecure_channel(address) as channel:
            stub = scheduler_pb2_grpc.ScheduleServiceStub(channel)
            return stub.Propose(proposal)


    def accept_remote(address: str, decision: scheduler_pb2.ScheduleDecision) -> scheduler_pb2.Ack:
        with grpc.insecure_channel(address) as channel:
            stub = scheduler_pb2_grpc.ScheduleServiceStub(channel)
            return stub.Accept(decision)


    __all__ = [
        "ScheduleServer",
        "propose_remote",
        "accept_remote",
    ]
else:  # pragma: no cover - optional dependency
    ScheduleServer = None  # type: ignore

    def propose_remote(address: str, proposal):  # type: ignore
        raise ImportError("grpcio is required for propose_remote")

    def accept_remote(address: str, decision):  # type: ignore
        raise ImportError("grpcio is required for accept_remote")

    __all__ = ["ScheduleServer", "propose_remote", "accept_remote"]
