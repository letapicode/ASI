import unittest
import time

from asi.scheduler_service import ScheduleServer, propose_remote, accept_remote
from asi import scheduler_pb2

class TestScheduleService(unittest.TestCase):
    def test_propose_accept(self):
        try:
            import grpc  # noqa: F401
        except Exception:
            self.skipTest("grpcio not available")
        server = ScheduleServer("localhost:51050")
        server.start()
        time.sleep(0.1)

        proposal = scheduler_pb2.ScheduleProposal(agent="a1")
        reply = propose_remote("localhost:51050", proposal)
        self.assertTrue(reply.accept)

        decision = scheduler_pb2.ScheduleDecision(id="job1", accepted=reply.accept, start_ts=reply.start_ts)
        ack = accept_remote("localhost:51050", decision)
        server.stop(0)

        self.assertTrue(ack.ok)


if __name__ == "__main__":
    unittest.main()
