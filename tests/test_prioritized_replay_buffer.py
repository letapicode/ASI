import unittest
import torch

from asi.self_play_env import PrioritizedReplayBuffer


class TestPrioritizedReplayBuffer(unittest.TestCase):
    def test_sampling_prefers_high_reward(self):
        buf = PrioritizedReplayBuffer(2)
        buf.add(torch.tensor([0.0]), 0, 1.0)
        buf.add(torch.tensor([1.0]), 1, 10.0)
        counts = {0: 0, 1: 0}
        for _ in range(200):
            _, acts = buf.sample(1)
            counts[acts[0]] += 1
        self.assertGreater(counts[1], counts[0])


if __name__ == "__main__":
    unittest.main()
