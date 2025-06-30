import unittest
import torch

from asi.multimodal_world_model import (
    MultiModalWorldModelConfig,
    MultiModalWorldModel,
    TrajectoryDataset,
    train_world_model,
    rollout,
)


def simple_tokenizer(text: str):
    return [ord(c) % 50 for c in text]


class TestMultiModalWorldModel(unittest.TestCase):
    def setUp(self):
        cfg = MultiModalWorldModelConfig(vocab_size=50, img_channels=3, action_dim=5, embed_dim=8)
        self.model = MultiModalWorldModel(cfg)
        img = torch.randn(3, 8, 8)
        entry = ("a", img, torch.tensor(0), "b", img.clone(), 0.0)
        self.dataset = TrajectoryDataset([entry], simple_tokenizer)

    def test_forward_shapes(self):
        t, img, a, _, _, _ = self.dataset[0]
        t = t.unsqueeze(0)
        img = img.unsqueeze(0)
        a = a.unsqueeze(0)
        state, reward = self.model(t, img, a)
        self.assertEqual(state.shape, (1, 8))
        self.assertEqual(reward.shape, (1,))

    def test_rollout(self):
        start_t, start_img, _, _, _, _ = self.dataset[0]
        policy = lambda s: torch.tensor(0, dtype=torch.long, device=s.device)
        states, rewards = rollout(self.model, start_t, start_img, policy, steps=3)
        self.assertEqual(len(states), 3)
        self.assertEqual(len(rewards), 3)
        self.assertEqual(states[0].shape, (8,))

    def test_train_world_model(self):
        before = self.model.dyn.reward_head.weight.clone()
        train_world_model(self.model, self.dataset, epochs=1, batch_size=1)
        after = self.model.dyn.reward_head.weight
        self.assertFalse(torch.allclose(before, after))


if __name__ == "__main__":
    unittest.main()
