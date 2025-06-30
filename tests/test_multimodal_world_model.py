import unittest
import torch
from asi.multimodal_world_model import (
    MultiModalWorldModelConfig,
    MultiModalWorldModel,
    TrajectoryDataset,
    train_world_model,
    rollout,
)


def _tokenizer(text: str):
    ids = [(ord(c) - 97) % 20 + 1 for c in text]
    ids = (ids + [0] * 4)[:4]
    return ids


def _make_dataset(n=4):
    items = []
    for i in range(n):
        t = "t" * 4
        img = torch.randn(3, 16, 16)
        action = i % 3
        nt = "n" * 4
        nimg = torch.randn(3, 16, 16)
        reward = float(i)
        items.append((t, img, action, nt, nimg, reward))
    return TrajectoryDataset(items, _tokenizer)


class TestMultiModalWorldModel(unittest.TestCase):
    def test_train_and_rollout(self):
        cfg = MultiModalWorldModelConfig(vocab_size=32, img_channels=3, action_dim=3, embed_dim=16)
        model = MultiModalWorldModel(cfg)
        ds = _make_dataset(6)
        before = model.obs_enc.text_emb.weight.clone()
        train_world_model(model, ds, epochs=1, batch_size=2)
        self.assertFalse(torch.allclose(before, model.obs_enc.text_emb.weight))
        tokens, img, _, _, _, _ = ds[0]
        start_t = tokens.unsqueeze(0)
        start_i = img.unsqueeze(0)
        policy = lambda s: torch.zeros(s.size(0), dtype=torch.long)
        states, rewards = rollout(model, start_t, start_i, policy, steps=3)
        self.assertEqual(len(states), 3)
        self.assertEqual(len(rewards), 3)


if __name__ == "__main__":
    unittest.main()
