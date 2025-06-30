import unittest
import torch
from torch.utils.data import DataLoader, Dataset

from asi.multimodal_world_model import (
    MultiModalWorldModel,
    train_world_model,
    rollout,
)


class DummyDataset(Dataset):
    def __init__(self, size: int, vocab: int, action_size: int, dim: int) -> None:
        self.text = torch.randint(0, vocab, (size, 4))
        self.img = torch.randn(size, 3, 8, 8)
        self.act = torch.randint(0, action_size, (size,))
        self.target = torch.randn(size, dim)

    def __len__(self):
        return self.text.size(0)

    def __getitem__(self, idx):
        return (
            self.text[idx],
            self.img[idx],
            self.act[idx],
            self.target[idx],
        )


class TestMultiModalWorldModel(unittest.TestCase):
    def setUp(self):
        self.vocab = 10
        self.action = 5
        self.dim = 16
        self.model = MultiModalWorldModel(self.vocab, self.action, dim=self.dim)

    def test_forward_shape(self):
        text = torch.randint(0, self.vocab, (2, 4))
        img = torch.randn(2, 3, 8, 8)
        act = torch.randint(0, self.action, (2,))
        out = self.model(text, img, act)
        self.assertEqual(out.shape, (2, self.dim))

    def test_train_and_rollout(self):
        dataset = DummyDataset(4, self.vocab, self.action, self.dim)
        loader = DataLoader(dataset, batch_size=2)
        train_world_model(self.model, loader, epochs=1)
        texts = torch.randint(0, self.vocab, (1, 3, 4))
        images = torch.randn(1, 3, 3, 8, 8)
        actions = torch.randint(0, self.action, (1, 3))
        preds = rollout(self.model, texts, images, actions)
        self.assertEqual(preds.shape, (1, 3, self.dim))


if __name__ == "__main__":
    unittest.main()
