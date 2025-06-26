import unittest
import torch
from torch import nn

from asi.chunkwise_retrainer import ChunkWiseRetrainer

class TestChunkWiseRetrainer(unittest.TestCase):
    def test_training_reduces_loss(self):
        torch.manual_seed(0)
        vocab = 10
        embed_dim = 8
        model = nn.Sequential(nn.Embedding(vocab, embed_dim), nn.Linear(embed_dim, vocab))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = ChunkWiseRetrainer(model, optimizer, chunk_size=4)
        seq = torch.tensor([0,1,2,3,4,5,6,7,8,9], dtype=torch.long)

        def compute_loss():
            logits = model(seq[:-1].unsqueeze(0))
            return nn.functional.cross_entropy(logits.view(-1, vocab), seq[1:]).item()

        loss_before = compute_loss()
        trainer.train([seq], epochs=3)
        loss_after = compute_loss()
        self.assertLess(loss_after, loss_before)

if __name__ == '__main__':
    unittest.main()
