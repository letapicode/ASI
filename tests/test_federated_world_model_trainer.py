import unittest
import torch
from torch.utils.data import TensorDataset

from asi.world_model_rl import RLBridgeConfig
from asi.federated_world_model_trainer import FederatedWorldModelTrainer

class TestFederatedWorldModelTrainer(unittest.TestCase):
    def test_train(self):
        cfg = RLBridgeConfig(state_dim=2, action_dim=2, epochs=1, batch_size=2)
        data = TensorDataset(torch.zeros(4,2), torch.zeros(4,dtype=torch.long), torch.zeros(4,2), torch.zeros(4))
        trainer = FederatedWorldModelTrainer(cfg, [data, data])
        model = trainer.train()
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
