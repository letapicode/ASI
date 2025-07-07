import unittest
import importlib.machinery
import importlib.util
import types
import sys
import torch
from torch.utils.data import TensorDataset

pkg = types.ModuleType("src")
pkg.__path__ = ["src"]
pkg.__spec__ = importlib.machinery.ModuleSpec("src", None, is_package=True)
sys.modules["src"] = pkg

mods = [
    "world_model_rl",
    "secure_federated_learner",
    "dp_federated_trainer",
]
for m in mods:
    loader = importlib.machinery.SourceFileLoader(f"src.{m}", f"src/{m}.py")
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "src"
    sys.modules[f"src.{m}"] = mod
    loader.exec_module(mod)

RLBridgeConfig = sys.modules["src.world_model_rl"].RLBridgeConfig
TransitionDataset = sys.modules["src.world_model_rl"].TransitionDataset
DPFederatedTrainer = sys.modules["src.dp_federated_trainer"].DPFederatedTrainer
DPFederatedTrainerConfig = sys.modules["src.dp_federated_trainer"].DPFederatedTrainerConfig


class TestDPFederatedTrainer(unittest.TestCase):
    def test_train_runs(self):
        cfg = RLBridgeConfig(state_dim=2, action_dim=2, epochs=1, batch_size=2)
        data = TensorDataset(torch.zeros(4, 2), torch.zeros(4, dtype=torch.long), torch.zeros(4, 2), torch.zeros(4))
        trainer = DPFederatedTrainer(cfg, [data, data], dp_cfg=DPFederatedTrainerConfig(rounds=1, local_epochs=1))
        model = trainer.train()
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
