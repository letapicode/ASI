import unittest
import importlib.machinery
import importlib.util
import types
import sys
import torch

pkg = types.ModuleType("src")
pkg.__path__ = ["src"]
pkg.__spec__ = importlib.machinery.ModuleSpec("src", None, is_package=True)
sys.modules["src"] = pkg

mods = [
    "self_play_env",
    "self_play_skill_loop",
    "secure_federated_learner",
    "federated_rl_trainer",
]

for m in mods:
    loader = importlib.machinery.SourceFileLoader(f"src.{m}", f"src/{m}.py")
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "src"
    sys.modules[f"src.{m}"] = mod
    loader.exec_module(mod)

FederatedRLTrainer = sys.modules["src.federated_rl_trainer"].FederatedRLTrainer
FederatedRLTrainerConfig = sys.modules[
    "src.federated_rl_trainer"
].FederatedRLTrainerConfig
SelfPlaySkillLoopConfig = sys.modules[
    "src.self_play_skill_loop"
].SelfPlaySkillLoopConfig


class TestFederatedRLTrainer(unittest.TestCase):
    def test_train_runs(self):
        sp_cfg = SelfPlaySkillLoopConfig(cycles=1, steps=2, epochs=1, batch_size=1)
        cfg = FederatedRLTrainerConfig(rounds=1, local_steps=2, lr=0.1)
        trainer = FederatedRLTrainer(sp_cfg, frl_cfg=cfg)
        model = trainer.train(num_agents=2)
        self.assertIsInstance(model, torch.nn.Module)


if __name__ == "__main__":
    unittest.main()
