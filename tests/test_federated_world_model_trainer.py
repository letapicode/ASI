import unittest
import importlib.machinery
import importlib.util
import types
import sys
import torch
from torch.utils.data import TensorDataset
import numpy as np

src_pkg = types.ModuleType('src')
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src'] = src_pkg

loader_bci = importlib.machinery.SourceFileLoader('src.bci_feedback_trainer', 'src/bci_feedback_trainer.py')
spec_bci = importlib.util.spec_from_loader(loader_bci.name, loader_bci)
bci_mod = importlib.util.module_from_spec(spec_bci)
bci_mod.__package__ = 'src'
sys.modules['src.bci_feedback_trainer'] = bci_mod
loader_bci.exec_module(bci_mod)
BCIFeedbackTrainer = bci_mod.BCIFeedbackTrainer

loader_learner = importlib.machinery.SourceFileLoader('src.secure_federated_learner', 'src/secure_federated_learner.py')
spec_l = importlib.util.spec_from_loader(loader_learner.name, loader_learner)
learner_mod = importlib.util.module_from_spec(spec_l)
learner_mod.__package__ = 'src'
sys.modules['src.secure_federated_learner'] = learner_mod
loader_learner.exec_module(learner_mod)
SecureFederatedLearner = learner_mod.SecureFederatedLearner

loader_wm = importlib.machinery.SourceFileLoader('src.world_model_rl', 'src/world_model_rl.py')
spec_wm = importlib.util.spec_from_loader(loader_wm.name, loader_wm)
wm_mod = importlib.util.module_from_spec(spec_wm)
wm_mod.__package__ = 'src'
sys.modules['src.world_model_rl'] = wm_mod
loader_wm.exec_module(wm_mod)
RLBridgeConfig = wm_mod.RLBridgeConfig

loader_fw = importlib.machinery.SourceFileLoader('src.federated_world_model_trainer', 'src/federated_world_model_trainer.py')
spec_fw = importlib.util.spec_from_loader(loader_fw.name, loader_fw)
fw_mod = importlib.util.module_from_spec(spec_fw)
fw_mod.__package__ = 'src'
sys.modules['src.federated_world_model_trainer'] = fw_mod
loader_fw.exec_module(fw_mod)
FederatedWorldModelTrainer = fw_mod.FederatedWorldModelTrainer

class TestFederatedWorldModelTrainer(unittest.TestCase):
    def test_train(self):
        cfg = RLBridgeConfig(state_dim=2, action_dim=2, epochs=1, batch_size=2)
        data = TensorDataset(torch.zeros(4,2), torch.zeros(4,dtype=torch.long), torch.zeros(4,2), torch.zeros(4))
        trainer = FederatedWorldModelTrainer(cfg, [data, data])
        model = trainer.train()
        self.assertIsNotNone(model)

    def test_reward_sync(self):
        cfg = RLBridgeConfig(state_dim=2, action_dim=2, epochs=1, batch_size=2)
        ds1 = TensorDataset(
            torch.zeros(2, 2),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(2, 2),
            torch.zeros(2),
        )
        ds2 = TensorDataset(
            torch.zeros(2, 2),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(2, 2),
            torch.zeros(2),
        )
        bci = BCIFeedbackTrainer(cfg)
        learner = SecureFederatedLearner()

        s1 = [np.ones(4), np.zeros(4)]
        s2 = [np.zeros(4), np.ones(4)]

        def hook():
            rewards = [
                bci.aggregate_signal_rewards(sig, learner)
                for sig in zip(s1, s2)
            ]
            return [rewards, rewards]

        trainer = FederatedWorldModelTrainer(
            cfg, [ds1, ds2], reward_sync_hook=hook
        )
        model = trainer.train()
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
