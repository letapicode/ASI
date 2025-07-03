import argparse
import torch
from asi.world_model_rl import RLBridgeConfig, TransitionDataset
from asi.federated_world_model_trainer import FederatedWorldModelTrainer, FederatedTrainerConfig

def main() -> None:
    parser = argparse.ArgumentParser(description="Federated world model training")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1, dest="local_epochs")
    args = parser.parse_args()

    cfg = RLBridgeConfig(state_dim=4, action_dim=2)
    data1 = [
        (torch.randn(4), 0, torch.randn(4), torch.randn(()))
        for _ in range(8)
    ]
    data2 = [
        (torch.randn(4), 1, torch.randn(4), torch.randn(()))
        for _ in range(8)
    ]
    ds1 = TransitionDataset(data1)
    ds2 = TransitionDataset(data2)
    tcfg = FederatedTrainerConfig(rounds=args.rounds, local_epochs=args.local_epochs)
    trainer = FederatedWorldModelTrainer(cfg, [ds1, ds2], trainer_cfg=tcfg)
    model = trainer.train()
    torch.save(model.state_dict(), "federated_model.pt")
    print("saved model to federated_model.pt")

if __name__ == "__main__":
    main()
