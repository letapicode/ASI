import argparse
import torch
from asi.world_model_rl import RLBridgeConfig, TransitionDataset
from asi.federated_world_model_trainer import (
    FederatedWorldModelTrainer,
    FederatedTrainerConfig,
)
from asi.fhe_federated_trainer import FHEFederatedTrainer, FHEFederatedTrainerConfig
import tenseal as ts

def main() -> None:
    parser = argparse.ArgumentParser(description="Federated world model training")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1, dest="local_epochs")
    parser.add_argument("--use-fhe", action="store_true", help="train with FHE-encrypted gradients")
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
    if args.use_fhe:
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        ctx.generate_galois_keys()
        ctx.global_scale = 2**40
        ftcfg = FHEFederatedTrainerConfig(
            rounds=args.rounds, local_epochs=args.local_epochs, lr=tcfg.lr
        )
        trainer = FHEFederatedTrainer(cfg, [ds1, ds2], ctx, trainer_cfg=ftcfg)
    else:
        trainer = FederatedWorldModelTrainer(cfg, [ds1, ds2], trainer_cfg=tcfg)
    model = trainer.train()
    torch.save(model.state_dict(), "federated_model.pt")
    print("saved model to federated_model.pt")

if __name__ == "__main__":
    main()
