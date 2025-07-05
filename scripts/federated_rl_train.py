import argparse
import torch
from asi.self_play_skill_loop import SelfPlaySkillLoopConfig
from asi.federated_rl_trainer import FederatedRLTrainer, FederatedRLTrainerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated self-play RL training")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--agents", type=int, default=2)
    args = parser.parse_args()

    sp_cfg = SelfPlaySkillLoopConfig()
    frl_cfg = FederatedRLTrainerConfig(rounds=args.rounds)
    trainer = FederatedRLTrainer(sp_cfg, frl_cfg=frl_cfg)
    model = trainer.train(num_agents=args.agents)
    torch.save(model.state_dict(), "federated_rl_policy.pt")
    print("saved model to federated_rl_policy.pt")


if __name__ == "__main__":
    main()
