import argparse
import torch
from asi.world_model_rl import TransitionDataset
from asi.ode_world_model import ODEWorldModelConfig, train_ode_world_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ODE world model")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    cfg = ODEWorldModelConfig(state_dim=4, action_dim=2, epochs=args.epochs, batch_size=2)
    transitions = []
    for _ in range(8):
        s = torch.randn(4)
        a = torch.randint(0, 2, (1,)).item()
        ns = torch.randn(4)
        r = torch.randn(())
        transitions.append((s, a, ns, r))
    dataset = TransitionDataset(transitions)

    model = train_ode_world_model(cfg, dataset)
    torch.save(model.state_dict(), "ode_world_model.pt")
    print("saved model to ode_world_model.pt")


if __name__ == "__main__":
    main()
