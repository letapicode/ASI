import torch
from asi.edge_rl_trainer import EdgeRLTrainer
from asi.compute_budget_tracker import ComputeBudgetTracker


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def main() -> None:
    model = ToyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    budget = ComputeBudgetTracker(0.001)
    budget.start("edge")
    data = [(torch.randn(1, 2), torch.randn(1, 2)) for _ in range(10)]
    trainer = EdgeRLTrainer(model, opt, budget)
    steps = trainer.train(data)
    budget.stop()
    print(f"steps:{steps}")


if __name__ == "__main__":
    main()
