import argparse
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from asi.secure_federated_learner import SecureFederatedLearner


def generate_dataset(n: int = 32, dim: int = 4) -> TensorDataset:
    """Return a toy classification dataset."""
    x = torch.randn(n, dim)
    y = (x.sum(dim=1) > 0).long()
    return TensorDataset(x, y)


def local_gradients(model: nn.Module, ds: TensorDataset, lr: float) -> torch.Tensor:
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    grads = [torch.zeros_like(p) for p in params]
    for x, y in loader:
        out = model(x)
        loss = loss_fn(out, y)
        opt.zero_grad()
        loss.backward()
        for g, p in zip(grads, params):
            g += p.grad.detach()
        opt.step()
    flat = torch.cat([g.view(-1) for g in grads]) / len(loader)
    return flat


def apply_gradient(model: nn.Module, grad: torch.Tensor, lr: float) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    start = 0
    for p in params:
        num = p.numel()
        g = grad[start : start + num].view_as(p)
        p.data -= lr * g
        start += num


def train(rounds: int, clients: int, lr: float) -> nn.Module:
    datasets = [generate_dataset() for _ in range(clients)]
    model = nn.Linear(4, 2)
    learner = SecureFederatedLearner()

    for _ in range(rounds):
        enc_grads: List[torch.Tensor] = []
        for ds in datasets:
            grad = local_gradients(model, ds, lr)
            enc_grads.append(learner.encrypt(grad))
        agg = learner.aggregate([learner.decrypt(g) for g in enc_grads])
        apply_gradient(model, agg, lr)
    return model


def evaluate(model: nn.Module, ds: TensorDataset) -> float:
    loader = DataLoader(ds, batch_size=len(ds))
    x, y = next(iter(loader))
    with torch.no_grad():
        pred = model(x).argmax(dim=-1)
        acc = (pred == y).float().mean().item()
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy federated training example")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    model = train(args.rounds, args.clients, args.lr)
    test_ds = generate_dataset(n=64)
    acc = evaluate(model, test_ds)
    print({"accuracy": acc})


if __name__ == "__main__":
    main()
