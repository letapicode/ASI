from __future__ import annotations

"""World-model distillation that records layer explanations."""

from dataclasses import dataclass
from typing import List
import torch
from torch.utils.data import Dataset

from .multimodal_world_model import MultiModalWorldModel
from .world_model_distiller import distill_world_model, DistillConfig


@dataclass
class DistilledComponent:
    student_layer: str
    teacher_layer: str
    mse: float


@dataclass
class DistillationReport:
    components: List[DistilledComponent]
    student: MultiModalWorldModel


def distill_with_explanation(
    teacher: MultiModalWorldModel,
    student: MultiModalWorldModel,
    dataset: Dataset,
    cfg: DistillConfig,
) -> DistillationReport:
    """Run distillation and collect layer-wise MSE explanations."""
    distill_world_model(teacher, student, dataset, cfg)
    comps: List[DistilledComponent] = []
    for (s_name, s_param), (t_name, t_param) in zip(
        student.named_parameters(), teacher.named_parameters()
    ):
        mse = torch.mean((s_param.detach() - t_param.detach()).float() ** 2).item()
        comps.append(DistilledComponent(s_name, t_name, mse))
    return DistillationReport(comps, student)


__all__ = ["DistilledComponent", "DistillationReport", "distill_with_explanation"]
