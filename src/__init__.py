"""Utility package for ASI prototype modules."""

# Modules are exposed under the :mod:`asi` namespace. Imports are kept lazy to
# avoid optional heavy dependencies at package import time.

__all__ = [
    "autobench",
    "meta_rl_refactor",
    "quantum_hpo",
    "collective_constitution",
    "deliberative_alignment",
    "streaming_compression",
    "hierarchical_memory",
    "iter_align",
    "critic_rlhf",
    "chunkwise_retrainer",
]

