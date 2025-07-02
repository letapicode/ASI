"""Utility package for ASI prototype modules."""

from .autobench import run_autobench, BenchResult
from .meta_rl_refactor import MetaRLRefactorAgent
from .quantum_hpo import (
    QAEHyperparamSearch,
    amplitude_estimate,
    amplitude_estimate_bayesian,
)
from .collective_constitution import CollectiveConstitution
from .deliberative_alignment import DeliberativeAligner
from .streaming_compression import ReservoirBuffer, StreamingCompressor
from .hierarchical_memory import (
    HierarchicalMemory,
    SSDCache,
    push_remote,
    query_remote,
    push_remote_async,
    query_remote_async,
    push_batch_remote,
    query_batch_remote,
    push_batch_remote_async,
    query_batch_remote_async,
)
from .distributed_memory import DistributedMemory
from .federated_memory_exchange import FederatedMemoryExchange
from .distributed_trainer import DistributedTrainer, MemoryConfig
from .remote_memory import RemoteMemory
from .edge_memory_client import EdgeMemoryClient
from .vector_store import VectorStore, FaissVectorStore
from .async_vector_store import AsyncFaissVectorStore
from .iter_align import IterativeAligner
from .critic_rlhf import CriticScorer, CriticRLHFTrainer
from .chunkwise_retrainer import ChunkWiseRetrainer
from .self_healing_trainer import SelfHealingTrainer
from .scaling_law import BreakpointScalingLaw
from .link_slot_attention import LinkSlotAttention
from .mamba_block import MambaBlock
from .retnet_retention import RetNetRetention
from .hybrid_retention import HybridRetention
from .adaptive_planner import GraphOfThoughtPlanner, AdaptivePlanner

from .pull_request_monitor import (
    list_open_prs,
    check_mergeable,
    list_open_prs_async,
    check_mergeable_async,
)
from .multimodal_world_model import (
    MultiModalWorldModelConfig,
    MultiModalWorldModel,
    TrajectoryDataset as MMTrajectoryDataset,
    train_world_model as train_mm_world_model,
    rollout as rollout_world_model,
)
from .robot_skill_transfer import (
    SkillTransferConfig,
    VideoPolicyDataset,
    SkillTransferModel,
    transfer_skills,
)
from .self_play_env import EnvStep, SimpleEnv, PrioritizedReplayBuffer, rollout_env
from .self_play_skill_loop import (
    SelfPlaySkillLoopConfig,
    run_loop,
    self_play_skill_loop,
)
from .adaptive_curriculum import AdaptiveCurriculum
from .formal_verifier import (
    VerificationResult,
    check_grad_norm,
    check_output_bounds,
    verify_model,
)
from .cross_modal_fusion import (
    CrossModalFusionConfig,
    CrossModalFusion,
    MultiModalDataset,
    train_fusion_model,
    encode_all,
)
from .world_model_rl import (
    RLBridgeConfig,
    TransitionDataset,
    TrajectoryDataset,
    WorldModel as RLWorldModel,
    train_world_model as train_rl_world_model,
    train_with_self_play,
    rollout_policy,
)
from .embodied_calibration import (
    CalibrationConfig,
    CalibrationDataset,
    CalibrationModel,
    calibrate,
)
from .lora_quant import LoRAQuantLinear, apply_quant_lora
from .gradient_compression import GradientCompressionConfig, GradientCompressor
from .data_ingest import (
    download_triples,
    download_triples_async,
    align_triples,
    random_crop,
    generate_transcript,
    pair_modalities,
    random_crop_image,
    add_gaussian_noise,
    text_dropout,
    synthesize_from_world_model,
    offline_synthesizer,
    filter_dataset,
    ActiveDataSelector,
)
from .generative_data_augmentor import GenerativeDataAugmentor
from .causal_graph_learner import CausalGraphLearner
from .transformer_circuits import (
    ActivationRecorder,
    record_attention_weights,
    zero_attention_head,
    restore_attention_head,
    patched_head,
    head_importance,
    AttentionVisualizer,
)
from .neural_arch_search import DistributedArchSearch
from .onnx_utils import export_to_onnx
from .graph_of_thought import GraphOfThought

