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
from .encrypted_vector_store import EncryptedVectorStore
from .pq_vector_store import PQVectorStore
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
    retrieval_accuracy,
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
from .low_rank_adapter import LowRankLinear, apply_low_rank_adaptation
from .parameter_efficient_adapter import ParameterEfficientAdapter, PEFTConfig
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
    ingest_translated_triples,
    ActiveDataSelector,
    CrossLingualTranslator,
)
from .generative_data_augmentor import GenerativeDataAugmentor
from .diffusion_world_model import DiffusionWorldModel
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
from .hierarchical_planner import HierarchicalPlanner
try:
    from .federated_memory_server import FederatedMemoryServer
except Exception:  # pragma: no cover - optional
    FederatedMemoryServer = None
try:
    from .federated_kg_memory import FederatedKGMemoryServer
except Exception:  # pragma: no cover - optional
    FederatedKGMemoryServer = None
from .differential_privacy_optimizer import DifferentialPrivacyOptimizer, DifferentialPrivacyConfig

from .embedding_visualizer import EmbeddingVisualizer
from .duplicate_detector import DuplicateDetector
from .telemetry import TelemetryLogger, FineGrainedProfiler, MemoryEventDetector
from .license_inspector import LicenseInspector
from .dataset_versioner import DatasetVersioner
from .dataset_lineage_manager import DatasetLineageManager
from .dataset_anonymizer import DatasetAnonymizer
from .streaming_compression import AdaptiveCompressor, TemporalVectorCompressor
from .context_profiler import profile_model, ContextWindowProfiler
from .accelerator_scheduler import AcceleratorScheduler
from .gpu_aware_scheduler import GPUAwareScheduler
from .adaptive_scheduler import AdaptiveScheduler
from .dataset_bias_detector import (
    compute_word_freq,
    bias_score,
    text_bias_score,
    file_bias_score,
)
from .auto_labeler import AutoLabeler
from .graphql_memory_gateway import GraphQLMemoryGateway
from .world_model_distiller import DistillConfig, distill_world_model
from .summarizing_memory import SummarizingMemory
from .cross_lingual_memory import CrossLingualMemory
from .context_summary_memory import ContextSummaryMemory
from .sensorimotor_pretrainer import (
    SensorimotorPretrainConfig,
    SensorimotorLogDataset,
    pretrain_sensorimotor,
)
from .prompt_optimizer import PromptOptimizer
from .training_anomaly_detector import TrainingAnomalyDetector
from .gradient_patch_editor import GradientPatchEditor, PatchConfig
from .secure_federated_learner import SecureFederatedLearner
from .enclave_runner import EnclaveRunner, EnclaveConfig
from .federated_world_model_trainer import (
    FederatedWorldModelTrainer,
    FederatedTrainerConfig,
)
from .adversarial_robustness import AdversarialRobustnessSuite
from .graph_of_thought import ReasoningDebugger
from .multi_stage_oversight import MultiStageOversight
from .knowledge_graph_memory import KnowledgeGraphMemory
from .memory_dashboard import MemoryDashboard
from .multi_agent_coordinator import MultiAgentCoordinator, RLNegotiator, NegotiationProtocol
from .dp_memory import DifferentialPrivacyMemory
from .privacy_budget_manager import PrivacyBudgetManager
from .causal_reasoner import CausalReasoner
from .multi_agent_graph_planner import MultiAgentGraphPlanner
from .world_model_debugger import WorldModelDebugger
from .model_version_manager import ModelVersionManager
from .model_card import ModelCardGenerator
from .resource_broker import ResourceBroker
from .research_ingest import run_ingestion, suggest_modules
from .quantum_sampler import sample_actions_qae
from .risk_scoreboard import RiskScoreboard
from .semantic_drift_detector import SemanticDriftDetector
from .data_provenance_ledger import DataProvenanceLedger
from .fairness_evaluator import FairnessEvaluator
from .risk_dashboard import RiskDashboard
from .graph_neural_reasoner import GraphNeuralReasoner
from .temporal_reasoner import TemporalReasoner
from .lora_merger import merge_adapters
from .edge_rl_trainer import EdgeRLTrainer
from .federated_rl_trainer import (
    FederatedRLTrainer,
    FederatedRLTrainerConfig,
    PolicyNet,
)
from .adaptive_micro_batcher import AdaptiveMicroBatcher
from .retrieval_explainer import RetrievalExplainer
from .retrieval_rl import RetrievalPolicy, train_policy
from .interpretability_dashboard import InterpretabilityDashboard
from .graph_ui import GraphUI
from .collaborative_healing import CollaborativeHealingLoop
from .compute_budget_tracker import ComputeBudgetTracker
from .budget_aware_scheduler import BudgetAwareScheduler
from .doc_summarizer import summarize_module

from .hpc_scheduler import submit_job, monitor_job, cancel_job
from .collaboration_portal import CollaborationPortal
from .cluster_carbon_dashboard import ClusterCarbonDashboard
from .spiking_layers import LIFNeuron, SpikingLinear


