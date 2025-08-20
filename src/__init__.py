"""Utility package for ASI prototype modules."""

from .autobench import run_autobench, BenchResult
from .meta_rl_refactor import MetaRLRefactorAgent
from .meta_optimizer import MetaOptimizer
from .rl_decision_narrator import RLDecisionNarrator
from .quantum_hpo import (
    QAEHyperparamSearch,
    amplitude_estimate,
    amplitude_estimate_bayesian,
)
from .collective_constitution import CollectiveConstitution
from .deliberative_alignment import DeliberativeAligner
from .normative_reasoner import NormativeReasoner
from .streaming_compression import ReservoirBuffer, StreamingCompressor
from .hierarchical_memory import (
    HierarchicalMemory,
    SSDCache,
)
from .memory_clients import (
    MemoryClientBase,
    RemoteMemoryClient,
    QuantumMemoryClient,
    QuantizedMemoryClient,
    EdgeMemoryClient,
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
from .vector_stores import (
    VectorStore,
    FaissVectorStore,
    LocalitySensitiveHashIndex,
    EncryptedVectorStore,
    EphemeralVectorStore,
    HolographicVectorStore,
    PQVectorStore,
    AsyncFaissVectorStore,
)
from .dnc_memory import DNCMemory
from .iter_align import IterativeAligner
from .critic_rlhf import CriticScorer, CriticRLHFTrainer
from .rlaif_trainer import RLAIFTrainer, SyntheticCritic
from .chunkwise_retrainer import ChunkWiseRetrainer
from .secure_dataset_exchange import SecureDatasetExchange, DatasetIntegrityProof
from .p2p_dataset_exchange import P2PDatasetExchange
from .code_refine import CodeRefinePipeline
from .self_healing_trainer import SelfHealingTrainer
from .scaling_law import BreakpointScalingLaw
from .link_slot_attention import LinkSlotAttention
from .mamba_block import MambaBlock
from .retention import RetNetRetention, HybridRetention
from .adaptive_planner import GraphOfThoughtPlanner, AdaptivePlanner

from .pull_request_tools import (
    list_open_prs,
    check_mergeable,
    list_open_prs_async,
    check_mergeable_async,
    check_pr_conflicts,
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
from .opponent_generator import OpponentGenerator
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
    simulate_counterfactual,
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
    paraphrase_multilingual,
    ingest_translated_triples,
    ActiveDataSelector,
    CrossLingualTranslator,
)
from .adaptive_translator import AdaptiveTranslator
from .advanced_ingest import LLMIngestParser
from .generative_data_augmentor import GenerativeDataAugmentor
from .diffusion_world_model import DiffusionWorldModel
from .ode_world_model import (
    ODEWorldModel,
    ODEWorldModelConfig,
    train_ode_world_model,
    rollout_policy as rollout_ode_policy,
)
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
from .transformer_circuit_analyzer import (
    TransformerCircuitAnalyzer,
    gradient_head_importance,
)
from .neural_arch_search import DistributedArchSearch
from .neuroevolution_search import NeuroevolutionSearch
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
from .graph_visualizer import (
    GOTVisualizer,
    GOT3DVisualizer,
    GOT3DViewer,
    WebSocketServer,
    D3GraphVisualizer,
    LineageVisualizer,
    DatasetLineageDashboard,
    KGVisualizer,
)
from .duplicate_detector import DuplicateDetector
from .telemetry import TelemetryLogger, MemoryEventDetector
from .fine_grained_profiler import FineGrainedProfiler
from .telemetry_aggregator import TelemetryAggregator
from .cognitive_load_monitor import CognitiveLoadMonitor
from .license_inspector import LicenseInspector
from .dataset_versioner import DatasetVersioner
from .dataset_lineage import DatasetLineageManager
from .anonymizer import DatasetAnonymizer, NERAnonymizer
from .dataset_discovery import (
    DiscoveredDataset,
    discover_huggingface,
    discover_kaggle,
    store_datasets,
)
from .rl_dataset_discovery import DatasetQualityAgent
from .streaming_dataset_watcher import StreamingDatasetWatcher
from .streaming_compression import AdaptiveCompressor, TemporalVectorCompressor
from .context_profiler import profile_model, ContextWindowProfiler
from .schedulers import AcceleratorScheduler, AdaptiveScheduler, EnergyAwareScheduler, BatteryAwareScheduler
from .dataset_bias_detector import (
    compute_word_freq,
    bias_score,
    text_bias_score,
    file_bias_score,
    DatasetBiasDetector,
    DataBiasMitigator,
)
from .dataset_weight_agent import DatasetWeightAgent
from .data_poison_detector import DataPoisonDetector
from .auto_labeler import AutoLabeler
from .graphql_memory_gateway import GraphQLMemoryGateway
from .world_model_distiller import DistillConfig, distill_world_model
from .summarizing_memory import SummarizingMemory
from .pruning_manager import MemoryPruningManager, GraphPruningManager
from .cross_lingual_memory import CrossLingualMemory
from .cross_lingual_kg_memory import CrossLingualKGMemory
from .cross_lingual_graph import CrossLingualReasoningGraph
from .cross_lingual_voice_chat import CrossLingualVoiceChat
from .context_summary_memory import ContextSummaryMemory
from .multimodal_summary_memory import MultiModalSummaryMemory
from .sensorimotor_pretrainer import (
    SensorimotorPretrainConfig,
    SensorimotorLogDataset,
    pretrain_sensorimotor,
)
from .prompt_optimizer import PromptOptimizer
from .user_preferences import UserPreferences
from .training_anomaly_detector import TrainingAnomalyDetector
from .gradient_patch_editor import GradientPatchEditor, PatchConfig
from .secure_federated_learner import SecureFederatedLearner

from .proofs import RetrievalProof, ZKGradientProof, ZKRetrievalProof, ZKVerifier


from .enclave_runner import EnclaveRunner, EnclaveConfig
from .federated_world_model_trainer import (
    FederatedWorldModelTrainer,
    FederatedTrainerConfig,
)
from .fhe_federated_trainer import FHEFederatedTrainer, FHEFederatedTrainerConfig
from .dp_federated_trainer import DPFederatedTrainer, DPFederatedTrainerConfig
from .adversarial_robustness import AdversarialRobustnessSuite
from .graph_of_thought import ReasoningDebugger, AnalogicalReasoningDebugger
from .multi_stage_oversight import MultiStageOversight
from .knowledge_graph_memory import KnowledgeGraphMemory
from .knowledge_base_client import KnowledgeBaseClient
from .dashboards import (
    MemoryDashboard,
    AlignmentDashboard,
    InterpretabilityDashboard,
    IntrospectionDashboard,
    ClusterCarbonDashboard,
    RiskDashboard,
    MultiAgentDashboard,
)
from .multi_agent_coordinator import MultiAgentCoordinator, RLNegotiator, NegotiationProtocol
from .dp_memory import DifferentialPrivacyMemory
from .privacy import PrivacyBudgetManager, PrivacyAuditor, PrivacyGuard
from .causal_reasoner import CausalReasoner
from .multi_agent_graph_planner import MultiAgentGraphPlanner
from .world_model_debugger import WorldModelDebugger
from .model_version_manager import ModelVersionManager
from .model_card import ModelCardGenerator
from .resource_broker import ResourceBroker
from .research_ingest import run_ingestion, suggest_modules
from .quantum_sampling import sample_actions_qae, amplify_search
from .quantum_multimodal_retrieval import quantum_crossmodal_search
from .risk_scoreboard import RiskScoreboard
from .semantic_drift_detector import SemanticDriftDetector
from .provenance_ledger import DataProvenanceLedger, BlockchainProvenanceLedger
from .fairness import (
    FairnessEvaluator,
    FairnessFeedback,
    FairnessVisualizer,
    CrossLingualFairnessEvaluator,
    FairnessAdaptationPipeline,
)
from .graph_neural_reasoner import GraphNeuralReasoner
from .gnn_memory import GNNMemory
from .temporal_reasoner import TemporalReasoner
from .lora_merger import merge_adapters
from .edge_rl_trainer import EdgeRLTrainer

from .federated_edge_rl import FederatedEdgeRLTrainer, FederatedEdgeConfig

from .federated_rl_trainer import (
    FederatedRLTrainer,
    FederatedRLTrainerConfig,
    PolicyNet,
)

from .adaptive_micro_batcher import AdaptiveMicroBatcher
from .retrieval_explainer import RetrievalExplainer
from .retrieval_rl import RetrievalPolicy, train_policy
from .retrieval_policy_updater import RetrievalPolicyUpdater
from .graph_ui import GraphUI
from .nl_graph_editor import NLGraphEditor
from .voice_graph_controller import VoiceGraphController, SignLanguageGraphController
from .collaborative_healing import CollaborativeHealingLoop
from .compute_budget_tracker import ComputeBudgetTracker
from .schedulers import BudgetAwareScheduler
from .doc_summarizer import summarize_module
from .dataset_summarizer import summarize_dataset

from .hpc_schedulers import (
    submit_job,
    monitor_job,
    cancel_job,
    HPCJobScheduler,
    HPCBaseScheduler,
    MultiClusterScheduler,
    make_scheduler,
    _record_carbon_saving,
)

from .carbon_aware_scheduler import (
    CarbonAwareScheduler,
    CostAwareScheduler,
    CarbonCostAwareScheduler,
    MultiProviderScheduler,
    get_current_price,
    get_hourly_price_forecast,
)

from .rl_schedulers import (
    RLCarbonScheduler,
    RLCostScheduler,
    AdaptiveCostScheduler,
    CoordinatedRLCostScheduler,
    RLMultiClusterScheduler,
    DeepRLScheduler,
)
from .meta_scheduler import MetaScheduler
from .carbon_aware_dataset_ingest import CarbonAwareDatasetIngest

from .collaboration_portal import CollaborationPortal
from .distributed_anomaly_monitor import DistributedAnomalyMonitor
from .spiking_layers import LIFNeuron, SpikingLinear


from .fhe_runner import run_fhe
from .hardware_backends import (
    FPGAAccelerator,
    FPGAConfig,
    configure_fpga,
    get_fpga_config,
    _HAS_FPGA,
    AnalogAccelerator,
    AnalogConfig,
    configure_analog,
    get_analog_config,
    _HAS_ANALOG,
)

from .emotion_detector import detect_emotion
from .bio_memory_replay import run_nightly_replay
from .sign_language import SignLanguageRecognizer
from .vr_graph_explorer import VRGraphVisualizer, VRGraphExplorer
from .reasoning_summary_translator import ReasoningSummaryTranslator
from .reasoning_kb_bridge import (
    graph_to_triples,
    HistoryKGExporter,
    get_following_steps,
    get_step_metadata,
)

from .multi_agent_self_play import MultiAgentSelfPlayConfig, run_multi_agent_self_play, MultiAgentSelfPlay

