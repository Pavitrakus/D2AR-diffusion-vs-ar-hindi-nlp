"""
Base configuration classes and enums for the research project.

Defines the core taxonomy:
- 8 Models (4 Diffusion + 4 Auto-Regressive)
- 4 Tasks (2 Language + 2 Legal)
- 5 Pipeline Steps
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
import os


# ============================================================
# Enums defining the experiment taxonomy
# ============================================================

class ModelType(str, Enum):
    """Category of language model."""
    DIFFUSION = "diffusion"
    AUTOREGRESSIVE = "autoregressive"


class ModelName(str, Enum):
    """All 8 models under study."""
    # Diffusion Models
    SEDD = "SEDD"
    LLADA = "LLaDA"
    D3PM = "D3PM"
    DIFFUSELM = "DiffuseLM"
    # Auto-Regressive Models
    LLAMA = "LLaMA"
    GEMMA = "Gemma"
    MISTRAL = "Mistral"
    BERT = "IndicBERT"

    @property
    def model_type(self) -> ModelType:
        """Return the model category."""
        diffusion_models = {self.SEDD, self.LLADA, self.D3PM, self.DIFFUSELM}
        return ModelType.DIFFUSION if self in diffusion_models else ModelType.AUTOREGRESSIVE

    @property
    def is_diffusion(self) -> bool:
        return self.model_type == ModelType.DIFFUSION

    @property
    def is_autoregressive(self) -> bool:
        return self.model_type == ModelType.AUTOREGRESSIVE


class TaskName(str, Enum):
    """All 4 Hindi NLP tasks."""
    HINDI_TRANSLATION = "Hindi_translation"
    HINDI_SUMMARY = "Hindi_summary"
    BAIL_PREDICTION = "Hindi_legal_bail"
    JUDGE_VERDICT = "Hindi_judge_verdict"

    @property
    def task_category(self) -> str:
        """Return 'language' or 'legal'."""
        if self in {self.HINDI_TRANSLATION, self.HINDI_SUMMARY}:
            return "language"
        return "legal"

    @property
    def is_generation(self) -> bool:
        """Whether this task requires text generation (vs classification)."""
        return self in {self.HINDI_TRANSLATION, self.HINDI_SUMMARY}

    @property
    def is_classification(self) -> bool:
        """Whether this task requires classification."""
        return self in {self.BAIL_PREDICTION, self.JUDGE_VERDICT}


class PipelineStep(str, Enum):
    """All 5 pipeline steps."""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    FINE_TUNING = "fine_tuning"
    RAG = "rag"
    AGENTIC = "agentic"


# ============================================================
# Configuration Dataclasses
# ============================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment (1 model × 1 task × 1 pipeline step)."""

    # Core identifiers
    model_name: ModelName
    task_name: TaskName
    pipeline_step: PipelineStep

    # Reproducibility
    seed: int = 42
    num_runs: int = 1

    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    num_beams: int = 1

    # Evaluation
    num_eval_samples: int = 200
    batch_size: int = 8

    # Fine-tuning (only used when pipeline_step == FINE_TUNING)
    learning_rate: float = 2e-4
    num_epochs: int = 3
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 1024

    # Few-shot
    num_few_shot_examples: int = 5

    # RAG
    rag_top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Device
    device: str = "cuda"
    dtype: str = "float16"
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    @property
    def experiment_id(self) -> str:
        """Unique identifier for this experiment."""
        return f"{self.model_name.value}__{self.task_name.value}__{self.pipeline_step.value}"

    @property
    def results_dir(self) -> Path:
        """Directory to save results for this experiment."""
        return Path("results") / "raw" / self.task_name.value / self.model_name.value / self.pipeline_step.value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result


@dataclass
class ProjectConfig:
    """Global project configuration."""

    # Project metadata
    project_name: str = "Diffusion vs AR Models for Hindi NLP"
    version: str = "1.0.0"

    # Directories
    project_root: Path = field(default_factory=lambda: Path.cwd())
    results_dir: Path = field(default_factory=lambda: Path("results"))
    data_dir: Path = field(default_factory=lambda: Path("data"))
    cache_dir: Path = field(default_factory=lambda: Path(".cache"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))

    # HuggingFace
    hf_token: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN"))
    hf_cache_dir: Optional[str] = field(default_factory=lambda: os.getenv("HF_HOME"))

    # Server (IIT Kanpur)
    server_data_dir: str = "/data/debkanta"

    # Global settings
    default_seed: int = 42
    default_device: str = "cuda"
    default_dtype: str = "float16"

    # Priority models (for initial experiments)
    priority_models: List[ModelName] = field(
        default_factory=lambda: [
            ModelName.SEDD,
            ModelName.LLADA,
            ModelName.LLAMA,
            ModelName.GEMMA,
        ]
    )

    # All models
    all_models: List[ModelName] = field(
        default_factory=lambda: list(ModelName)
    )

    # All tasks
    all_tasks: List[TaskName] = field(
        default_factory=lambda: list(TaskName)
    )

    # All pipeline steps
    all_steps: List[PipelineStep] = field(
        default_factory=lambda: list(PipelineStep)
    )

    def ensure_dirs(self):
        """Create all necessary directories."""
        for task in self.all_tasks:
            for model in self.all_models:
                for step in self.all_steps:
                    dir_path = self.results_dir / "raw" / task.value / model.value / step.value
                    dir_path.mkdir(parents=True, exist_ok=True)

        (self.results_dir / "metrics" / "per_experiment").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "metrics" / "aggregated").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "figures" / "comparison_heatmaps").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "figures" / "bar_charts").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "figures" / "radar_charts").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "figures" / "training_curves").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "figures" / "confusion_matrices").mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def get_experiment_config(
        self,
        model: ModelName,
        task: TaskName,
        step: PipelineStep,
        **overrides
    ) -> ExperimentConfig:
        """Create an ExperimentConfig with project defaults."""
        defaults = {
            "model_name": model,
            "task_name": task,
            "pipeline_step": step,
            "seed": self.default_seed,
            "device": self.default_device,
            "dtype": self.default_dtype,
        }
        defaults.update(overrides)
        return ExperimentConfig(**defaults)

    def get_all_experiment_configs(self, priority_only: bool = False) -> List[ExperimentConfig]:
        """Generate all experiment configurations."""
        models = self.priority_models if priority_only else self.all_models
        configs = []
        for model in models:
            for task in self.all_tasks:
                for step in self.all_steps:
                    configs.append(self.get_experiment_config(model, task, step))
        return configs
