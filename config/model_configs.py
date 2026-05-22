"""
Model-specific configurations for all 8 models.

Contains HuggingFace model IDs, tokenizer settings, LoRA targets,
and generation parameters optimized per model.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from config.base_config import ModelName, ModelType


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    # Identification
    name: ModelName
    model_type: ModelType
    display_name: str
    paper_title: str
    paper_year: int

    # HuggingFace / Source
    hf_model_id: str
    hf_tokenizer_id: Optional[str] = None  # defaults to hf_model_id
    github_repo: Optional[str] = None
    requires_custom_code: bool = False

    # Architecture
    architecture: str = "transformer"
    num_parameters: str = "unknown"
    max_context_length: int = 2048
    vocab_size: int = 32000

    # Tokenizer
    padding_side: str = "left"
    use_fast_tokenizer: bool = True
    add_special_tokens: bool = True

    # LoRA fine-tuning targets
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    lora_modules_to_save: Optional[List[str]] = None

    # Generation defaults (can be overridden per experiment)
    default_max_new_tokens: int = 512
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    supports_system_prompt: bool = True

    # Quantization
    supports_4bit: bool = True
    supports_8bit: bool = True
    bnb_compute_dtype: str = "float16"

    # Special flags
    trust_remote_code: bool = False
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True

    # Chat template format
    chat_template: str = "default"  # Options: default, llama, gemma, mistral, chatml

    @property
    def tokenizer_id(self) -> str:
        return self.hf_tokenizer_id or self.hf_model_id


# ============================================================
# Model Configurations
# ============================================================

MODEL_CONFIGS: Dict[ModelName, ModelConfig] = {

    # --------------------------------------------------------
    # DIFFUSION MODELS
    # --------------------------------------------------------

    ModelName.SEDD: ModelConfig(
        name=ModelName.SEDD,
        model_type=ModelType.DIFFUSION,
        display_name="SEDD (Score Entropy Discrete Diffusion)",
        paper_title="Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution",
        paper_year=2024,
        hf_model_id="louaaron/sedd-medium",
        github_repo="https://github.com/louaaron/Score-Entropy-Discrete-Diffusion",
        requires_custom_code=True,
        architecture="transformer_diffusion",
        num_parameters="350M",
        max_context_length=1024,
        vocab_size=32128,
        supports_4bit=False,  # Custom architecture
        supports_8bit=False,
        use_flash_attention=False,
        supports_system_prompt=False,
        chat_template="none",
        lora_target_modules=["attn.qkv", "attn.out_proj"],
        default_temperature=1.0,
        trust_remote_code=True,
    ),

    ModelName.LLADA: ModelConfig(
        name=ModelName.LLADA,
        model_type=ModelType.DIFFUSION,
        display_name="LLaDA (Large Language Diffusion with mAsking)",
        paper_title="Large Language Diffusion Models",
        paper_year=2025,
        hf_model_id="GSAI-ML/LLaDA-8B-Instruct",
        github_repo="https://github.com/GSAI-ML/LLaDA",
        requires_custom_code=True,
        architecture="transformer_masking_diffusion",
        num_parameters="8B",
        max_context_length=4096,
        vocab_size=128256,
        supports_4bit=True,
        use_flash_attention=True,
        supports_system_prompt=True,
        chat_template="llada",
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        trust_remote_code=True,
    ),

    ModelName.D3PM: ModelConfig(
        name=ModelName.D3PM,
        model_type=ModelType.DIFFUSION,
        display_name="D3PM (Discrete Denoising Diffusion Probabilistic Models)",
        paper_title="Structured Denoising Diffusion Models in Discrete State-Spaces",
        paper_year=2021,
        hf_model_id="google/d3pm-absorbing-text8",
        github_repo="https://github.com/google-research/google-research/tree/master/d3pm",
        requires_custom_code=True,
        architecture="transformer_discrete_diffusion",
        num_parameters="110M",
        max_context_length=256,
        vocab_size=27,  # character-level for text8
        supports_4bit=False,
        supports_8bit=False,
        use_flash_attention=False,
        supports_system_prompt=False,
        chat_template="none",
        lora_target_modules=[],
        default_temperature=1.0,
        trust_remote_code=True,
    ),

    ModelName.DIFFUSELM: ModelConfig(
        name=ModelName.DIFFUSELM,
        model_type=ModelType.DIFFUSION,
        display_name="DiffuseLM (Diffusion-LM for Controllable Text Generation)",
        paper_title="Diffusion-LM Improves Controllable Text Generation",
        paper_year=2022,
        hf_model_id="XiangLi1999/Diffusion-LM",
        github_repo="https://github.com/XiangLi1999/Diffusion-LM",
        requires_custom_code=True,
        architecture="continuous_diffusion_lm",
        num_parameters="100M",
        max_context_length=128,
        vocab_size=30522,
        supports_4bit=False,
        supports_8bit=False,
        use_flash_attention=False,
        supports_system_prompt=False,
        chat_template="none",
        lora_target_modules=[],
        default_temperature=1.0,
        trust_remote_code=True,
    ),

    # --------------------------------------------------------
    # AUTO-REGRESSIVE MODELS
    # --------------------------------------------------------

    ModelName.LLAMA: ModelConfig(
        name=ModelName.LLAMA,
        model_type=ModelType.AUTOREGRESSIVE,
        display_name="LLaMA 3 8B Instruct",
        paper_title="LLaMA: Open and Efficient Foundation Language Models",
        paper_year=2024,
        hf_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        architecture="causal_lm",
        num_parameters="8B",
        max_context_length=8192,
        vocab_size=128256,
        supports_4bit=True,
        supports_8bit=True,
        use_flash_attention=True,
        supports_system_prompt=True,
        chat_template="llama",
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        padding_side="left",
    ),

    ModelName.GEMMA: ModelConfig(
        name=ModelName.GEMMA,
        model_type=ModelType.AUTOREGRESSIVE,
        display_name="Gemma 2 9B Instruct",
        paper_title="Gemma: Open Models Based on Gemini Research and Technology",
        paper_year=2024,
        hf_model_id="google/gemma-2-9b-it",
        architecture="causal_lm",
        num_parameters="9B",
        max_context_length=8192,
        vocab_size=256000,
        supports_4bit=True,
        supports_8bit=True,
        use_flash_attention=True,
        supports_system_prompt=True,
        chat_template="gemma",
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        padding_side="left",
    ),

    ModelName.MISTRAL: ModelConfig(
        name=ModelName.MISTRAL,
        model_type=ModelType.AUTOREGRESSIVE,
        display_name="Mistral 7B Instruct v0.3",
        paper_title="Mistral 7B",
        paper_year=2023,
        hf_model_id="mistralai/Mistral-7B-Instruct-v0.3",
        architecture="causal_lm",
        num_parameters="7B",
        max_context_length=32768,
        vocab_size=32768,
        supports_4bit=True,
        supports_8bit=True,
        use_flash_attention=True,
        supports_system_prompt=True,
        chat_template="mistral",
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        padding_side="left",
    ),

    ModelName.BERT: ModelConfig(
        name=ModelName.BERT,
        model_type=ModelType.AUTOREGRESSIVE,
        display_name="IndicBERTv2 (AI4Bharat)",
        paper_title="IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Models for Indian Languages",
        paper_year=2023,
        hf_model_id="ai4bharat/IndicBERTv2-MLM-Sam-TLM",
        architecture="masked_lm",
        num_parameters="278M",
        max_context_length=512,
        vocab_size=128000,
        supports_4bit=False,
        supports_8bit=False,
        use_flash_attention=False,
        supports_system_prompt=False,
        chat_template="none",
        lora_target_modules=["query", "value", "key"],
        padding_side="right",
        default_max_new_tokens=128,
    ),
}


def get_model_config(model_name: ModelName) -> ModelConfig:
    """Get configuration for a specific model."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]


def get_diffusion_models() -> List[ModelConfig]:
    """Get all diffusion model configurations."""
    return [cfg for cfg in MODEL_CONFIGS.values() if cfg.model_type == ModelType.DIFFUSION]


def get_autoregressive_models() -> List[ModelConfig]:
    """Get all auto-regressive model configurations."""
    return [cfg for cfg in MODEL_CONFIGS.values() if cfg.model_type == ModelType.AUTOREGRESSIVE]
