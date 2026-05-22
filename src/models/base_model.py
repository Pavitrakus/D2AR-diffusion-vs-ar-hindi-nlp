"""
Abstract base class for all language models.

Provides a unified interface for both diffusion and auto-regressive models,
enabling consistent experiment execution across all 8 models.
"""

import abc
import json
import time
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from loguru import logger

from config.base_config import ModelName, ExperimentConfig
from config.model_configs import ModelConfig, get_model_config


@dataclass
class GenerationOutput:
    """Standardized output from model generation."""
    generated_texts: List[str]
    input_texts: List[str]
    model_name: str
    generation_time_seconds: float
    num_tokens_generated: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_texts": self.generated_texts,
            "input_texts": self.input_texts,
            "model_name": self.model_name,
            "generation_time_seconds": self.generation_time_seconds,
            "num_tokens_generated": self.num_tokens_generated,
            "metadata": self.metadata,
        }

    def save(self, path: Union[str, Path]):
        """Save generation output to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Saved generation output to {path}")


@dataclass
class ClassificationOutput:
    """Standardized output from model classification."""
    predictions: List[str]
    prediction_scores: List[Dict[str, float]]
    input_texts: List[str]
    model_name: str
    classification_time_seconds: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predictions": self.predictions,
            "prediction_scores": self.prediction_scores,
            "input_texts": self.input_texts,
            "model_name": self.model_name,
            "classification_time_seconds": self.classification_time_seconds,
            "metadata": self.metadata,
        }

    def save(self, path: Union[str, Path]):
        """Save classification output to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Saved classification output to {path}")


class BaseModel(abc.ABC):
    """
    Abstract base class for all language models.

    All 8 models (4 diffusion + 4 auto-regressive) implement this interface,
    ensuring consistent experiment execution.

    Subclasses must implement:
        - _load_model(): Load model weights and tokenizer
        - generate(): Generate text from prompts
        - classify(): Classify input text (for legal tasks)
        - get_model_info(): Return model metadata
    """

    def __init__(
        self,
        model_name: ModelName,
        experiment_config: Optional[ExperimentConfig] = None,
        device: str = "cuda",
        dtype: str = "float16",
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
    ):
        self.model_name = model_name
        self.model_config: ModelConfig = get_model_config(model_name)
        self.experiment_config = experiment_config
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.load_in_4bit = load_in_4bit and self.model_config.supports_4bit
        self.load_in_8bit = load_in_8bit and self.model_config.supports_8bit

        self.model = None
        self.tokenizer = None
        self._is_loaded = False

        logger.info(
            f"Initialized {self.model_config.display_name} wrapper "
            f"(device={device}, dtype={dtype}, 4bit={self.load_in_4bit})"
        )

    @abc.abstractmethod
    def _load_model(self) -> None:
        """Load model weights, tokenizer, and any required components."""
        pass

    def load(self) -> "BaseModel":
        """Load the model (with error handling and logging)."""
        if self._is_loaded:
            logger.info(f"{self.model_name.value} already loaded, skipping...")
            return self

        logger.info(f"Loading {self.model_config.display_name}...")
        start_time = time.time()

        try:
            self._load_model()
            self._is_loaded = True
            elapsed = time.time() - start_time
            logger.success(
                f"Loaded {self.model_config.display_name} in {elapsed:.1f}s"
            )
        except Exception as e:
            logger.error(f"Failed to load {self.model_config.display_name}: {e}")
            raise

        return self

    def unload(self) -> None:
        """Unload model to free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Unloaded {self.model_name.value}")

    @abc.abstractmethod
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_beams: int = 1,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate text from a list of prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling (vs greedy)
            num_beams: Number of beams for beam search

        Returns:
            GenerationOutput with generated texts
        """
        pass

    @abc.abstractmethod
    def classify(
        self,
        texts: List[str],
        labels: List[str],
        **kwargs,
    ) -> ClassificationOutput:
        """
        Classify input texts into given labels.

        For classification tasks (bail prediction, verdict prediction),
        this method determines the model's predicted class.

        Args:
            texts: List of input texts
            labels: List of possible class labels

        Returns:
            ClassificationOutput with predictions
        """
        pass

    def format_prompt(
        self,
        user_message: str,
        system_message: Optional[str] = None,
    ) -> str:
        """
        Format a prompt according to the model's chat template.

        Args:
            user_message: The user's input
            system_message: Optional system prompt

        Returns:
            Formatted prompt string
        """
        if not self.model_config.supports_system_prompt or system_message is None:
            return user_message

        template = self.model_config.chat_template

        if template == "llama":
            return (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_message}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_message}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        elif template == "gemma":
            return (
                f"<start_of_turn>user\n"
                f"{system_message}\n\n{user_message}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
        elif template == "mistral":
            return (
                f"<s>[INST] {system_message}\n\n{user_message} [/INST]"
            )
        elif template == "chatml":
            return (
                f"<|im_start|>system\n{system_message}<|im_end|>\n"
                f"<|im_start|>user\n{user_message}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            # Generic format
            return f"{system_message}\n\n{user_message}"

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "model_name": self.model_name.value,
            "display_name": self.model_config.display_name,
            "model_type": self.model_config.model_type.value,
            "hf_model_id": self.model_config.hf_model_id,
            "num_parameters": self.model_config.num_parameters,
            "architecture": self.model_config.architecture,
            "paper_title": self.model_config.paper_title,
            "paper_year": self.model_config.paper_year,
            "is_loaded": self._is_loaded,
            "device": self.device,
            "dtype": str(self.dtype),
            "quantization": "4bit" if self.load_in_4bit else ("8bit" if self.load_in_8bit else "none"),
        }

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return f"<{self.__class__.__name__}({self.model_name.value}, {status})>"

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False
