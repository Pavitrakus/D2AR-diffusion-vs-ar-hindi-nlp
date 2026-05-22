"""
Mistral 7B Instruct model wrapper.

Mistral AI's efficient model with sliding window attention.
"""

import time
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from config.base_config import ModelName, ExperimentConfig
from src.models.base_model import BaseModel, GenerationOutput, ClassificationOutput


class MistralWrapper(BaseModel):
    """Wrapper for Mistral 7B Instruct v0.3."""

    def __init__(self, experiment_config: Optional[ExperimentConfig] = None, **kwargs):
        super().__init__(
            model_name=ModelName.MISTRAL,
            experiment_config=experiment_config,
            **kwargs,
        )

    def _load_model(self) -> None:
        """Load Mistral model with quantization."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.hf_model_id,
            use_fast=self.model_config.use_fast_tokenizer,
            padding_side=self.model_config.padding_side,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model_kwargs = {"device_map": "auto", "trust_remote_code": self.model_config.trust_remote_code}

        if self.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype, bnb_4bit_use_double_quant=True,
            )
        elif self.load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["torch_dtype"] = self.dtype

        if self.model_config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(self.model_config.hf_model_id, **model_kwargs)
        self.model.eval()
        logger.info(f"Mistral loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def generate(
        self, prompts: List[str], max_new_tokens: int = 512,
        temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50,
        do_sample: bool = True, num_beams: int = 1, **kwargs,
    ) -> GenerationOutput:
        """Generate text using Mistral."""
        if not self._is_loaded:
            self.load()

        start_time = time.time()
        generated_texts = []

        for prompt in prompts:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True,
                max_length=self.model_config.max_context_length - max_new_tokens,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else 1.0,
                    top_p=top_p if do_sample else 1.0,
                    top_k=top_k if do_sample else 0,
                    do_sample=do_sample, num_beams=num_beams,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            input_length = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_texts.append(generated_text.strip())

        elapsed = time.time() - start_time
        total_tokens = sum(len(self.tokenizer.encode(t)) for t in generated_texts)

        return GenerationOutput(
            generated_texts=generated_texts, input_texts=prompts,
            model_name=self.model_name.value, generation_time_seconds=elapsed,
            num_tokens_generated=total_tokens,
            metadata={"temperature": temperature, "top_p": top_p},
        )

    def classify(self, texts: List[str], labels: List[str], **kwargs) -> ClassificationOutput:
        """Classify text using Mistral via generation."""
        if not self._is_loaded:
            self.load()

        start_time = time.time()
        predictions, prediction_scores = [], []
        label_str = "/".join(labels)

        for text in texts:
            prompt = (
                f"Classify the following text into one of these categories: {label_str}.\n\n"
                f"Text: {text}\n\nClassification (output ONLY the category name):"
            )
            formatted = self.format_prompt(prompt)
            gen_output = self.generate([formatted], max_new_tokens=32, temperature=0.1, do_sample=False)
            response = gen_output.generated_texts[0].strip().upper()

            best_label = labels[0]
            scores = {}
            for label in labels:
                score = 1.0 if label.upper() in response else 0.0
                scores[label] = score
                if label.upper() in response:
                    best_label = label

            predictions.append(best_label)
            prediction_scores.append(scores)

        return ClassificationOutput(
            predictions=predictions, prediction_scores=prediction_scores,
            input_texts=texts, model_name=self.model_name.value,
            classification_time_seconds=time.time() - start_time,
        )
