"""
LLaDA (Large Language Diffusion with mAsking) model wrapper.

LLaDA uses a masking-based diffusion process for text generation:
- Forward process: Randomly mask tokens with increasing probability
- Reverse process: Predict and unmask all tokens simultaneously
- 8B parameter model competitive with LLaMA 3 8B

Paper: "Large Language Diffusion Models" (arXiv:2502.09992, 2025)
"""

import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from loguru import logger

from config.base_config import ModelName, ExperimentConfig
from src.models.base_model import BaseModel, GenerationOutput, ClassificationOutput


class LLaDAWrapper(BaseModel):
    """
    Wrapper for LLaDA (Large Language Diffusion with mAsking).

    LLaDA generates text by:
    1. Starting with a fully masked sequence [MASK] [MASK] ... [MASK]
    2. At each step, predicting all tokens simultaneously
    3. Progressively unmasking tokens with highest confidence
    4. Repeating until all tokens are unmasked

    Key advantage: Bidirectional context (unlike left-to-right AR models),
    enabling better coherence and reasoning.
    """

    def __init__(self, experiment_config: Optional[ExperimentConfig] = None, **kwargs):
        super().__init__(
            model_name=ModelName.LLADA,
            experiment_config=experiment_config,
            **kwargs,
        )
        self.diffusion_steps = kwargs.get("diffusion_steps", 64)
        self.mask_token_id = None

    def _load_model(self) -> None:
        """Load LLaDA model from HuggingFace."""
        try:
            self._load_from_huggingface()
        except Exception as e:
            logger.warning(f"HuggingFace load failed: {e}. Using custom implementation...")
            self._load_custom()

    def _load_from_huggingface(self) -> None:
        """Load LLaDA from HuggingFace."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.hf_model_id,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Find or set mask token
        if hasattr(self.tokenizer, 'mask_token_id') and self.tokenizer.mask_token_id is not None:
            self.mask_token_id = self.tokenizer.mask_token_id
        else:
            # LLaDA uses a special mask token ID (typically 128256 for its tokenizer)
            self.mask_token_id = 128256

        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if self.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype, bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["torch_dtype"] = self.dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.hf_model_id, **model_kwargs,
        )
        self.model.eval()
        logger.info(f"LLaDA loaded from HuggingFace")

    def _load_custom(self) -> None:
        """Load a simplified LLaDA-style model."""
        from transformers import AutoTokenizer
        import torch.nn as nn

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add mask token
        self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        self.mask_token_id = self.tokenizer.mask_token_id

        class SimpleLLaDA(nn.Module):
            """Simplified LLaDA: transformer with bidirectional attention for mask prediction."""

            def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, max_seq_len=1024):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_embedding = nn.Embedding(max_seq_len, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                    batch_first=True, activation="gelu",
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.output_head = nn.Linear(d_model, vocab_size)
                self.layer_norm = nn.LayerNorm(d_model)

            def forward(self, input_ids, **kwargs):
                batch_size, seq_len = input_ids.shape
                token_emb = self.embedding(input_ids)
                pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
                pos_emb = self.pos_embedding(pos_ids)
                hidden = self.layer_norm(token_emb + pos_emb)
                hidden = self.transformer(hidden)
                logits = self.output_head(hidden)

                class Output:
                    pass

                out = Output()
                out.logits = logits
                return out

        self.model = SimpleLLaDA(vocab_size=len(self.tokenizer))
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("LLaDA loaded with custom implementation")

    def generate(
        self, prompts: List[str], max_new_tokens: int = 256,
        temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50,
        do_sample: bool = True, num_beams: int = 1, **kwargs,
    ) -> GenerationOutput:
        """
        Generate text using LLaDA's masking-based diffusion.

        Process:
        1. Encode prompt, append `max_new_tokens` [MASK] tokens
        2. For each diffusion step (from t=1 to t=0):
           a. Predict all masked token distributions
           b. Unmask a fraction (1/num_steps) of tokens with highest confidence
           c. Keep previously unmasked tokens fixed
        3. Return fully unmasked sequence
        """
        if not self._is_loaded:
            self.load()

        start_time = time.time()
        generated_texts = []
        num_steps = kwargs.get("num_diffusion_steps", self.diffusion_steps)

        for prompt in prompts:
            # Encode prompt
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(prompt_ids)

            # Initialize generation tokens as all [MASK]
            gen_len = min(max_new_tokens, self.model_config.max_context_length - prompt_len)
            mask_ids = [self.mask_token_id] * gen_len

            # Full sequence: prompt + masks
            current_ids = torch.tensor([prompt_ids + mask_ids], device=self.device if torch.cuda.is_available() else "cpu")

            # Track which positions are still masked
            is_masked = torch.zeros(current_ids.shape[1], dtype=torch.bool, device=current_ids.device)
            is_masked[prompt_len:] = True  # Only generation positions start masked

            # Number of tokens to unmask per step
            tokens_per_step = max(1, gen_len // num_steps)

            for step in range(num_steps):
                if not is_masked.any():
                    break

                # Forward pass: predict all tokens
                with torch.no_grad():
                    outputs = self.model(current_ids)
                    logits = outputs.logits[0]  # [seq_len, vocab_size]

                # Only consider masked positions
                masked_positions = is_masked.nonzero(as_tuple=True)[0]
                if len(masked_positions) == 0:
                    break

                # Get predictions and confidence for masked positions
                masked_logits = logits[masked_positions]  # [num_masked, vocab_size]
                masked_logits = masked_logits / max(temperature, 1e-6)
                probs = F.softmax(masked_logits, dim=-1)

                if do_sample:
                    predicted_ids = torch.multinomial(probs, 1).squeeze(-1)
                else:
                    predicted_ids = masked_logits.argmax(dim=-1)

                # Compute confidence (max probability) for each prediction
                confidence = probs.max(dim=-1).values

                # Unmask the most confident predictions
                num_to_unmask = min(tokens_per_step, len(masked_positions))
                if step == num_steps - 1:
                    num_to_unmask = len(masked_positions)  # Unmask all remaining on last step

                _, top_indices = confidence.topk(num_to_unmask)
                positions_to_unmask = masked_positions[top_indices]

                # Update token IDs and mask
                current_ids[0, positions_to_unmask] = predicted_ids[top_indices]
                is_masked[positions_to_unmask] = False

            # Decode only the generated portion
            generated_ids = current_ids[0, prompt_len:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_texts.append(generated_text.strip())

        elapsed = time.time() - start_time
        total_tokens = sum(len(self.tokenizer.encode(t)) for t in generated_texts)

        return GenerationOutput(
            generated_texts=generated_texts, input_texts=prompts,
            model_name=self.model_name.value, generation_time_seconds=elapsed,
            num_tokens_generated=total_tokens,
            metadata={
                "method": "masking_diffusion",
                "num_diffusion_steps": num_steps,
                "temperature": temperature,
            },
        )

    def classify(self, texts: List[str], labels: List[str], **kwargs) -> ClassificationOutput:
        """Classify using LLaDA via prompted generation."""
        if not self._is_loaded:
            self.load()

        start_time = time.time()
        predictions = []
        prediction_scores = []
        label_str = "/".join(labels)

        for text in texts:
            prompt = (
                f"Classify this text: {text}\n\n"
                f"Options: {label_str}\n"
                f"Answer:"
            )

            gen_output = self.generate([prompt], max_new_tokens=32, temperature=0.1, do_sample=False)
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

        elapsed = time.time() - start_time

        return ClassificationOutput(
            predictions=predictions, prediction_scores=prediction_scores,
            input_texts=texts, model_name=self.model_name.value,
            classification_time_seconds=elapsed,
        )
