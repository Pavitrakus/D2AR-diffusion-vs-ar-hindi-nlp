"""
DiffuseLM (Diffusion-LM) model wrapper.

Uses continuous diffusion in embedding space for controllable text generation.
Unlike discrete diffusion models (SEDD, D3PM), DiffuseLM operates in continuous
space by mapping tokens to embeddings, applying Gaussian noise, and denoising.

Paper: "Diffusion-LM Improves Controllable Text Generation" (NeurIPS 2022)
"""

import time
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from config.base_config import ModelName, ExperimentConfig
from src.models.base_model import BaseModel, GenerationOutput, ClassificationOutput


class DiffuseLMWrapper(BaseModel):
    """
    Wrapper for DiffuseLM (Diffusion-LM).

    Key differences from discrete diffusion:
    - Operates in continuous embedding space, not discrete token space
    - Uses Gaussian noise (like image diffusion) on word embeddings
    - Applies "rounding" step to map continuous vectors back to discrete tokens
    - Supports gradient-based controllable generation via plug-and-play classifiers
    """

    def __init__(self, experiment_config: Optional[ExperimentConfig] = None, **kwargs):
        kwargs.setdefault("load_in_4bit", False)
        kwargs.setdefault("load_in_8bit", False)
        super().__init__(
            model_name=ModelName.DIFFUSELM,
            experiment_config=experiment_config,
            **kwargs,
        )
        self.diffusion_steps = kwargs.get("diffusion_steps", 200)
        self.embedding_dim = kwargs.get("embedding_dim", 128)

    def _load_model(self) -> None:
        """Load DiffuseLM model (custom implementation)."""
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        vocab_size = len(self.tokenizer)

        self.model = DiffuseLMModel(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            d_model=512,
            nhead=8,
            num_layers=6,
            max_seq_len=self.model_config.max_context_length,
        )

        if torch.cuda.is_available():
            self.model = self.model.to(self.device)
        self.model.eval()

        # Pre-compute noise schedule (similar to DDPM)
        self._compute_noise_schedule()

        logger.info(f"DiffuseLM loaded with continuous diffusion in {self.embedding_dim}D space")

    def _compute_noise_schedule(self) -> None:
        """Compute the Gaussian noise schedule."""
        device = self.device if torch.cuda.is_available() else "cpu"

        # Cosine schedule
        steps = torch.linspace(0, 1, self.diffusion_steps + 1, device=device)
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]

        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

    def _q_sample(self, x_0: torch.Tensor, t: int) -> tuple:
        """
        Forward process: Add Gaussian noise to embeddings.

        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        Args:
            x_0: [batch, seq_len, embedding_dim] - clean embeddings
            t: timestep index

        Returns:
            x_t: noised embeddings
            noise: the noise that was added
        """
        sqrt_ab = self.sqrt_alpha_bar[t]
        sqrt_1_ab = self.sqrt_one_minus_alpha_bar[t]

        noise = torch.randn_like(x_0)
        x_t = sqrt_ab * x_0 + sqrt_1_ab * noise

        return x_t, noise

    def _rounding_step(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Round continuous embeddings to nearest discrete tokens.

        Maps each embedding vector to the closest word embedding in the
        model's embedding table using cosine similarity.

        Args:
            embeddings: [batch, seq_len, embedding_dim] - continuous embeddings

        Returns:
            token_ids: [batch, seq_len] - discrete token IDs
        """
        # Get embedding table
        word_embeddings = self.model.embedding.weight  # [vocab_size, embedding_dim]

        # Normalize for cosine similarity
        emb_norm = F.normalize(embeddings, dim=-1)
        vocab_norm = F.normalize(word_embeddings, dim=-1)

        # Compute similarity: [batch, seq_len, vocab_size]
        similarity = torch.matmul(emb_norm, vocab_norm.T)

        # Argmax to get closest token
        token_ids = similarity.argmax(dim=-1)

        return token_ids

    def generate(
        self, prompts: List[str], max_new_tokens: int = 128,
        temperature: float = 1.0, top_p: float = 0.9, top_k: int = 50,
        do_sample: bool = True, num_beams: int = 1, **kwargs,
    ) -> GenerationOutput:
        """
        Generate text using continuous diffusion in embedding space.

        Process:
        1. Start with random Gaussian noise in embedding space
        2. Iteratively denoise using the learned model
        3. At each step, predict the noise and subtract it
        4. Final step: round continuous embeddings to discrete tokens
        """
        if not self._is_loaded:
            self.load()

        start_time = time.time()
        generated_texts = []
        device = next(self.model.parameters()).device

        for prompt in prompts:
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(prompt_ids)
            gen_len = min(max_new_tokens, self.model_config.max_context_length - prompt_len)

            # Get prompt embeddings
            prompt_tensor = torch.tensor([prompt_ids], device=device)
            prompt_emb = self.model.embedding(prompt_tensor)  # [1, prompt_len, emb_dim]

            # Start with random Gaussian noise for generation portion
            x_t = torch.randn(1, gen_len, self.embedding_dim, device=device)

            # Concatenate prompt embeddings with noisy generation embeddings
            # During denoising, prompt embeddings stay fixed

            # Reverse diffusion: from t=T to t=0
            for t in reversed(range(self.diffusion_steps)):
                # Full sequence: prompt embeddings + current noisy embeddings
                full_emb = torch.cat([prompt_emb, x_t], dim=1)
                timestep = torch.tensor([[t / self.diffusion_steps]],
                                        device=device, dtype=torch.float)

                with torch.no_grad():
                    predicted_noise = self.model.denoise(full_emb, timestep)

                # Only update the generation portion
                gen_noise = predicted_noise[:, prompt_len:, :]

                # DDPM denoising step
                if t > 0:
                    alpha_t = self.alpha_bar[t]
                    alpha_t_minus_1 = self.alpha_bar[t - 1]

                    # Compute predicted x_0
                    x_0_pred = (x_t - self.sqrt_one_minus_alpha_bar[t] * gen_noise) / self.sqrt_alpha_bar[t]
                    x_0_pred = torch.clamp(x_0_pred, -5, 5)  # Stability

                    # Compute posterior mean
                    beta_t = 1 - alpha_t / alpha_t_minus_1
                    posterior_mean = (
                        torch.sqrt(alpha_t_minus_1) * beta_t / (1 - alpha_t) * x_0_pred +
                        torch.sqrt(alpha_t / alpha_t_minus_1) * (1 - alpha_t_minus_1) / (1 - alpha_t) * x_t
                    )

                    # Add noise (except at t=1)
                    if t > 1:
                        noise = torch.randn_like(x_t) * math.sqrt(beta_t) * temperature
                        x_t = posterior_mean + noise
                    else:
                        x_t = posterior_mean
                else:
                    # Final step: directly predict x_0
                    x_t = (x_t - self.sqrt_one_minus_alpha_bar[0] * gen_noise) / self.sqrt_alpha_bar[0]

            # Round continuous embeddings to discrete tokens
            generated_ids = self._rounding_step(x_t)
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            generated_texts.append(generated_text.strip())

        elapsed = time.time() - start_time
        total_tokens = sum(len(self.tokenizer.encode(t)) for t in generated_texts)

        return GenerationOutput(
            generated_texts=generated_texts, input_texts=prompts,
            model_name=self.model_name.value, generation_time_seconds=elapsed,
            num_tokens_generated=total_tokens,
            metadata={
                "method": "continuous_embedding_diffusion",
                "num_diffusion_steps": self.diffusion_steps,
                "embedding_dim": self.embedding_dim,
                "rounding": "cosine_similarity_argmax",
            },
        )

    def classify(self, texts: List[str], labels: List[str], **kwargs) -> ClassificationOutput:
        """Classify using DiffuseLM via embedding-space label scoring."""
        if not self._is_loaded:
            self.load()

        start_time = time.time()
        predictions = []
        prediction_scores = []
        device = next(self.model.parameters()).device

        for text in texts:
            scores = {}
            for label in labels:
                combined = f"{text} {label}"
                input_ids = self.tokenizer.encode(combined, return_tensors="pt").to(device)
                input_emb = self.model.embedding(input_ids)

                # Evaluate reconstruction quality at multiple noise levels
                total_score = 0.0
                for t in [20, 50, 100]:
                    t_idx = min(t, self.diffusion_steps - 1)
                    x_t, noise = self._q_sample(input_emb, t_idx)
                    timestep = torch.tensor([[t_idx / self.diffusion_steps]],
                                            device=device, dtype=torch.float)

                    with torch.no_grad():
                        predicted_noise = self.model.denoise(x_t, timestep)

                    # MSE between predicted and actual noise (lower = better)
                    mse = F.mse_loss(predicted_noise, noise, reduction="mean")
                    total_score -= mse.item()

                scores[label] = total_score / 3

            max_score = max(scores.values())
            exp_scores = {k: torch.exp(torch.tensor(v - max_score)).item() for k, v in scores.items()}
            total = sum(exp_scores.values())
            normalized = {k: v / total for k, v in exp_scores.items()}

            predictions.append(max(normalized, key=normalized.get))
            prediction_scores.append(normalized)

        return ClassificationOutput(
            predictions=predictions, prediction_scores=prediction_scores,
            input_texts=texts, model_name=self.model_name.value,
            classification_time_seconds=time.time() - start_time,
        )


class DiffuseLMModel(nn.Module):
    """
    DiffuseLM neural network: denoising model in continuous embedding space.

    Architecture:
    - Token → embedding (shared embedding for input and output)
    - Transformer encoder (bidirectional)
    - Noise prediction head
    """

    def __init__(self, vocab_size, embedding_dim=128, d_model=512,
                 nhead=8, num_layers=6, max_seq_len=128):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_proj = nn.Linear(embedding_dim, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, embedding_dim)
        self.layer_norm = nn.LayerNorm(d_model)

    def denoise(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Predict the noise added to embeddings.

        Args:
            x_t: [batch, seq_len, embedding_dim] - noised embeddings
            timestep: [batch, 1] - diffusion timestep

        Returns:
            predicted_noise: [batch, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = x_t.shape

        hidden = self.input_proj(x_t)  # [batch, seq_len, d_model]
        pos_ids = torch.arange(seq_len, device=x_t.device).unsqueeze(0).expand(batch_size, -1)
        pos_ids = torch.clamp(pos_ids, 0, self.pos_embedding.num_embeddings - 1)
        pos_emb = self.pos_embedding(pos_ids)
        time_emb = self.time_mlp(timestep).unsqueeze(1)

        hidden = self.layer_norm(hidden + pos_emb + time_emb)
        hidden = self.transformer(hidden)
        predicted_noise = self.output_proj(hidden)

        return predicted_noise
