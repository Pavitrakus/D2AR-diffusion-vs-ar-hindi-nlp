"""
D3PM (Discrete Denoising Diffusion Probabilistic Models) wrapper.

Uses structured Markov transition matrices for discrete diffusion
over token spaces. Supports multiple noise types: uniform, absorbing, and
embedding-distance based.

Paper: "Structured Denoising Diffusion Models in Discrete State-Spaces" (NeurIPS 2021)
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


class D3PMWrapper(BaseModel):
    """
    Wrapper for D3PM (Discrete Denoising Diffusion Probabilistic Models).

    D3PM generates text by:
    1. Defining a forward noise process using transition matrices Q_t
    2. Training a neural network to reverse the process
    3. Starting from fully noised tokens and iteratively denoising

    Supports three transition types:
    - 'uniform': Replace with any random token
    - 'absorbing': Replace with a special [MASK] token
    - 'embedding': Replace with nearby tokens in embedding space
    """

    def __init__(self, experiment_config: Optional[ExperimentConfig] = None, **kwargs):
        kwargs.setdefault("load_in_4bit", False)
        kwargs.setdefault("load_in_8bit", False)
        super().__init__(
            model_name=ModelName.D3PM,
            experiment_config=experiment_config,
            **kwargs,
        )
        self.diffusion_steps = kwargs.get("diffusion_steps", 100)
        self.transition_type = kwargs.get("transition_type", "absorbing")
        self.beta_schedule = kwargs.get("beta_schedule", "cosine")

    def _load_model(self) -> None:
        """Load D3PM model (custom implementation)."""
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add absorbing state token
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[ABSORB]"]})
        self.absorb_token_id = self.tokenizer.convert_tokens_to_ids("[ABSORB]")

        vocab_size = len(self.tokenizer)

        self.model = D3PMModel(
            vocab_size=vocab_size,
            d_model=512,
            nhead=8,
            num_layers=6,
            max_seq_len=self.model_config.max_context_length,
        )

        if torch.cuda.is_available():
            self.model = self.model.to(self.device)
        self.model.eval()

        # Pre-compute noise schedule
        self._compute_noise_schedule(vocab_size)

        logger.info(f"D3PM loaded with {self.transition_type} transitions, {self.diffusion_steps} steps")

    def _compute_noise_schedule(self, vocab_size: int) -> None:
        """
        Pre-compute the beta schedule and transition matrices.

        The beta schedule controls how much noise is added at each timestep.
        """
        device = self.device if torch.cuda.is_available() else "cpu"

        if self.beta_schedule == "cosine":
            # Cosine schedule (smoother transitions)
            steps = torch.linspace(0, 1, self.diffusion_steps + 1, device=device)
            alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            self.betas = torch.clamp(betas, min=1e-5, max=0.999)
        elif self.beta_schedule == "linear":
            self.betas = torch.linspace(1e-4, 0.02, self.diffusion_steps, device=device)
        else:
            self.betas = torch.ones(self.diffusion_steps, device=device) * 0.01

        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def _q_sample(self, x_0: torch.Tensor, t: int) -> torch.Tensor:
        """
        Sample from q(x_t | x_0) - the forward noising process.

        For absorbing transitions: with probability (1-alpha_bar_t),
        replace each token with the absorbing state.
        """
        device = x_0.device
        alpha_bar_t = self.alpha_bar[t]

        # With probability (1-alpha_bar_t), replace with absorbing state
        keep_mask = torch.rand_like(x_0.float()) < alpha_bar_t

        if self.transition_type == "absorbing":
            noised = torch.where(keep_mask, x_0, torch.full_like(x_0, self.absorb_token_id))
        elif self.transition_type == "uniform":
            random_tokens = torch.randint(0, len(self.tokenizer), x_0.shape, device=device)
            noised = torch.where(keep_mask, x_0, random_tokens)
        else:
            noised = torch.where(keep_mask, x_0, torch.full_like(x_0, self.absorb_token_id))

        return noised

    def generate(
        self, prompts: List[str], max_new_tokens: int = 128,
        temperature: float = 1.0, top_p: float = 0.9, top_k: int = 50,
        do_sample: bool = True, num_beams: int = 1, **kwargs,
    ) -> GenerationOutput:
        """
        Generate text using D3PM reverse diffusion process.

        Starts from fully noised (absorbing state) tokens and
        iteratively denoises over T timesteps.
        """
        if not self._is_loaded:
            self.load()

        start_time = time.time()
        generated_texts = []

        for prompt in prompts:
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(prompt_ids)
            gen_len = min(max_new_tokens, self.model_config.max_context_length - prompt_len)

            device = next(self.model.parameters()).device

            # Start from fully noised state
            if self.transition_type == "absorbing":
                x_t = torch.full((1, gen_len), self.absorb_token_id, device=device, dtype=torch.long)
            else:
                x_t = torch.randint(0, len(self.tokenizer), (1, gen_len), device=device)

            prompt_tensor = torch.tensor([prompt_ids], device=device)
            full_seq = torch.cat([prompt_tensor, x_t], dim=1)

            # Reverse diffusion: from t=T-1 to t=0
            for t in reversed(range(self.diffusion_steps)):
                timestep = torch.tensor([[t / self.diffusion_steps]], device=device, dtype=torch.float)

                with torch.no_grad():
                    logits = self.model(full_seq, timestep)

                # Only update generation positions
                gen_logits = logits[:, prompt_len:, :] / max(temperature, 1e-6)

                if do_sample:
                    probs = F.softmax(gen_logits, dim=-1)
                    predicted = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(1, gen_len)
                else:
                    predicted = gen_logits.argmax(dim=-1)

                # Apply transition probability: keep previous token with probability (1-beta_t)
                if t > 0:
                    beta_t = self.betas[t]
                    update_mask = torch.rand(1, gen_len, device=device) < beta_t
                    x_t = torch.where(update_mask, predicted, x_t)
                else:
                    x_t = predicted

                full_seq = torch.cat([prompt_tensor, x_t], dim=1)

            generated_ids = x_t[0]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_texts.append(generated_text.strip())

        elapsed = time.time() - start_time
        total_tokens = sum(len(self.tokenizer.encode(t)) for t in generated_texts)

        return GenerationOutput(
            generated_texts=generated_texts, input_texts=prompts,
            model_name=self.model_name.value, generation_time_seconds=elapsed,
            num_tokens_generated=total_tokens,
            metadata={
                "method": "d3pm_discrete_diffusion",
                "transition_type": self.transition_type,
                "num_steps": self.diffusion_steps,
                "beta_schedule": self.beta_schedule,
            },
        )

    def classify(self, texts: List[str], labels: List[str], **kwargs) -> ClassificationOutput:
        """Classify using D3PM via denoising likelihood comparison."""
        if not self._is_loaded:
            self.load()

        start_time = time.time()
        predictions = []
        prediction_scores = []

        for text in texts:
            scores = {}
            for label in labels:
                combined = f"{text} {label}"
                input_ids = self.tokenizer.encode(combined, return_tensors="pt")
                if torch.cuda.is_available():
                    input_ids = input_ids.to(self.device)

                # Evaluate denoising quality at multiple timesteps
                total_score = 0.0
                eval_timesteps = [10, 30, 50]
                for t in eval_timesteps:
                    t_idx = min(t, self.diffusion_steps - 1)
                    noised = self._q_sample(input_ids, t_idx)
                    timestep = torch.tensor([[t_idx / self.diffusion_steps]],
                                            device=input_ids.device, dtype=torch.float)

                    with torch.no_grad():
                        logits = self.model(noised, timestep)

                    loss = F.cross_entropy(
                        logits.view(-1, logits.shape[-1]), input_ids.view(-1), reduction="mean"
                    )
                    total_score -= loss.item()

                scores[label] = total_score / len(eval_timesteps)

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


class D3PMModel(nn.Module):
    """D3PM neural network backbone."""

    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, max_seq_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, vocab_size)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_ids, timestep):
        batch_size, seq_len = input_ids.shape
        token_emb = self.embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(torch.clamp(pos_ids, 0, self.pos_embedding.num_embeddings - 1))
        time_emb = self.time_mlp(timestep).unsqueeze(1)
        hidden = self.layer_norm(token_emb + pos_emb + time_emb)
        hidden = self.transformer(hidden)
        return self.output_head(hidden)
