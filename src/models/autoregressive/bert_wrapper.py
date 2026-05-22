"""
IndicBERT v2 model wrapper.

AI4Bharat's multilingual BERT model pre-trained on Indian languages.
Primarily suited for classification tasks; adapted for generation via
masked language model sampling.
"""

import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from loguru import logger

from config.base_config import ModelName, ExperimentConfig
from src.models.base_model import BaseModel, GenerationOutput, ClassificationOutput


class BERTWrapper(BaseModel):
    """Wrapper for AI4Bharat IndicBERTv2."""

    def __init__(self, experiment_config: Optional[ExperimentConfig] = None, **kwargs):
        # BERT doesn't support 4-bit/8-bit quantization in the standard sense
        kwargs.setdefault("load_in_4bit", False)
        kwargs.setdefault("load_in_8bit", False)
        super().__init__(
            model_name=ModelName.BERT,
            experiment_config=experiment_config,
            **kwargs,
        )
        self.classification_head = None

    def _load_model(self) -> None:
        """Load IndicBERT model."""
        from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.hf_model_id,
            use_fast=self.model_config.use_fast_tokenizer,
            padding_side=self.model_config.padding_side,
        )

        # Load as Masked LM for generation tasks
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_config.hf_model_id,
            torch_dtype=self.dtype,
        )

        if torch.cuda.is_available():
            self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"IndicBERT loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def _load_classification_model(self, num_labels: int) -> None:
        """Load IndicBERT with a sequence classification head."""
        from transformers import AutoModelForSequenceClassification

        self.classification_head = AutoModelForSequenceClassification.from_pretrained(
            self.model_config.hf_model_id,
            num_labels=num_labels,
            torch_dtype=self.dtype,
        )
        if torch.cuda.is_available():
            self.classification_head = self.classification_head.to(self.device)
        self.classification_head.eval()

    def generate(
        self, prompts: List[str], max_new_tokens: int = 128,
        temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50,
        do_sample: bool = True, num_beams: int = 1, **kwargs,
    ) -> GenerationOutput:
        """
        Generate text using iterative MASK prediction.

        Since BERT is not an auto-regressive model, we use iterative masked
        token prediction: append [MASK] tokens and predict them one by one.
        This is a best-effort generation approach for BERT-style models.
        """
        if not self._is_loaded:
            self.load()

        start_time = time.time()
        generated_texts = []
        mask_token = self.tokenizer.mask_token
        mask_token_id = self.tokenizer.mask_token_id

        for prompt in prompts:
            current_text = prompt
            generated_tokens = []

            for step in range(min(max_new_tokens, 64)):  # Limit for BERT
                masked_input = current_text + " " + mask_token
                inputs = self.tokenizer(
                    masked_input, return_tensors="pt",
                    truncation=True, max_length=self.model_config.max_context_length,
                ).to(self.model.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits

                # Find the [MASK] position and predict
                mask_positions = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)
                if len(mask_positions[0]) == 0:
                    break

                mask_logits = logits[mask_positions[0][-1], mask_positions[1][-1]]

                if do_sample:
                    probs = F.softmax(mask_logits / temperature, dim=-1)
                    predicted_id = torch.multinomial(probs, 1).item()
                else:
                    predicted_id = mask_logits.argmax().item()

                predicted_token = self.tokenizer.decode([predicted_id]).strip()

                # Stop at special tokens or end-of-sentence
                if predicted_token in ["[SEP]", "[CLS]", "[PAD]", "", "।", "."]:
                    if generated_tokens:
                        break

                generated_tokens.append(predicted_token)
                current_text = current_text + " " + predicted_token

            generated_text = " ".join(generated_tokens)
            generated_texts.append(generated_text.strip())

        elapsed = time.time() - start_time

        return GenerationOutput(
            generated_texts=generated_texts, input_texts=prompts,
            model_name=self.model_name.value, generation_time_seconds=elapsed,
            num_tokens_generated=sum(len(t.split()) for t in generated_texts),
            metadata={"method": "iterative_mask_prediction", "temperature": temperature},
        )

    def classify(self, texts: List[str], labels: List[str], **kwargs) -> ClassificationOutput:
        """
        Classify text using IndicBERT.

        Uses the MLM head to score each label token likelihood for the input,
        rather than requiring a fine-tuned classification head.
        """
        if not self._is_loaded:
            self.load()

        start_time = time.time()
        predictions = []
        prediction_scores = []
        mask_token = self.tokenizer.mask_token
        mask_token_id = self.tokenizer.mask_token_id

        for text in texts:
            # Create a cloze-style prompt: "This text is about [MASK]"
            prompt = f"{text} यह मामला {mask_token} है।"

            inputs = self.tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=self.model_config.max_context_length,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            mask_positions = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)
            if len(mask_positions[0]) == 0:
                predictions.append(labels[0])
                prediction_scores.append({label: 0.0 for label in labels})
                continue

            mask_logits = logits[mask_positions[0][0], mask_positions[1][0]]
            probs = F.softmax(mask_logits, dim=-1)

            # Score each label
            scores = {}
            for label in labels:
                label_tokens = self.tokenizer.encode(label, add_special_tokens=False)
                if label_tokens:
                    label_prob = probs[label_tokens[0]].item()
                else:
                    label_prob = 0.0
                scores[label] = label_prob

            best_label = max(scores, key=scores.get)
            predictions.append(best_label)
            prediction_scores.append(scores)

        elapsed = time.time() - start_time

        return ClassificationOutput(
            predictions=predictions, prediction_scores=prediction_scores,
            input_texts=texts, model_name=self.model_name.value,
            classification_time_seconds=elapsed,
            metadata={"method": "mlm_cloze_scoring"},
        )
