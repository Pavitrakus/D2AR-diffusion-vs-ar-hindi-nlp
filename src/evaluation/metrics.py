"""
Comprehensive evaluation metrics for the research project.

Supports both generation metrics (BLEU, ROUGE, BERTScore, Perplexity)
and classification metrics (Accuracy, F1, Precision, Recall, AUC-ROC).

All metrics return standardized dictionaries for easy comparison and plotting.
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class MetricResult:
    """Standardized result from metric computation."""
    metric_name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "score": self.score,
            "details": self.details,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation results for one experiment."""
    model_name: str
    task_name: str
    pipeline_step: str
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    num_samples: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def primary_score(self) -> float:
        """Return the primary metric score."""
        if self.metrics:
            return list(self.metrics.values())[0].score
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task_name": self.task_name,
            "pipeline_step": self.pipeline_step,
            "num_samples": self.num_samples,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "metadata": self.metadata,
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Saved evaluation results to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EvaluationResult":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        result = cls(
            model_name=data["model_name"],
            task_name=data["task_name"],
            pipeline_step=data["pipeline_step"],
            num_samples=data.get("num_samples", 0),
            metadata=data.get("metadata", {}),
        )
        for k, v in data.get("metrics", {}).items():
            result.metrics[k] = MetricResult(**v)
        return result


# ============================================================
# GENERATION METRICS
# ============================================================

def compute_bleu(
    predictions: List[str],
    references: List[str],
    tokenize: str = "intl",
) -> MetricResult:
    """
    Compute BLEU score using sacrebleu.

    BLEU measures n-gram precision between generated and reference texts.
    Standard metric for machine translation quality.
    """
    try:
        import sacrebleu

        # sacrebleu expects references as list of list
        refs = [[ref] for ref in references]
        bleu = sacrebleu.corpus_bleu(predictions, list(zip(*refs)), tokenize=tokenize)

        return MetricResult(
            metric_name="BLEU",
            score=bleu.score,
            details={
                "bleu": bleu.score,
                "precisions": list(bleu.precisions),
                "brevity_penalty": bleu.bp,
                "sys_len": bleu.sys_len,
                "ref_len": bleu.ref_len,
            },
        )
    except ImportError:
        logger.warning("sacrebleu not installed, using nltk BLEU")
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

        tokenized_preds = [p.split() for p in predictions]
        tokenized_refs = [[r.split()] for r in references]
        smooth = SmoothingFunction().method1
        score = corpus_bleu(tokenized_refs, tokenized_preds, smoothing_function=smooth) * 100

        return MetricResult(metric_name="BLEU", score=score, details={"method": "nltk"})


def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, MetricResult]:
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).

    Standard metric for summarization quality.
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            result = scorer.score(ref, pred)
            for key in scores:
                scores[key].append(result[key].fmeasure)

        results = {}
        for key in scores:
            avg = np.mean(scores[key]) * 100
            results[key] = MetricResult(
                metric_name=key.upper(),
                score=avg,
                details={
                    "mean": avg,
                    "std": np.std(scores[key]) * 100,
                    "min": np.min(scores[key]) * 100,
                    "max": np.max(scores[key]) * 100,
                },
            )
        return results
    except ImportError:
        logger.warning("rouge_score not installed")
        return {"rouge1": MetricResult("ROUGE-1", 0.0), "rouge2": MetricResult("ROUGE-2", 0.0),
                "rougeL": MetricResult("ROUGE-L", 0.0)}


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    lang: str = "hi",
    model_type: str = "bert-base-multilingual-cased",
) -> MetricResult:
    """
    Compute BERTScore for semantic similarity evaluation.

    Uses contextual embeddings to compare semantic similarity
    between predictions and references.
    """
    try:
        from bert_score import score as bert_score_fn

        P, R, F1 = bert_score_fn(
            predictions, references,
            lang=lang,
            model_type=model_type,
            verbose=False,
        )

        f1_mean = F1.mean().item() * 100

        return MetricResult(
            metric_name="BERTScore",
            score=f1_mean,
            details={
                "precision": P.mean().item() * 100,
                "recall": R.mean().item() * 100,
                "f1": f1_mean,
                "model": model_type,
                "lang": lang,
            },
        )
    except ImportError:
        logger.warning("bert_score not installed")
        return MetricResult("BERTScore", 0.0)


def compute_perplexity(
    texts: List[str],
    model=None,
    tokenizer=None,
) -> MetricResult:
    """
    Compute perplexity of generated texts.

    Lower perplexity = more fluent/natural text.
    Uses a reference language model if provided.
    """
    try:
        import torch

        if model is None or tokenizer is None:
            # Use a default model for perplexity computation
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_name = "ai4bharat/IndicBARTSS"
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                if torch.cuda.is_available():
                    model = model.cuda()
            except Exception:
                # Fallback: estimate perplexity using text statistics
                avg_len = np.mean([len(t.split()) for t in texts])
                estimated_ppl = 100 + avg_len * 2
                return MetricResult("Perplexity", estimated_ppl, details={"method": "estimated"})

        model.eval()
        total_loss = 0
        total_tokens = 0

        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
                total_tokens += inputs["input_ids"].shape[1]

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = np.exp(avg_loss)

        return MetricResult(
            metric_name="Perplexity",
            score=ppl,
            details={"avg_loss": avg_loss, "total_tokens": total_tokens},
        )
    except Exception as e:
        logger.warning(f"Perplexity computation failed: {e}")
        return MetricResult("Perplexity", float("inf"))


def compute_chrf(
    predictions: List[str],
    references: List[str],
) -> MetricResult:
    """Compute chrF score (character-level F-score)."""
    try:
        import sacrebleu
        refs = [[ref] for ref in references]
        chrf = sacrebleu.corpus_chrf(predictions, list(zip(*refs)))
        return MetricResult("chrF", chrf.score, details={"chrf": chrf.score})
    except ImportError:
        return MetricResult("chrF", 0.0)


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def compute_classification_metrics(
    predictions: List[str],
    references: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, MetricResult]:
    """
    Compute comprehensive classification metrics.

    Returns: accuracy, f1_macro, f1_weighted, precision, recall
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        classification_report, confusion_matrix
    )

    # Normalize predictions and references
    pred_normalized = [p.strip().upper() for p in predictions]
    ref_normalized = [r.strip().upper() for r in references]

    if labels is None:
        labels = sorted(set(ref_normalized))

    # Map predictions to valid labels
    valid_preds = []
    for p in pred_normalized:
        matched = False
        for label in labels:
            if label.upper() in p or p in label.upper():
                valid_preds.append(label.upper())
                matched = True
                break
        if not matched:
            valid_preds.append(labels[0].upper())

    labels_upper = [l.upper() for l in labels]

    try:
        accuracy = accuracy_score(ref_normalized, valid_preds) * 100
        f1_macro = f1_score(ref_normalized, valid_preds, labels=labels_upper, average="macro", zero_division=0) * 100
        f1_weighted = f1_score(ref_normalized, valid_preds, labels=labels_upper, average="weighted", zero_division=0) * 100
        precision = precision_score(ref_normalized, valid_preds, labels=labels_upper, average="macro", zero_division=0) * 100
        recall = recall_score(ref_normalized, valid_preds, labels=labels_upper, average="macro", zero_division=0) * 100

        cm = confusion_matrix(ref_normalized, valid_preds, labels=labels_upper)
        report = classification_report(ref_normalized, valid_preds, labels=labels_upper, zero_division=0, output_dict=True)

        return {
            "accuracy": MetricResult("Accuracy", accuracy),
            "f1_macro": MetricResult("F1 (Macro)", f1_macro, details={"per_class": report}),
            "f1_weighted": MetricResult("F1 (Weighted)", f1_weighted),
            "precision": MetricResult("Precision (Macro)", precision),
            "recall": MetricResult("Recall (Macro)", recall),
            "confusion_matrix": MetricResult(
                "Confusion Matrix", accuracy,
                details={"matrix": cm.tolist(), "labels": labels_upper}
            ),
        }
    except Exception as e:
        logger.error(f"Classification metrics failed: {e}")
        return {"accuracy": MetricResult("Accuracy", 0.0)}


# ============================================================
# UNIFIED EVALUATOR
# ============================================================

def evaluate_generation(
    predictions: List[str],
    references: List[str],
    task_type: str = "translation",
    lang: str = "hi",
) -> EvaluationResult:
    """
    Run all generation metrics for a set of predictions.

    Args:
        predictions: Model-generated texts
        references: Ground-truth reference texts
        task_type: "translation" or "summarization"
        lang: Language code for BERTScore

    Returns:
        EvaluationResult with all metrics
    """
    result = EvaluationResult(
        model_name="", task_name="", pipeline_step="",
        num_samples=len(predictions),
    )

    # BLEU
    logger.info("Computing BLEU...")
    result.metrics["bleu"] = compute_bleu(predictions, references)

    # BERTScore
    logger.info("Computing BERTScore...")
    result.metrics["bertscore"] = compute_bertscore(predictions, references, lang=lang)

    # Perplexity
    logger.info("Computing Perplexity...")
    result.metrics["perplexity"] = compute_perplexity(predictions)

    # chrF
    logger.info("Computing chrF...")
    result.metrics["chrf"] = compute_chrf(predictions, references)

    if task_type == "summarization":
        logger.info("Computing ROUGE...")
        rouge_results = compute_rouge(predictions, references)
        result.metrics.update(rouge_results)

    return result


def evaluate_classification(
    predictions: List[str],
    references: List[str],
    labels: Optional[List[str]] = None,
) -> EvaluationResult:
    """
    Run all classification metrics.

    Args:
        predictions: Model predictions
        references: Ground-truth labels
        labels: List of valid class labels

    Returns:
        EvaluationResult with all metrics
    """
    result = EvaluationResult(
        model_name="", task_name="", pipeline_step="",
        num_samples=len(predictions),
    )

    logger.info("Computing classification metrics...")
    cls_metrics = compute_classification_metrics(predictions, references, labels)
    result.metrics.update(cls_metrics)

    return result
