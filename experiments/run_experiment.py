"""
Main experiment orchestrator.

Runs a single experiment: one model × one task × one pipeline step.
Handles model loading, data loading, inference, evaluation, and result saving.

Usage:
    python experiments/run_experiment.py --model llama --task Hindi_translation --pipeline zero_shot
    python experiments/run_experiment.py --model sedd --task Hindi_legal_bail --pipeline few_shot
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from config.base_config import (
    ModelName, TaskName, PipelineStep, ExperimentConfig, ProjectConfig
)
from config.model_configs import get_model_config
from config.task_configs import get_task_config


# ============================================================
# Model Registry
# ============================================================

MODEL_REGISTRY = {
    ModelName.SEDD: "src.models.diffusion.sedd_wrapper.SEDDWrapper",
    ModelName.LLADA: "src.models.diffusion.llada_wrapper.LLaDAWrapper",
    ModelName.D3PM: "src.models.diffusion.d3pm_wrapper.D3PMWrapper",
    ModelName.DIFFUSELM: "src.models.diffusion.diffuselm_wrapper.DiffuseLMWrapper",
    ModelName.LLAMA: "src.models.autoregressive.llama_wrapper.LLaMAWrapper",
    ModelName.GEMMA: "src.models.autoregressive.gemma_wrapper.GemmaWrapper",
    ModelName.MISTRAL: "src.models.autoregressive.mistral_wrapper.MistralWrapper",
    ModelName.BERT: "src.models.autoregressive.bert_wrapper.BERTWrapper",
}


def get_model_class(model_name: ModelName):
    """Dynamically import and return the model class."""
    import importlib

    class_path = MODEL_REGISTRY[model_name]
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ============================================================
# Experiment Runner
# ============================================================

def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run a single experiment.

    Args:
        config: ExperimentConfig specifying model, task, and pipeline step

    Returns:
        Dictionary with all results, metrics, and metadata
    """
    experiment_id = config.experiment_id
    logger.info(f"{'='*60}")
    logger.info(f"Starting Experiment: {experiment_id}")
    logger.info(f"  Model: {config.model_name.value}")
    logger.info(f"  Task: {config.task_name.value}")
    logger.info(f"  Pipeline: {config.pipeline_step.value}")
    logger.info(f"{'='*60}")

    start_time = time.time()

    # Create results directory
    results_dir = config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Step 1: Load Data
    # --------------------------------------------------------
    logger.info("Step 1: Loading data...")
    from src.data.data_loader import load_dataset_for_task

    task_config = get_task_config(config.task_name)

    inputs, targets = load_dataset_for_task(
        task_name=config.task_name.value,
        split="test",
        num_samples=config.num_eval_samples,
    )
    logger.info(f"  Loaded {len(inputs)} samples")

    # --------------------------------------------------------
    # Step 2: Prepare Prompts
    # --------------------------------------------------------
    logger.info("Step 2: Preparing prompts...")
    prompts = _prepare_prompts(
        inputs=inputs,
        task_config=task_config,
        pipeline_step=config.pipeline_step,
        num_few_shot=config.num_few_shot_examples,
    )

    # --------------------------------------------------------
    # Step 3: Load Model
    # --------------------------------------------------------
    logger.info("Step 3: Loading model...")
    ModelClass = get_model_class(config.model_name)
    model = ModelClass(
        experiment_config=config,
        device=config.device,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )
    model.load()

    # --------------------------------------------------------
    # Step 4: Run Inference
    # --------------------------------------------------------
    logger.info("Step 4: Running inference...")

    if task_config.task_type == "generation":
        # Generation tasks: translation, summarization
        output = model.generate(
            prompts=prompts,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.do_sample,
        )
        predictions = output.generated_texts
        generation_time = output.generation_time_seconds

    else:
        # Classification tasks: bail prediction, verdict
        system_prompt = task_config.system_prompt

        # Construct prompts with classification format
        cls_prompts = []
        for prompt in prompts:
            formatted = model.format_prompt(prompt, system_prompt)
            cls_prompts.append(formatted)

        output = model.classify(
            texts=cls_prompts,
            labels=task_config.class_labels,
        )
        predictions = output.predictions
        generation_time = output.classification_time_seconds

    logger.info(f"  Generated {len(predictions)} predictions in {generation_time:.1f}s")

    # Save raw outputs
    raw_output = {
        "experiment_id": experiment_id,
        "predictions": predictions,
        "targets": targets,
        "inputs": inputs[:len(predictions)],
        "generation_time_seconds": generation_time,
    }
    raw_output_path = results_dir / "raw_output.json"
    with open(raw_output_path, "w", encoding="utf-8") as f:
        json.dump(raw_output, f, ensure_ascii=False, indent=2)

    # --------------------------------------------------------
    # Step 5: Evaluate
    # --------------------------------------------------------
    logger.info("Step 5: Evaluating results...")
    from src.evaluation.metrics import evaluate_generation, evaluate_classification

    # Align predictions and targets
    min_len = min(len(predictions), len(targets))
    predictions = predictions[:min_len]
    eval_targets = targets[:min_len]

    if task_config.task_type == "generation":
        eval_result = evaluate_generation(
            predictions=predictions,
            references=eval_targets,
            task_type="translation" if config.task_name == TaskName.HINDI_TRANSLATION else "summarization",
            lang="hi",
        )
    else:
        eval_result = evaluate_classification(
            predictions=predictions,
            references=eval_targets,
            labels=task_config.class_labels,
        )

    eval_result.model_name = config.model_name.value
    eval_result.task_name = config.task_name.value
    eval_result.pipeline_step = config.pipeline_step.value

    # Save evaluation results
    metrics_path = results_dir / "metrics.json"
    eval_result.save(metrics_path)

    # --------------------------------------------------------
    # Step 6: Unload Model
    # --------------------------------------------------------
    model.unload()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment Complete: {experiment_id}")
    logger.info(f"  Total time: {elapsed:.1f}s")
    logger.info(f"  Results saved to: {results_dir}")
    logger.info(f"  Metrics:")
    for name, metric in eval_result.metrics.items():
        logger.info(f"    {name}: {metric.score:.2f}")
    logger.info(f"{'='*60}\n")

    return eval_result.to_dict()


def _prepare_prompts(
    inputs: List[str],
    task_config,
    pipeline_step: PipelineStep,
    num_few_shot: int = 5,
) -> List[str]:
    """
    Prepare prompts based on the pipeline step.

    - Zero-Shot: task instruction + input only
    - Few-Shot: task instruction + examples + input
    - Fine-Tuning/RAG/Agentic: similar to zero-shot but model is adapted
    """
    prompts = []

    if pipeline_step == PipelineStep.ZERO_SHOT:
        for inp in inputs:
            prompt = task_config.zero_shot_template.format(input=inp)
            prompts.append(prompt)

    elif pipeline_step == PipelineStep.FEW_SHOT:
        # Format few-shot examples
        examples_str = ""
        for ex in task_config.few_shot_examples[:num_few_shot]:
            if task_config.task_type == "generation":
                if task_config.name.value == "Hindi_translation":
                    examples_str += f"English: {ex['input']}\nHindi: {ex['output']}\n\n"
                else:
                    examples_str += f"पाठ: {ex['input']}\nसारांश: {ex['output']}\n\n"
            else:
                examples_str += f"दस्तावेज़: {ex['input']}\nभविष्यवाणी: {ex['output']}\n\n"

        for inp in inputs:
            prompt = task_config.few_shot_template.format(
                examples=examples_str.strip(),
                input=inp,
            )
            prompts.append(prompt)

    else:
        # For fine-tuning, RAG, and agentic, use zero-shot format
        # (the model itself is adapted, not the prompt)
        for inp in inputs:
            prompt = task_config.zero_shot_template.format(input=inp)
            prompts.append(prompt)

    return prompts


# ============================================================
# CLI Entry Point
# ============================================================

MODEL_NAME_MAP = {
    "sedd": ModelName.SEDD,
    "llada": ModelName.LLADA,
    "d3pm": ModelName.D3PM,
    "diffuselm": ModelName.DIFFUSELM,
    "llama": ModelName.LLAMA,
    "gemma": ModelName.GEMMA,
    "mistral": ModelName.MISTRAL,
    "bert": ModelName.BERT,
    "indicbert": ModelName.BERT,
}

TASK_NAME_MAP = {
    "hindi_translation": TaskName.HINDI_TRANSLATION,
    "translation": TaskName.HINDI_TRANSLATION,
    "hindi_summary": TaskName.HINDI_SUMMARY,
    "summary": TaskName.HINDI_SUMMARY,
    "summarization": TaskName.HINDI_SUMMARY,
    "bail": TaskName.BAIL_PREDICTION,
    "bail_prediction": TaskName.BAIL_PREDICTION,
    "hindi_legal_bail": TaskName.BAIL_PREDICTION,
    "verdict": TaskName.JUDGE_VERDICT,
    "judge_verdict": TaskName.JUDGE_VERDICT,
    "hindi_judge_verdict": TaskName.JUDGE_VERDICT,
}

PIPELINE_MAP = {
    "zero_shot": PipelineStep.ZERO_SHOT,
    "0shot": PipelineStep.ZERO_SHOT,
    "few_shot": PipelineStep.FEW_SHOT,
    "fewshot": PipelineStep.FEW_SHOT,
    "fine_tuning": PipelineStep.FINE_TUNING,
    "finetune": PipelineStep.FINE_TUNING,
    "finetuning": PipelineStep.FINE_TUNING,
    "rag": PipelineStep.RAG,
    "agentic": PipelineStep.AGENTIC,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run a single experiment: model × task × pipeline step"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_NAME_MAP.keys()),
        help="Model to use"
    )
    parser.add_argument(
        "--task", type=str, required=True,
        choices=list(TASK_NAME_MAP.keys()),
        help="Task to evaluate on"
    )
    parser.add_argument(
        "--pipeline", type=str, required=True,
        choices=list(PIPELINE_MAP.keys()),
        help="Pipeline step"
    )
    parser.add_argument("--num_samples", type=int, default=200, help="Number of evaluation samples")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")

    args = parser.parse_args()

    # Resolve names
    model_name = MODEL_NAME_MAP[args.model.lower()]
    task_name = TASK_NAME_MAP[args.task.lower()]
    pipeline_step = PIPELINE_MAP[args.pipeline.lower()]

    # Create experiment config
    config = ExperimentConfig(
        model_name=model_name,
        task_name=task_name,
        pipeline_step=pipeline_step,
        num_eval_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        device=args.device,
        load_in_4bit=not args.no_4bit,
    )

    # Set random seeds
    import torch
    import random
    import numpy as np

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Run experiment
    results = run_experiment(config)

    # Print summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {config.experiment_id}")
    print("=" * 60)
    for metric_name, metric_data in results.get("metrics", {}).items():
        print(f"  {metric_data['metric_name']}: {metric_data['score']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
