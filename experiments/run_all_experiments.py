"""
Run all experiments in the 8 models × 4 tasks × 5 steps matrix.

Supports priority mode (4 models × 4 tasks) and full mode.
Tracks progress, handles failures gracefully, and generates comparison reports.

Usage:
    python experiments/run_all_experiments.py --priority-only
    python experiments/run_all_experiments.py --full
    python experiments/run_all_experiments.py --models sedd llama --tasks translation bail
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from config.base_config import (
    ModelName, TaskName, PipelineStep, ExperimentConfig, ProjectConfig,
)
from experiments.run_experiment import run_experiment, MODEL_NAME_MAP, TASK_NAME_MAP, PIPELINE_MAP


def run_all_experiments(
    models: List[ModelName],
    tasks: List[TaskName],
    steps: List[PipelineStep],
    num_samples: int = 200,
    device: str = "cuda",
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Run all experiments in the specified matrix.

    Args:
        models: List of models to evaluate
        tasks: List of tasks to evaluate on
        steps: List of pipeline steps
        num_samples: Number of evaluation samples per experiment
        device: Computing device
        resume: Skip already-completed experiments

    Returns:
        Dictionary of all results
    """
    total = len(models) * len(tasks) * len(steps)
    completed = 0
    failed = 0
    skipped = 0
    all_results = {}

    logger.info(f"\n{'='*70}")
    logger.info(f"STARTING EXPERIMENT MATRIX")
    logger.info(f"  Models: {[m.value for m in models]}")
    logger.info(f"  Tasks: {[t.value for t in tasks]}")
    logger.info(f"  Steps: {[s.value for s in steps]}")
    logger.info(f"  Total experiments: {total}")
    logger.info(f"  Samples per experiment: {num_samples}")
    logger.info(f"{'='*70}\n")

    start_time = time.time()

    for model in models:
        for task in tasks:
            for step in steps:
                experiment_id = f"{model.value}__{task.value}__{step.value}"

                # Check if already completed
                results_path = Path("results") / "raw" / task.value / model.value / step.value / "metrics.json"
                if resume and results_path.exists():
                    logger.info(f"[SKIP] {experiment_id} — already completed")
                    try:
                        with open(results_path, "r") as f:
                            all_results[experiment_id] = json.load(f)
                    except Exception:
                        pass
                    skipped += 1
                    continue

                try:
                    logger.info(f"\n[{completed + failed + skipped + 1}/{total}] Running: {experiment_id}")

                    config = ExperimentConfig(
                        model_name=model,
                        task_name=task,
                        pipeline_step=step,
                        num_eval_samples=num_samples,
                        device=device,
                        seed=42,
                    )

                    result = run_experiment(config)
                    all_results[experiment_id] = result
                    completed += 1

                except Exception as e:
                    logger.error(f"[FAILED] {experiment_id}: {e}")
                    all_results[experiment_id] = {
                        "error": str(e),
                        "model_name": model.value,
                        "task_name": task.value,
                        "pipeline_step": step.value,
                    }
                    failed += 1

    elapsed = time.time() - start_time

    # Save aggregated results
    agg_path = Path("results") / "metrics" / "aggregated" / "all_results.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    # Generate comparison report
    _generate_comparison_report(all_results, models, tasks, steps)

    # Generate visualizations
    _generate_visualizations(all_results, models, tasks, steps)

    logger.info(f"\n{'='*70}")
    logger.info(f"EXPERIMENT MATRIX COMPLETE")
    logger.info(f"  Completed: {completed}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Total time: {elapsed / 60:.1f} minutes")
    logger.info(f"  Results: {agg_path}")
    logger.info(f"{'='*70}\n")

    return all_results


def _generate_comparison_report(
    all_results: Dict[str, Any],
    models: List[ModelName],
    tasks: List[TaskName],
    steps: List[PipelineStep],
) -> None:
    """Generate a markdown comparison report."""
    report_path = Path("results") / "metrics" / "aggregated" / "comparison_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Diffusion vs Auto-Regressive Models: Comparison Report\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"**Models:** {', '.join(m.value for m in models)}\n",
        f"**Tasks:** {', '.join(t.value for t in tasks)}\n",
        f"**Pipeline Steps:** {', '.join(s.value for s in steps)}\n",
        "\n---\n",
    ]

    # For each task, create a comparison table
    for task in tasks:
        lines.append(f"\n## {task.value}\n")

        # Table header
        header = "| Model | Type |"
        separator = "|-------|------|"

        # Get metrics from first available result
        metric_names = []
        for model in models:
            for step in steps:
                exp_id = f"{model.value}__{task.value}__{step.value}"
                if exp_id in all_results and "metrics" in all_results[exp_id]:
                    metric_names = list(all_results[exp_id]["metrics"].keys())[:5]
                    break
            if metric_names:
                break

        if not metric_names:
            metric_names = ["primary_score"]

        for step in steps:
            for metric in metric_names[:3]:  # Top 3 metrics
                header += f" {step.value}_{metric} |"
                separator += f"---|"

        lines.append(header)
        lines.append(separator)

        # Table rows
        for model in models:
            model_type = "🌀 Diff" if model.is_diffusion else "➡️ AR"
            row = f"| **{model.value}** | {model_type} |"

            for step in steps:
                exp_id = f"{model.value}__{task.value}__{step.value}"
                if exp_id in all_results and "metrics" in all_results[exp_id]:
                    metrics = all_results[exp_id]["metrics"]
                    for metric in metric_names[:3]:
                        if metric in metrics:
                            score = metrics[metric]["score"]
                            row += f" {score:.1f} |"
                        else:
                            row += " - |"
                else:
                    for _ in metric_names[:3]:
                        row += " - |"

            lines.append(row)

        lines.append("")

    # Key findings
    lines.append("\n## Key Findings\n")
    lines.append("_Analysis will be populated after all experiments complete._\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Comparison report saved to {report_path}")


def _generate_visualizations(
    all_results: Dict[str, Any],
    models: List[ModelName],
    tasks: List[TaskName],
    steps: List[PipelineStep],
) -> None:
    """Generate all visualization plots from results."""
    try:
        from src.utils.visualization import (
            plot_comparison_heatmap,
            plot_grouped_bar_chart,
            plot_radar_chart,
            plot_pipeline_progression,
        )

        figures_dir = Path("results") / "figures"

        # 1. Heatmap: primary metric per model × task (zero-shot)
        for step in steps:
            heatmap_data = {}
            for model in models:
                heatmap_data[model.value] = {}
                for task in tasks:
                    exp_id = f"{model.value}__{task.value}__{step.value}"
                    if exp_id in all_results and "metrics" in all_results[exp_id]:
                        metrics = all_results[exp_id]["metrics"]
                        primary = list(metrics.values())[0]["score"] if metrics else 0
                        heatmap_data[model.value][task.value] = primary
                    else:
                        heatmap_data[model.value][task.value] = 0

            plot_comparison_heatmap(
                heatmap_data,
                metric_name="Primary Metric",
                title=f"Model × Task Comparison ({step.value})",
                output_path=str(figures_dir / "comparison_heatmaps" / f"heatmap_{step.value}.png"),
            )

        # 2. Bar chart: all models on all tasks
        bar_data = {}
        for model in models:
            bar_data[model.value] = {}
            for task in tasks:
                exp_id = f"{model.value}__{task.value}__zero_shot"
                if exp_id in all_results and "metrics" in all_results[exp_id]:
                    metrics = all_results[exp_id]["metrics"]
                    primary = list(metrics.values())[0]["score"] if metrics else 0
                    bar_data[model.value][task.value] = primary
                else:
                    bar_data[model.value][task.value] = 0

        plot_grouped_bar_chart(
            bar_data,
            metric_name="Score",
            title="All Models × All Tasks (Zero-Shot)",
            output_path=str(figures_dir / "bar_charts" / "all_models_zero_shot.png"),
        )

        # 3. Pipeline progression for each task
        for task in tasks:
            prog_data = {}
            for model in models:
                prog_data[model.value] = {}
                for step in steps:
                    exp_id = f"{model.value}__{task.value}__{step.value}"
                    if exp_id in all_results and "metrics" in all_results[exp_id]:
                        metrics = all_results[exp_id]["metrics"]
                        primary = list(metrics.values())[0]["score"] if metrics else 0
                        prog_data[model.value][step.value] = primary
                    else:
                        prog_data[model.value][step.value] = 0

            plot_pipeline_progression(
                prog_data,
                metric_name="Score",
                task_name=task.value,
                output_path=str(figures_dir / "bar_charts" / f"progression_{task.value}.png"),
            )

        logger.info("All visualizations generated successfully")

    except Exception as e:
        logger.warning(f"Visualization generation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run the full experiment matrix")
    parser.add_argument("--priority-only", action="store_true",
                        help="Run only priority models (SEDD, LLaDA, LLaMA, Gemma)")
    parser.add_argument("--full", action="store_true",
                        help="Run full 8×4×5 matrix")
    parser.add_argument("--models", nargs="+", type=str, default=None,
                        help="Specific models to run")
    parser.add_argument("--tasks", nargs="+", type=str, default=None,
                        help="Specific tasks to run")
    parser.add_argument("--steps", nargs="+", type=str, default=None,
                        help="Specific pipeline steps to run")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-run completed experiments")

    args = parser.parse_args()

    project = ProjectConfig()

    # Determine models
    if args.models:
        models = [MODEL_NAME_MAP[m.lower()] for m in args.models]
    elif args.priority_only:
        models = project.priority_models
    else:
        models = list(ModelName)

    # Determine tasks
    if args.tasks:
        tasks = [TASK_NAME_MAP[t.lower()] for t in args.tasks]
    else:
        tasks = list(TaskName)

    # Determine pipeline steps
    if args.steps:
        steps = [PIPELINE_MAP[s.lower()] for s in args.steps]
    else:
        steps = list(PipelineStep)

    # Ensure directories exist
    project.ensure_dirs()

    # Run
    run_all_experiments(
        models=models,
        tasks=tasks,
        steps=steps,
        num_samples=args.num_samples,
        device=args.device,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
