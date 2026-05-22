"""
Publication-quality visualization module for experiment results.

Generates heatmaps, bar charts, radar charts, training curves,
confusion matrices, and comprehensive dashboards comparing all
8 models across 4 tasks and 5 pipeline steps.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib/seaborn not installed. Visualization disabled.")


# ============================================================
# Style Configuration
# ============================================================

DIFFUSION_COLOR = "#FF6B6B"    # Coral red for diffusion models
AR_COLOR = "#4ECDC4"           # Teal for auto-regressive models
ACCENT_COLOR = "#FFE66D"       # Yellow accent
BG_COLOR = "#2C3E50"           # Dark background
TEXT_COLOR = "#ECF0F1"         # Light text

MODEL_COLORS = {
    "SEDD": "#FF6B6B",
    "LLaDA": "#FF8E53",
    "D3PM": "#FFA07A",
    "DiffuseLM": "#FF69B4",
    "LLaMA": "#4ECDC4",
    "Gemma": "#45B7D1",
    "Mistral": "#96CEB4",
    "IndicBERT": "#88D8B0",
}

TASK_SHORT_NAMES = {
    "Hindi_translation": "Translation",
    "Hindi_summary": "Summarization",
    "Hindi_legal_bail": "Bail Prediction",
    "Hindi_judge_verdict": "Verdict Prediction",
}

STEP_SHORT_NAMES = {
    "zero_shot": "0-Shot",
    "few_shot": "Few-Shot",
    "fine_tuning": "Fine-Tuned",
    "rag": "RAG",
    "agentic": "Agentic",
}


def setup_style():
    """Set up publication-quality plot style."""
    if not HAS_MATPLOTLIB:
        return

    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "text.color": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "grid.color": "#333366",
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "#1a1a2e",
    })


def plot_comparison_heatmap(
    results: Dict[str, Dict[str, float]],
    metric_name: str,
    title: str = "Model-Task Comparison",
    output_path: Optional[str] = None,
) -> None:
    """
    Create a heatmap comparing models across tasks.

    Args:
        results: {model_name: {task_name: score}}
        metric_name: Name of the metric being visualized
        title: Plot title
        output_path: Path to save the figure
    """
    if not HAS_MATPLOTLIB:
        return

    setup_style()

    models = list(results.keys())
    tasks = list(next(iter(results.values())).keys())
    task_labels = [TASK_SHORT_NAMES.get(t, t) for t in tasks]

    # Build matrix
    matrix = np.array([[results[m].get(t, 0) for t in tasks] for m in models])

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(tasks)):
            value = matrix[i, j]
            text_color = "white" if value < (matrix.max() + matrix.min()) / 2 else "black"
            ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                    color=text_color, fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(task_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    # Color-code model labels by type
    for i, model in enumerate(models):
        color = MODEL_COLORS.get(model, "#ffffff")
        ax.get_yticklabels()[i].set_color(color)

    ax.set_title(f"{title}\n({metric_name})", fontsize=16, fontweight="bold", pad=20)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(metric_name, fontsize=12)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Saved heatmap to {output_path}")

    plt.close()


def plot_grouped_bar_chart(
    results: Dict[str, Dict[str, float]],
    metric_name: str,
    title: str = "Diffusion vs Auto-Regressive Comparison",
    output_path: Optional[str] = None,
) -> None:
    """
    Create grouped bar chart comparing models on a specific metric.

    Groups models by type (Diffusion vs Auto-Regressive).
    """
    if not HAS_MATPLOTLIB:
        return

    setup_style()

    tasks = list(next(iter(results.values())).keys())
    task_labels = [TASK_SHORT_NAMES.get(t, t) for t in tasks]
    models = list(results.keys())

    x = np.arange(len(tasks))
    width = 0.08
    num_models = len(models)

    fig, ax = plt.subplots(figsize=(16, 8))

    for i, model in enumerate(models):
        scores = [results[model].get(t, 0) for t in tasks]
        offset = (i - num_models / 2 + 0.5) * width
        color = MODEL_COLORS.get(model, "#999999")
        bars = ax.bar(x + offset, scores, width, label=model, color=color,
                      edgecolor="white", linewidth=0.5, alpha=0.85)

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            if score > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{score:.1f}", ha="center", va="bottom",
                        fontsize=7, fontweight="bold", color=color)

    ax.set_xlabel("Tasks", fontsize=13)
    ax.set_ylabel(metric_name, fontsize=13)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=11)
    ax.legend(loc="upper left", ncol=4, framealpha=0.3)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Saved bar chart to {output_path}")

    plt.close()


def plot_radar_chart(
    model_scores: Dict[str, Dict[str, float]],
    title: str = "Multi-Metric Model Profile",
    output_path: Optional[str] = None,
) -> None:
    """
    Create radar/spider chart for multi-metric comparison.

    Args:
        model_scores: {model_name: {metric_name: score}}
    """
    if not HAS_MATPLOTLIB:
        return

    setup_style()

    models = list(model_scores.keys())
    metrics = list(next(iter(model_scores.values())).keys())
    num_metrics = len(metrics)

    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_facecolor("#16213e")

    for model in models:
        values = [model_scores[model].get(m, 0) for m in metrics]
        values += values[:1]
        color = MODEL_COLORS.get(model, "#999999")
        ax.plot(angles, values, "o-", linewidth=2, label=model, color=color, markersize=4)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=30, color="#e0e0e0")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), framealpha=0.3)
    ax.grid(color="#333366", alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Saved radar chart to {output_path}")

    plt.close()


def plot_pipeline_progression(
    results: Dict[str, Dict[str, float]],
    metric_name: str,
    task_name: str,
    title: str = "Performance Across Pipeline Steps",
    output_path: Optional[str] = None,
) -> None:
    """
    Line chart showing how each model's performance changes
    across the 5 pipeline steps for a specific task.
    """
    if not HAS_MATPLOTLIB:
        return

    setup_style()

    steps = ["zero_shot", "few_shot", "fine_tuning", "rag", "agentic"]
    step_labels = [STEP_SHORT_NAMES.get(s, s) for s in steps]

    fig, ax = plt.subplots(figsize=(12, 7))

    for model, step_scores in results.items():
        scores = [step_scores.get(s, 0) for s in steps]
        color = MODEL_COLORS.get(model, "#999999")
        linestyle = "--" if model in ["SEDD", "LLaDA", "D3PM", "DiffuseLM"] else "-"
        ax.plot(step_labels, scores, "o-", label=model, color=color,
                linewidth=2.5, markersize=8, linestyle=linestyle)

    task_label = TASK_SHORT_NAMES.get(task_name, task_name)
    ax.set_xlabel("Pipeline Step", fontsize=13)
    ax.set_ylabel(metric_name, fontsize=13)
    ax.set_title(f"{title}\n{task_label} — {metric_name}", fontsize=16, fontweight="bold", pad=20)
    ax.legend(loc="best", ncol=2, framealpha=0.3)
    ax.grid(axis="both", alpha=0.2)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Saved progression chart to {output_path}")

    plt.close()


def plot_confusion_matrix(
    matrix: List[List[int]],
    labels: List[str],
    model_name: str,
    task_name: str,
    output_path: Optional[str] = None,
) -> None:
    """Plot a confusion matrix for classification tasks."""
    if not HAS_MATPLOTLIB:
        return

    setup_style()

    fig, ax = plt.subplots(figsize=(8, 6))
    matrix_np = np.array(matrix)

    sns.heatmap(
        matrix_np, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=1, linecolor="#333366",
        cbar_kws={"shrink": 0.8},
    )

    task_label = TASK_SHORT_NAMES.get(task_name, task_name)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}\n{task_label}",
                 fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Saved confusion matrix to {output_path}")

    plt.close()


def generate_summary_dashboard(
    all_results: Dict[str, Any],
    output_path: str = "results/figures/summary_dashboard.png",
) -> None:
    """
    Generate a comprehensive single-page summary dashboard.

    Contains: heatmap, bar chart, and key findings in one figure.
    """
    if not HAS_MATPLOTLIB:
        return

    setup_style()

    fig = plt.figure(figsize=(24, 16))
    fig.suptitle(
        "Diffusion vs Auto-Regressive Models for Hindi NLP\nComprehensive Results Dashboard",
        fontsize=20, fontweight="bold", color="#e0e0e0", y=0.98,
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Generate sub-plots using the results data
    # This is a placeholder structure - actual data fills during experiment runs

    # Subplot 1: Overall comparison heatmap
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_title("Model × Task Scores", fontsize=14, fontweight="bold")
    ax1.text(0.5, 0.5, "Generated after experiments", ha="center", va="center",
             fontsize=12, color="#999", transform=ax1.transAxes)

    # Subplot 2: Best model per task
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_title("Best Model per Task", fontsize=14, fontweight="bold")
    ax2.text(0.5, 0.5, "Generated after experiments", ha="center", va="center",
             fontsize=12, color="#999", transform=ax2.transAxes)

    # Subplot 3: Diffusion vs AR average
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("Diffusion vs AR (Average)", fontsize=14, fontweight="bold")
    ax3.text(0.5, 0.5, "Generated after experiments", ha="center", va="center",
             fontsize=12, color="#999", transform=ax3.transAxes)

    # Subplot 4: Pipeline progression
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("Performance by Pipeline Step", fontsize=14, fontweight="bold")
    ax4.text(0.5, 0.5, "Generated after experiments", ha="center", va="center",
             fontsize=12, color="#999", transform=ax4.transAxes)

    # Subplot 5: Key findings
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_title("Key Findings", fontsize=14, fontweight="bold")
    ax5.axis("off")
    findings = [
        "1. Experiment matrix: 8 models × 4 tasks × 5 steps",
        "2. Diffusion models: SEDD, LLaDA, D3PM, DiffuseLM",
        "3. AR models: LLaMA, Gemma, Mistral, IndicBERT",
        "4. Tasks: Translation, Summary, Bail, Verdict",
        "5. Metrics: BLEU, ROUGE, BERTScore, F1, etc.",
    ]
    for i, finding in enumerate(findings):
        ax5.text(0.05, 0.85 - i * 0.15, finding, fontsize=11, color="#e0e0e0",
                 transform=ax5.transAxes, va="top")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    logger.info(f"Saved summary dashboard to {output_path}")
    plt.close()
