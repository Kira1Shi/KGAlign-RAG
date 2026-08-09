from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import TASKS


plt.rcParams.update({"font.size": 13})


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def predicted_rows(predictions: pd.DataFrame, task: str) -> pd.DataFrame:
    rows = predictions[
        predictions.task_type.eq(task) & predictions.risk_score.notna()
    ].copy()
    rows["hallucinated"] = rows.hallucinated.astype(int)
    return rows


def roc_points(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    true_positive = np.cumsum(labels)
    false_positive = np.cumsum(1 - labels)
    return (
        np.r_[0, false_positive / false_positive[-1]],
        np.r_[0, true_positive / true_positive[-1]],
    )


def pr_points(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    true_positive = np.cumsum(labels)
    false_positive = np.cumsum(1 - labels)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / true_positive[-1]
    return np.r_[0, recall], np.r_[1, precision]


def plot_main_metrics(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Plot the four most useful test metrics for each task."""
    columns = [
        ("roc_auc", "ROC AUC"),
        ("pr_auc", "PR AUC"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("f1", "F1"),
    ]
    ordered = metrics.set_index("task_type").reindex(TASKS)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for axis, (column, title) in zip(axes.flat, columns):
        axis.bar(TASKS, ordered[column])
        axis.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        axis.set_ylim(0, 1)
        axis.set_title(title)
    figure.suptitle("Held-out test performance")
    save_figure(figure, output_dir / "01_main_metrics.png")


def plot_roc_curves(predictions: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    for task in TASKS:
        rows = predicted_rows(predictions, task)
        if rows.hallucinated.nunique() == 2:
            false_positive, true_positive = roc_points(
                rows.hallucinated.to_numpy(), rows.risk_score.to_numpy()
            )
            axis.plot(false_positive, true_positive, label=task)
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    axis.set_title("Test ROC curves")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend()
    save_figure(figure, output_dir / "02_roc_curves.png")


def plot_pr_curves(predictions: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    for task in TASKS:
        rows = predicted_rows(predictions, task)
        if rows.hallucinated.nunique() == 2:
            recall, precision = pr_points(
                rows.hallucinated.to_numpy(), rows.risk_score.to_numpy()
            )
            axis.plot(recall, precision, label=task)
    axis.set_title("Test precision-recall curves")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend()
    save_figure(figure, output_dir / "03_pr_curves.png")


def plot_score_distributions(predictions: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    bins = np.linspace(0, 1, 16)
    for axis, task in zip(axes.flat, TASKS):
        rows = predicted_rows(predictions, task)
        factual = rows.loc[rows.hallucinated.eq(0), "risk_score"]
        hallucinated = rows.loc[rows.hallucinated.eq(1), "risk_score"]
        axis.hist(factual, bins=bins, alpha=0.6, label="Factual")
        axis.hist(hallucinated, bins=bins, alpha=0.6, label="Hallucinated")
        axis.axvline(0.5, color="gray", linestyle="--", linewidth=1)
        axis.set_title(task)
        axis.set_xlabel("Risk score")
        axis.set_ylabel("Examples")
        axis.legend()
    figure.suptitle("Test risk-score distributions")
    save_figure(figure, output_dir / "04_score_distributions.png")


def plot_coefficients(coefficients: pd.DataFrame, output_dir: Path) -> None:
    """Create one large coefficient plot per task so labels remain readable."""
    for task in TASKS:
        rows = coefficients[coefficients.task_type.eq(task)].copy()
        rows = rows.sort_values("standardized_coefficient")
        labels = [
            "\n".join(textwrap.wrap(name.replace("_", " "), width=32))
            for name in rows.feature
        ]
        height = max(4.0, 0.85 * len(rows) + 1.8)
        figure, axis = plt.subplots(figsize=(12, height))
        bars = axis.barh(labels, rows.standardized_coefficient)
        axis.axvline(0, color="black", linewidth=1)
        axis.set_title(f"{task} model coefficients", fontsize=16)
        axis.set_xlabel("Standardized coefficient")
        axis.tick_params(axis="y", labelsize=12)
        axis.bar_label(bars, fmt="%.2f", padding=4, fontsize=11)
        axis.margins(x=0.15)
        filename = f"05_coefficients_{task.lower()}.png"
        save_figure(figure, output_dir / filename)


def make_all_plots(results_dir: Path, output_dir: Path | None = None) -> None:
    """Generate every standard plot from one training result directory."""
    output_dir = output_dir or results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(require_file(results_dir / "test_metrics.csv"))
    predictions = pd.read_csv(require_file(results_dir / "test_predictions.csv"))
    coefficients = pd.read_csv(require_file(results_dir / "coefficients.csv"))

    plot_main_metrics(metrics, output_dir)
    plot_roc_curves(predictions, output_dir)
    plot_pr_curves(predictions, output_dir)
    plot_score_distributions(predictions, output_dir)
    plot_coefficients(coefficients, output_dir)
    print(f"Plots saved to {output_dir}")
