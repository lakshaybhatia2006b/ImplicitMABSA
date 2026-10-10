"""
Evaluation Metrics and Visualizations
---------------------------------------
Computes and plots:
  - Accuracy, Precision, Recall, Macro-F1
  - Per-class metrics
  - Confusion matrices
  - Comparative bar charts (ablation study)

All plots saved to outputs/plots/.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PLOTS_DIR = Path("outputs/plots")
RESULTS_DIR = Path("outputs/results")

CLASS_NAMES = ["Negative", "Neutral", "Positive"]
LABEL_ORDER = [-1, 0, 1]


# ── Core Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> dict:
    """
    Compute accuracy, precision, recall, macro-F1.
    Returns a dict and prints a formatted table.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        precision_recall_fscore_support,
    )

    # Determine labels to evaluate
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    # If standard 3-class set, use LABEL_ORDER & CLASS_NAMES; otherwise use dynamic mapping
    if set(unique_labels).issubset(set(LABEL_ORDER)):
        eval_labels = [l for l in LABEL_ORDER if l in unique_labels] or LABEL_ORDER
        label_to_name = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        eval_names = [label_to_name[l] for l in eval_labels]
    else:
        eval_labels = unique_labels
        eval_names = [str(l) for l in eval_labels]

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average="macro",
        labels=eval_labels,
        zero_division=0,
    )
    report = classification_report(
        y_true, y_pred,
        labels=eval_labels,
        target_names=eval_names,
        zero_division=0,
    )

    result = {
        "model": model_name,
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(prec), 4),
        "macro_recall": round(float(rec), 4),
        "macro_f1": round(float(f1), 4),
        "classification_report": report,
    }

    print(f"\n{'='*50}")
    print(f" Model: {model_name}")
    print(f"{'='*50}")
    print(f" Accuracy  : {acc:.4f}")
    print(f" Precision : {prec:.4f} (macro)")
    print(f" Recall    : {rec:.4f} (macro)")
    print(f" Macro-F1  : {f1:.4f}")
    print("\nClassification Report:")
    print(report)
    return result


# ── Confusion Matrix ───────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
    save: bool = True,
) -> Optional[str]:
    """
    Plot and optionally save a confusion matrix.
    Returns the path to the saved figure.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    if set(unique_labels).issubset(set(LABEL_ORDER)):
        eval_labels = [l for l in LABEL_ORDER if l in unique_labels] or LABEL_ORDER
        label_to_name = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        eval_names = [label_to_name[l] for l in eval_labels]
    else:
        eval_labels = unique_labels
        eval_names = [str(l) for l in eval_labels]

    cm = confusion_matrix(y_true, y_pred, labels=eval_labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=eval_names,
        yticklabels=eval_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        safe_name = model_name.replace(" ", "_").replace("/", "-")
        path = PLOTS_DIR / f"cm_{safe_name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[Metrics] Confusion matrix saved to {path}")
        return str(path)
    plt.show()
    return None


# ── Ablation Bar Chart ─────────────────────────────────────────────────────────

def plot_ablation(
    results: Dict[str, dict],
    metric: str = "macro_f1",
    save: bool = True,
) -> Optional[str]:
    """
    Create a grouped bar chart comparing all models on key metrics.
    results: {model_name: metrics_dict}
    """
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_to_plot = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "Macro-F1"]

    model_names = list(results.keys())
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    for i, (name, res) in enumerate(results.items()):
        vals = [res.get(m, 0.0) for m in metrics_to_plot]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=colors[i], alpha=0.85)
        # Annotate bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Ablation Study — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()

    if save:
        path = PLOTS_DIR / "ablation_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[Metrics] Ablation chart saved to {path}")
        return str(path)
    plt.show()
    return None


# ── Save / Load Results ────────────────────────────────────────────────────────

def save_all_results(results: Dict[str, dict], filename: str = "all_results.json"):
    """Save all model results to a JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {name: {k: v for k, v in res.items() if k != "classification_report"}
           for name, res in results.items()}
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Metrics] All results saved to {path}")
