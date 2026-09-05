"""Plotting helpers shared across the notebooks."""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve

TARGET_PALETTE = ["skyblue", "salmon"]


def plot_boxplot_by_target(df, feature, title=None, target="Biopsy",
                           target_label="Biopsy Result", figsize=(10, 5)):
    """Box plot of ``feature`` split by the binary target."""
    plt.figure(figsize=figsize)
    sns.boxplot(x=target, y=feature, data=df, palette=TARGET_PALETTE)
    plt.title(title or f"{feature} by {target_label}", fontsize=16)
    plt.xlabel(target_label)
    plt.ylabel(feature)
    plt.show()


def plot_value_count_bars(ax, series, title, colors=("lightgreen", "tomato"),
                          xlabel="Result", ylabel="Count"):
    """Bar chart of value counts for a binary column on the given axis."""
    counts = series.value_counts()
    ax.bar(counts.index, counts.values, color=list(colors), edgecolor="black")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_roc_curves(y_test, probabilities, title="ROC Curves - Model Comparison",
                    figsize=(12, 8)):
    """Overlay ROC curves for every model in ``probabilities``."""
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive"]
    plt.figure(figsize=figsize)
    for i, (name, y_prob) in enumerate(probabilities.items()):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, color=colors[i % len(colors)],
                 label=f"{name} (AUC = {auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(y_test, predictions, model_names,
                            display_labels=("Negative", "Positive"),
                            suptitle=None):
    """Side-by-side confusion matrices for the given models."""
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, model_names):
        ConfusionMatrixDisplay.from_predictions(
            y_test, predictions[name],
            display_labels=list(display_labels),
            ax=ax, cmap="Blues", colorbar=False,
        )
        ax.set_title(name, fontsize=14)
    plt.suptitle(suptitle or f"Confusion Matrices - Top {n} Models", fontsize=16)
    plt.tight_layout()
    plt.show()
