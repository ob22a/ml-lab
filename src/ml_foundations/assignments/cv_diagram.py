"""K-fold cross-validation fold diagram for Notebook 07."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_kfold_diagram(n_splits=5, seed=0):
    """Visual diagram of train/validation folds in K-fold CV."""
    rng = np.random.default_rng(seed)
    n_samples = n_splits * 4
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)

    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.55 * n_splits))
    colors = {"train": "#4C72B0", "val": "#DD8452"}

    for fold_idx in range(n_splits):
        y = n_splits - fold_idx - 1
        for i in range(n_samples):
            is_val = i in folds[fold_idx]
            color = colors["val"] if is_val else colors["train"]
            rect = mpatches.Rectangle((i, y), 1, 0.8, facecolor=color, edgecolor="white")
            ax.add_patch(rect)
        ax.text(-1.5, y + 0.4, f"Fold {fold_idx + 1}", va="center", ha="right", fontsize=10)

    ax.set_xlim(-2, n_samples)
    ax.set_ylim(-0.2, n_splits)
    ax.set_xticks(np.arange(n_samples) + 0.5)
    ax.set_xticklabels([str(i) for i in range(n_samples)], fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Sample index (shuffled)")
    ax.set_title(f"{n_splits}-Fold Cross-Validation: orange = validation fold")

    train_patch = mpatches.Patch(color=colors["train"], label="Training")
    val_patch = mpatches.Patch(color=colors["val"], label="Validation")
    ax.legend(handles=[train_patch, val_patch], loc="upper right")
    plt.tight_layout()
    plt.show()
