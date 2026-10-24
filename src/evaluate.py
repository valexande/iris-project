"""Classification metrics and confusion-matrix helpers."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from . import viz


def top_k_accuracy(y_true, scores, k: int = 3) -> float:
    topk = np.argsort(-scores, axis=1)[:, :k]
    return float((topk == np.asarray(y_true)[:, None]).any(axis=1).mean())


def metrics(y_true, scores, names) -> dict:
    """`scores` is (n, n_classes): logits, probabilities or decision values."""
    y_pred = scores.argmax(1)
    labels = np.arange(len(names))
    return {
        "n_classes": len(names),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "top3_accuracy": top_k_accuracy(y_true, scores, 3) if len(names) > 3 else 1.0,
        "per_class": classification_report(y_true, y_pred, labels=labels, target_names=list(names),
                                           output_dict=True, zero_division=0),
    }


def most_confused(y_true, y_pred, names, n: int = 10):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(names)))
    np.fill_diagonal(cm, 0)
    flat = np.argsort(-cm, axis=None)[:n]
    return [(names[i], names[j], int(cm[i, j])) for i, j in zip(*np.unravel_index(flat, cm.shape)) if cm[i, j] > 0]


def plot_confusion(y_true, y_pred, names, title, ax=None):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(names))).astype(float)
    cm /= cm.sum(1, keepdims=True).clip(min=1)
    ax = ax or plt.subplots(figsize=(7, 6))[1]
    im = ax.imshow(cm, cmap=viz.SEQ_BLUE, vmin=0, vmax=1)
    if len(names) <= 30:
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=7)
        if len(names) <= 12:
            for i in range(len(names)):
                for j in range(len(names)):
                    if cm[i, j] >= 0.01:
                        ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                                color="white" if cm[i, j] > 0.5 else viz.TEXT)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    plt.colorbar(im, ax=ax, shrink=0.7, label="row-normalised share")
    return ax
