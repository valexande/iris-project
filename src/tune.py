"""Hyperparameter search: Optuna (TPE + median pruning) for transformers, grid search for the baseline."""
import itertools
from dataclasses import replace

import numpy as np
import optuna
import pandas as pd
from transformers import TrainerCallback

from . import evaluate
from .train import TrainConfig, tfidf_logreg, train_transformer
from .utils import SEED


class OptunaPruningCallback(TrainerCallback):
    """Report validation macro-F1 after every epoch; stop unpromising trials early."""

    def __init__(self, trial):
        self.trial = trial

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        step = int(round(state.epoch or 0))
        self.trial.report(metrics["eval_macro_f1"], step)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def suggest(trial, base: TrainConfig) -> TrainConfig:
    return replace(
        base,
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        epochs=trial.suggest_int("epochs", 2, 4),
        warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 0.15),
        weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1),
        max_length=trial.suggest_categorical("max_length", [128, 256, 384]),
        loss=trial.suggest_categorical("loss", ["weighted_ce", "focal"]),
        dropout=trial.suggest_float("dropout", 0.0, 0.3),
        early_stopping_patience=10,  # pruning replaces early stopping during the search
    )


def optuna_search(base: TrainConfig, tr_texts, y_tr, va_texts, y_va, n_classes: int, n_trials: int,
                  timeout: float | None = None):
    def objective(trial):
        cfg = suggest(trial, base)
        scores, info = train_transformer(cfg, tr_texts, y_tr, {"val": (va_texts, y_va)}, n_classes,
                                         f"/tmp/ckpt/optuna_{trial.number}",
                                         callbacks=[OptunaPruningCallback(trial)])
        trial.set_user_attr("train_seconds", info["train_seconds"])
        return evaluate.metrics(y_va, scores["val"], [str(i) for i in range(n_classes)])["macro_f1"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1),
    )
    # Start from the default configuration so the search can only match or beat it.
    study.enqueue_trial({"learning_rate": base.learning_rate, "batch_size": base.batch_size, "epochs": base.epochs,
                         "warmup_ratio": base.warmup_ratio, "weight_decay": base.weight_decay,
                         "max_length": base.max_length, "loss": base.loss, "dropout": 0.1})
    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)
    return study


def baseline_grid(tr_texts, y_tr, va_texts, y_va, names) -> pd.DataFrame:
    rows = []
    grid = itertools.product([1.0, 4.0, 16.0], [1, 2], ["balanced", None])
    for C, ngram, cw in grid:
        scores, secs = tfidf_logreg(tr_texts, y_tr, {"val": va_texts}, C=C, ngram_max=ngram, class_weight=cw)
        m = evaluate.metrics(y_va, scores["val"], names)
        rows.append({"C": C, "ngram_max": ngram, "class_weight": str(cw), "macro_f1": m["macro_f1"],
                     "accuracy": m["accuracy"], "fit_seconds": secs})
        print(rows[-1], flush=True)
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False)


def mean_std(values) -> str:
    """'mean ± std' over seeds; just the value when there is only one run."""
    if len(values) == 1:
        return f"{np.mean(values):.4f} (n=1)"
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"
