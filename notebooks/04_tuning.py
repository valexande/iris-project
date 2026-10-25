# %% [markdown]
# # 04 · Model selection, hyperparameter tuning and final test evaluation
#
# All on the **discovered-k** label set from notebook 02.
#
# Notebook 03 showed one full-size BERT run takes ~80 min on Kaggle's T4s, so every training here uses
# a class-balanced **20k-paper subset** of the training split (validation and test stay full-size).
#
# 1. **Model selection:** `bert-base-uncased` vs `bert-base-cased` vs `distilbert-base-uncased`,
#    all with the default configuration.
#    All three are Apache-2.0 English BERT-family checkpoints with TensorFlow weights, as the task's
#    Hugging Face filter asks.
# 2. **Optuna** (TPE sampler, median pruner) on the winner: learning rate, batch size, epochs, warmup,
#    weight decay, max length, loss (weighted CE vs focal) and dropout. Trials use a 10k-paper subset
#    of that; the first trial is the default configuration.
# 3. **Baseline grid search:** C × n-gram range × class weighting for TF-IDF + logistic regression.
# 4. **Final:** retrain default and tuned configurations on the 20k subset with up to 3 seeds each,
#    then evaluate once on the held-out **test** split (mean ± std).
#
# Runs on a Kaggle GPU (T4 ×2) under a hard 6 h time budget (see `BUDGET_HOURS`).

# %%
import glob
import shutil
import sys
from pathlib import Path


def add_src_to_path():
    hits = glob.glob("/kaggle/input/**/taxonomy.py", recursive=True)
    if hits:
        pkg = Path("/tmp/pkg/src")
        shutil.copytree(Path(hits[0]).parent, pkg, dirs_exist_ok=True)
        sys.path.insert(0, str(pkg.parent))
    else:
        sys.path.insert(0, str(Path.cwd().parent))


add_src_to_path()

from dataclasses import asdict, replace

import time

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import torch

from src import evaluate, taxonomy, viz
from src.train import LabelSpace, TrainConfig, tfidf_logreg, train_transformer
from src.tune import baseline_grid, mean_std, optuna_search
from src.utils import FIG_DIR, RESULTS_DIR, ensure_dirs, find_input, load_json, save_json, set_seed

set_seed()
ensure_dirs()
viz.apply_style()
optuna.logging.set_verbosity(optuna.logging.WARNING)
assert torch.cuda.is_available(), "No GPU: set Accelerator to GPU (needs a phone-verified Kaggle account)"

N_TRIALS = 12
# Kaggle kills a session at 12 h. The notebook keeps its own clock and stops starting new work
# once the next step would not fit in BUDGET_HOURS, so a slow GPU costs runs, never the whole session.
BUDGET_HOURS = 6.0
T0 = time.time()


def hours_left():
    return BUDGET_HOURS - (time.time() - T0) / 3600


TRAIN_SUBSET = 20_000
TUNE_SUBSET = 10_000
SEEDS = [42, 43, 44]

train = pd.read_parquet(find_input("train.parquet"))
val = pd.read_parquet(find_input("val.parquet"))
test = pd.read_parquet(find_input("test.parquet"))
best_k = load_json(find_input("best_k.json"))
discovered = taxonomy.discovered_label(best_k["mapping"])
for df in (train, val, test):
    df["label"] = df["fine"].map(discovered)
train = (train.groupby("label", group_keys=False).sample(frac=TRAIN_SUBSET / len(train), random_state=0)
         .reset_index(drop=True))
sp = LabelSpace(train["label"])
y_tr, y_va, y_te = (sp.encode(d["label"]) for d in (train, val, test))
print("classes:", len(sp), "| train/val/test:", len(train), len(val), len(test))


# %% [markdown]
# ## 1. Model selection

# %%
selection = []
for name in ["bert-base-uncased", "bert-base-cased", "distilbert-base-uncased"]:
    scores, info = train_transformer(TrainConfig(model_name=name), train["text"], y_tr,
                                     {"val": (val["text"], y_va)}, len(sp), f"/tmp/ckpt/sel_{name}")
    m = evaluate.metrics(y_va, scores["val"], sp.names)
    selection.append({"model_name": name, "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                      "train_seconds": info["train_seconds"]})
    print(selection[-1])
    torch.cuda.empty_cache()
selection = pd.DataFrame(selection).sort_values("macro_f1", ascending=False)
selection.to_csv(RESULTS_DIR / "model_selection.csv", index=False)
BEST_MODEL = selection.iloc[0]["model_name"]
print("selected:", BEST_MODEL)
selection

# %% [markdown]
# ## 2. Optuna search

# %%
sub = train.groupby("label", group_keys=False).sample(frac=TUNE_SUBSET / len(train), random_state=0)
# Reserve time for two full final runs (one default, one tuned) plus the baseline grid, using the
# measured duration of a full BERT run from notebook 03 and of the model-selection runs above.
RUN_HOURS = selection["train_seconds"].max() / 3600 * 1.15
optuna_hours = hours_left() - 2 * RUN_HOURS - 0.75
print(f"full run ~ {RUN_HOURS:.2f} h | Optuna budget {optuna_hours:.2f} h")
study = optuna_search(TrainConfig(model_name=BEST_MODEL), sub["text"], sp.encode(sub["label"]),
                      val["text"], y_va, len(sp), N_TRIALS, timeout=max(optuna_hours, 0.25) * 3600)
trials = study.trials_dataframe()
trials.to_csv(RESULTS_DIR / "optuna_trials.csv", index=False)
print("best trial:", study.best_trial.number, study.best_value, study.best_params)
trials[["number", "value", "state"] + [c for c in trials.columns if c.startswith("params_")]].round(5)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
done = trials[trials["state"] == "COMPLETE"]
pruned = trials[trials["state"] == "PRUNED"]
ax = axes[0]
ax.plot(done["number"], done["value"], "o", color=viz.ACCENT, label="completed", mec=viz.SURFACE, mew=1.5, ms=8)
ax.plot(done["number"], done["value"].cummax(), "-", color=viz.TEXT_2, lw=1.5, label="best so far")
if len(pruned):
    ax.plot(pruned["number"], [done["value"].min()] * len(pruned), "x", color=viz.SERIES[1], label="pruned")
ax.set_xlabel("trial")
ax.set_ylabel("validation macro-F1 (10k-paper trials)")
ax.set_title("Optuna optimisation history")
ax.legend(loc="lower right")

try:
    imp = optuna.importance.get_param_importances(study)
except Exception as e:  # needs >= 2 completed trials with varied params
    imp = {}
    print("importance unavailable:", e)
ax = axes[1]
if imp:
    names, vals = list(imp)[::-1], list(imp.values())[::-1]
    ax.barh(names, vals, color=viz.ACCENT, height=0.6)
    ax.set_xlabel("fANOVA importance")
ax.set_title("Which hyperparameters mattered")
ax.grid(axis="y", visible=False)
fig.tight_layout()
viz.save(fig, "fig15_optuna", FIG_DIR)
save_json({"best_params": study.best_params, "best_value": study.best_value, "importance": imp},
          RESULTS_DIR / "optuna_best.json")

# %% [markdown]
# ## 3. Baseline grid search

# %%
grid = baseline_grid(train["text"], y_tr, val["text"], y_va, sp.names)
grid.to_csv(RESULTS_DIR / "baseline_grid.csv", index=False)
grid

# %% [markdown]
# ## 4. Final models on the test set
# Default vs tuned configuration, each retrained on the 20k training subset with up to three seeds (as many as the time budget allows).
# The test split is used here for the first and only time.

# %%
bp = study.best_params
tuned_cfg = replace(TrainConfig(model_name=BEST_MODEL), early_stopping_patience=1,
                    **{k: bp[k] for k in ["learning_rate", "batch_size", "epochs", "warmup_ratio",
                                          "weight_decay", "max_length", "loss", "dropout"]})
configs = {"default": TrainConfig(model_name=BEST_MODEL), "tuned": tuned_cfg}

final_rows, per_class = [], {}
# Seed-major order: the first seed of both configs always runs; later seeds run only if they fit.
seed_hours = 2 * RUN_HOURS
for seed in SEEDS:
    if seed != SEEDS[0] and hours_left() < seed_hours * 1.1 + 0.25:
        print(f"skipping seed {seed}: {hours_left():.2f} h left, a seed takes ~{seed_hours:.2f} h")
        break
    seed_start = time.time()
    for label, cfg in configs.items():
        scores, info = train_transformer(replace(cfg, seed=seed), train["text"], y_tr,
                                         {"val": (val["text"], y_va), "test": (test["text"], y_te)},
                                         len(sp), f"/tmp/ckpt/final_{label}_{seed}")
        for split, y in [("val", y_va), ("test", y_te)]:
            m = evaluate.metrics(y, scores[split], sp.names)
            final_rows.append({"config": label, "seed": seed, "split": split,
                               **{k: v for k, v in m.items() if k != "per_class"}})
            if split == "test" and seed == SEEDS[0]:
                per_class[label] = m["per_class"]
        print(final_rows[-1], f"| {hours_left():.2f} h left", flush=True)
        pd.DataFrame(final_rows).to_csv(RESULTS_DIR / "final_results_runs.csv", index=False)
        torch.cuda.empty_cache()
    seed_hours = (time.time() - seed_start) / 3600

g = grid.iloc[0]
scores, _ = tfidf_logreg(train["text"], y_tr, {"val": val["text"], "test": test["text"]}, C=g["C"],
                         ngram_max=int(g["ngram_max"]), class_weight=None if g["class_weight"] == "None" else "balanced")
for split, y in [("val", y_va), ("test", y_te)]:
    m = evaluate.metrics(y, scores[split], sp.names)
    final_rows.append({"config": "baseline (tuned)", "seed": 0, "split": split,
                       **{k: v for k, v in m.items() if k != "per_class"}})

final = pd.DataFrame(final_rows)
final.to_csv(RESULTS_DIR / "final_results_runs.csv", index=False)
summary = (final.groupby(["config", "split"])[["accuracy", "macro_f1", "micro_f1", "top3_accuracy"]]
           .agg(mean_std).reset_index())
summary.to_csv(RESULTS_DIR / "final_results_summary.csv", index=False)
save_json({"model": BEST_MODEL, "default": asdict(configs["default"]), "tuned": asdict(tuned_cfg),
           "per_class_test": per_class}, RESULTS_DIR / "final_configs.json")
summary

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.6))
t = final[final["split"] == "test"]
order = ["baseline (tuned)", "default", "tuned"]
# Dot plot rather than bars: the axis is zoomed in, and bars must start at zero.
for i, c in enumerate(order):
    v = t.loc[t["config"] == c, "macro_f1"]
    color = [viz.NEUTRAL, viz.SERIES[1], viz.ACCENT][i]
    ax.plot([i] * len(v), v, "o", color=color, alpha=0.35, ms=6)
    ax.errorbar(i, v.mean(), yerr=v.std() if len(v) > 1 else None, fmt="o", color=color, ms=10,
                mec=viz.SURFACE, mew=2, capsize=5, ecolor=viz.TEXT_2, elinewidth=1)
    ax.text(i + 0.12, v.mean(), f"{v.mean():.3f}", va="center", fontsize=9)
ax.set_xticks(range(3), ["TF-IDF + LogReg\n(tuned)", f"{BEST_MODEL}\n(default)", f"{BEST_MODEL}\n(tuned)"])
lo = t["macro_f1"].min()
ax.set_ylim(max(0, lo - 0.08), min(1, t["macro_f1"].max() + 0.04))
ax.set_ylabel("test macro-F1 (mean ± std over seeds)")
ax.set_title(f"Final test results, {len(sp)} discovered categories")
ax.grid(axis="x", visible=False)
viz.save(fig, "fig16_final_test", FIG_DIR)
