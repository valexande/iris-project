# %% [markdown]
# # 03 · Training and evaluation
#
# Models, each trained on four label granularities:
#
# | label set | classes | source |
# |---|---|---|
# | `group` | 8 | arXiv top-level groups |
# | `archive` | ~20 | arXiv archives |
# | `discovered` | k from notebook 02 | arXiv categories merged by abstract similarity |
# | `fine` | ~140 | arXiv primary categories |
#
# * **Baseline:** TF-IDF (1–2-grams) + logistic regression.
# * **Transformer:** `bert-base-uncased` fine-tuned with Hugging Face `Trainer` — max length 256,
#   lr 2e-5, effective batch 32, 3 epochs, 10% warmup, class-weighted cross-entropy, fp16,
#   early stopping on validation macro-F1.
#
# Evaluation is on the **validation** split; the test split is reserved for the final model in notebook 04.
# Runs on a Kaggle GPU (T4 ×2, ~2 h).

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src import evaluate, taxonomy, viz
from src.train import LabelSpace, TrainConfig, tfidf_logreg, train_transformer
from src.utils import DATA_DIR, FIG_DIR, RESULTS_DIR, ensure_dirs, find_input, load_json, save_json, set_seed

set_seed()
ensure_dirs()
viz.apply_style()
print("GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert torch.cuda.is_available(), "No GPU: set Accelerator to GPU (needs a phone-verified Kaggle account)"

train = pd.read_parquet(find_input("train.parquet"))
val = pd.read_parquet(find_input("val.parquet"))
best_k = load_json(find_input("best_k.json"))
K = best_k["best_k"]
discovered = taxonomy.discovered_label(best_k["mapping"])
for df in (train, val):
    df["discovered"] = df["fine"].map(discovered)

LEVELS = ["group", "archive", "discovered", "fine"]
spaces = {lvl: LabelSpace(train[lvl]) for lvl in LEVELS}
print({lvl: len(s) for lvl, s in spaces.items()}, "| discovered k =", K)

# %% [markdown]
# ## Token lengths: is 256 enough?

# %%
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
lengths = np.array([len(x) for x in tok(train["text"].sample(20000, random_state=0).tolist())["input_ids"]])
share_256 = float((lengths <= 256).mean())
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.hist(np.clip(lengths, 0, 600), bins=80, color=viz.ACCENT)
ax.axvline(256, color=viz.TEXT, lw=1, ls="--")
ax.text(262, ax.get_ylim()[1] * 0.9, f"256 tokens covers {share_256:.0%}", fontsize=9)
ax.set_xlabel("BERT wordpiece tokens (title + abstract, clipped at 600)")
ax.set_ylabel("papers")
ax.set_title("Most inputs fit in 256 tokens")
ax.grid(axis="x", visible=False)
viz.save(fig, "fig12_token_lengths", FIG_DIR)
print("median", np.median(lengths), "p95", np.percentile(lengths, 95), "share<=256", share_256)

# %% [markdown]
# ## Smoke test
# A few steps on a tiny slice, so a bug fails in one minute rather than after an hour.

# %%
small = train.sample(256, random_state=0)
_ = train_transformer(TrainConfig(epochs=1, max_length=64), small["text"], spaces["group"].encode(small["group"]),
                      {"val": (small["text"][:64], spaces["group"].encode(small["group"][:64]))},
                      len(spaces["group"]), "/tmp/ckpt/smoke")
print("smoke test ok")

# %% [markdown]
# ## Baseline: TF-IDF + logistic regression

# %%
rows, all_metrics = [], {}
for lvl in LEVELS:
    sp = spaces[lvl]
    scores, secs = tfidf_logreg(train["text"], sp.encode(train[lvl]), {"val": val["text"]})
    m = evaluate.metrics(sp.encode(val[lvl]), scores["val"], sp.names)
    all_metrics[f"tfidf_logreg/{lvl}"] = m
    rows.append({"model": "TF-IDF + LogReg", "label_set": lvl, "n_classes": len(sp), "train_seconds": secs,
                 **{k: v for k, v in m.items() if k != "per_class"}})
    print(rows[-1])

# %% [markdown]
# ## BERT fine-tuning

# %%
cfg = TrainConfig()  # defaults listed in the header
bert_scores, bert_info = {}, {}
for lvl in LEVELS:
    sp = spaces[lvl]
    scores, info = train_transformer(cfg, train["text"], sp.encode(train[lvl]),
                                     {"val": (val["text"], sp.encode(val[lvl]))}, len(sp), f"/tmp/ckpt/bert_{lvl}")
    m = evaluate.metrics(sp.encode(val[lvl]), scores["val"], sp.names)
    all_metrics[f"bert/{lvl}"] = m
    bert_scores[lvl], bert_info[lvl] = scores["val"], info
    np.save(DATA_DIR / f"val_logits_bert_{lvl}.npy", scores["val"])
    rows.append({"model": "BERT-base", "label_set": lvl, "n_classes": len(sp), "train_seconds": info["train_seconds"],
                 **{k: v for k, v in m.items() if k != "per_class"}})
    print(rows[-1])
    torch.cuda.empty_cache()

results = pd.DataFrame(rows)
results.to_csv(RESULTS_DIR / "model_results.csv", index=False)
save_json({"metrics": all_metrics, "bert_training": bert_info, "discovered_k": K}, RESULTS_DIR / "model_results_detail.json")
results[["model", "label_set", "n_classes", "accuracy", "macro_f1", "micro_f1", "top3_accuracy", "train_seconds"]].round(4)

# %% [markdown]
# ## Macro-F1 against the number of classes

# %%
fig, ax = plt.subplots(figsize=(7, 4))
for (model, d), color in zip(results.groupby("model", sort=False), viz.SERIES):
    d = d.sort_values("n_classes")
    ax.plot(d["n_classes"], d["macro_f1"], "-o", color=color, label=model, mec=viz.SURFACE, mew=2, ms=8)
    for _, r in d.iterrows():
        ax.annotate(f"{r['label_set']} ({r['n_classes']})", (r["n_classes"], r["macro_f1"]),
                    textcoords="offset points", xytext=(6, 6 if model == "BERT-base" else -12),
                    fontsize=8, color=viz.TEXT_2)
ax.set_xscale("log")
ax.set_xlabel("number of classes (log scale)")
ax.set_ylabel("validation macro-F1")
ax.set_title("Finer label sets are harder; BERT leads at every granularity")
ax.legend(loc="lower left")
viz.save(fig, "fig13_f1_vs_classes", FIG_DIR)

# %% [markdown]
# ## Confusion matrices (BERT)

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
for ax, lvl in zip(axes, ["group", "archive", "discovered"]):
    sp = spaces[lvl]
    evaluate.plot_confusion(sp.encode(val[lvl]), bert_scores[lvl].argmax(1), sp.names,
                            f"{lvl} ({len(sp)} classes)", ax=ax)
fig.tight_layout()
viz.save(fig, "fig14_confusion", FIG_DIR)

# %%
# Do the model's mistakes follow arXiv cross-listing? Compare the most-confused fine pairs.
sp = spaces["fine"]
confused = evaluate.most_confused(sp.encode(val["fine"]), bert_scores["fine"].argmax(1), sp.names, n=15)
confused_df = pd.DataFrame(confused, columns=["true", "predicted", "count"])
confused_df.to_csv(RESULTS_DIR / "most_confused_fine.csv", index=False)
confused_df

# %%
# Weakest and strongest fine categories.
pc = pd.DataFrame(all_metrics["bert/fine"]["per_class"]).T.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
pc = pc.sort_values("f1-score")
pd.concat([pc.head(8), pc.tail(8)])[["precision", "recall", "f1-score", "support"]].round(3)
