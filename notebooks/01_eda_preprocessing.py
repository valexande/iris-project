# %% [markdown]
# # 01 · Exploratory data analysis and preprocessing
#
# Input: the full arXiv metadata snapshot (`Cornell-University/arxiv` on Kaggle).
# Output: EDA figures and statistics, plus `train/val/test.parquet` used by the later notebooks.
#
# Runs on a CPU session (~15 min); no GPU needed.

# %%
import gc
import glob
import shutil
import sys
from pathlib import Path


def add_src_to_path():
    """Locally the repo root is the parent of notebooks/; on Kaggle `src` is attached as a dataset."""
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
from scipy import sparse

from src import preprocess, taxonomy, viz
from src.utils import DATA_DIR, FIG_DIR, RESULTS_DIR, ensure_dirs, find_input, save_json, set_seed

set_seed()
ensure_dirs()
viz.apply_style()
pd.set_option("display.max_columns", 20)

MIN_PER_CLASS = 500  # drop fine categories with fewer primary papers than this
CAP_PER_CLASS = 500  # papers sampled per fine category
stats = {}

# %% [markdown]
# ## 1. Load
# The snapshot is JSON-lines, one record per paper. We keep id, title, abstract, categories,
# comments (used to spot withdrawals), last update date and the year of the first version.

# %%
raw_path = find_input("arxiv-metadata-oai-snapshot.json")
df = preprocess.load_raw(raw_path)
df = preprocess.add_labels(df)
stats["n_papers_raw"] = len(df)
print(f"{len(df):,} papers")
df.head(3)

# %%
df.isna().sum().to_frame("missing")

# %% [markdown]
# ## 2. How many categories are there?
# The primary category is the first one listed. Legacy names (e.g. `alg-geom`) and official
# aliases (e.g. `math.MP` = `math-ph`) are mapped to their canonical modern category.

# %%
levels = {lvl: df[lvl].nunique() for lvl in ["group", "archive", "fine"]}
stats["n_classes_raw"] = levels
print(levels)

group_counts = df["group"].value_counts()
stats["group_counts"] = group_counts.to_dict()

fig, ax = plt.subplots(figsize=(7, 3.6))
order = list(group_counts.index[::-1])  # largest at the top
ax.barh(order, group_counts[order], color=[viz.GROUP_COLORS[g] for g in order], height=0.65)
for i, g in enumerate(order):
    ax.text(group_counts[g], i, f"  {group_counts[g] / len(df):.1%}", va="center", fontsize=9, color=viz.TEXT_2)
ax.set_xlabel("papers (primary category)")
ax.set_title(f"Physics, CS and mathematics make up {group_counts.head(3).sum() / len(df):.0%} of arXiv")
ax.grid(axis="y", visible=False)
ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1e6:.1f}M")
viz.save(fig, "fig01_groups", FIG_DIR)

# %% [markdown]
# ### The long tail of fine categories

# %%
fine_counts = df.loc[~df["fine"].map(taxonomy.is_ambiguous_fine), "fine"].value_counts()
stats["fine_top10"] = fine_counts.head(10).to_dict()
stats["fine_bottom10"] = fine_counts.tail(10).to_dict()
stats["fine_counts"] = fine_counts.to_dict()
n_kept = int((fine_counts >= MIN_PER_CLASS).sum())

fig, ax = plt.subplots(figsize=(7.5, 3.8))
colors = [viz.GROUP_COLORS[taxonomy.group_of(c)] for c in fine_counts.index]
ax.bar(np.arange(len(fine_counts)), fine_counts.values, color=colors, width=0.9)
ax.set_yscale("log")
ax.axhline(MIN_PER_CLASS, color=viz.TEXT_2, lw=1, ls="--")
ax.text(2, MIN_PER_CLASS * 0.75, f"cut-off: {MIN_PER_CLASS} papers ({n_kept} categories kept)",
        va="top", fontsize=9, color=viz.TEXT_2)
top3 = ", ".join(f"{c} {n / 1e3:.0f}k" for c, n in fine_counts.head(3).items())
ax.text(len(fine_counts) * 0.98, fine_counts.iloc[0] * 0.9, f"largest: {top3}", ha="right", va="top",
        fontsize=9, color=viz.TEXT_2)
ax.set_xlabel("fine category, ranked by size")
ax.set_ylabel("papers (log scale)")
ax.set_title(f"{len(fine_counts)} fine categories with a long tail")
handles = [plt.Rectangle((0, 0), 1, 1, color=viz.GROUP_COLORS[g]) for g in viz.GROUP_ORDER]
ax.legend(handles, viz.GROUP_ORDER, ncol=2, fontsize=8, loc="upper right", bbox_to_anchor=(1, 0.88))
ax.set_ylim(top=fine_counts.iloc[0] * 3)
ax.grid(axis="x", visible=False)
viz.save(fig, "fig02_long_tail", FIG_DIR)
fine_counts.head(15)

# %% [markdown]
# ## 3. Cross-listing
# Papers can list several categories. This matters for the "how many categories" question:
# heavily co-listed fields are hard to separate.

# %%
ncat = df["n_categories"].clip(upper=6)
cross_rate = float((df["n_categories"] > 1).mean())
stats["cross_listed_share"] = cross_rate
stats["mean_categories_per_paper"] = float(df["n_categories"].mean())

fig, ax = plt.subplots(figsize=(6, 3.2))
vc = ncat.value_counts().sort_index()
ax.bar(vc.index.astype(str).str.replace("6", "6+"), vc.values / len(df), color=viz.ACCENT, width=0.6)
ax.set_xlabel("categories listed per paper")
ax.set_ylabel("share of papers")
ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
ax.set_title(f"{cross_rate:.0%} of papers are cross-listed")
ax.grid(axis="x", visible=False)
viz.save(fig, "fig03_cross_listing", FIG_DIR)

# %%
# Archive-level co-occurrence: P(paper also lists archive B | it lists archive A).
arch_lists = df["categories"].str.split().map(lambda cs: sorted({taxonomy.archive_of(taxonomy.canonical(c)) for c in cs}))
archives = sorted({a for lst in arch_lists for a in lst})
idx = {a: i for i, a in enumerate(archives)}
rows = np.repeat(np.arange(len(arch_lists)), arch_lists.str.len())
cols = np.fromiter((idx[a] for lst in arch_lists for a in lst), dtype=np.int64)
M = sparse.csr_matrix((np.ones_like(cols, dtype=np.float32), (rows, cols)), shape=(len(arch_lists), len(archives)))
C = (M.T @ M).toarray()
del arch_lists, rows, cols, M
P = C / np.diag(C)[:, None]
np.fill_diagonal(P, np.nan)

fig, ax = plt.subplots(figsize=(8, 6.8))
im = ax.imshow(P, cmap=viz.SEQ_BLUE, vmin=0, vmax=np.nanpercentile(P, 99))
ax.set_xticks(range(len(archives)), archives, rotation=90)
ax.set_yticks(range(len(archives)), archives)
ax.grid(False)
ax.set_xlabel("also lists archive B")
ax.set_ylabel("paper lists archive A")
ax.set_title("Cross-listing between archives, P(B | A)")
fig.colorbar(im, ax=ax, shrink=0.75, label="share of A's papers")
viz.save(fig, "fig04_cooccurrence", FIG_DIR)

top_pairs = (pd.DataFrame(P, index=archives, columns=archives).stack().sort_values(ascending=False).head(10))
stats["top_cooccurrence"] = {f"{a}->{b}": round(float(v), 3) for (a, b), v in top_pairs.items()}
top_pairs

# %% [markdown]
# ## 4. Abstract length and growth over time

# %%
words = preprocess.word_count(df["abstract"])
stats["abstract_words"] = words.describe(percentiles=[0.05, 0.5, 0.95, 0.99]).round(1).to_dict()

fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.hist(words.clip(upper=400), bins=100, color=viz.ACCENT)
med = words.median()
ax.axvline(med, color=viz.TEXT, lw=1, ls="--")
ax.text(med + 8, ax.get_ylim()[1] * 0.9, f"median {med:.0f} words", fontsize=9)
ax.set_xlabel("abstract length (words, clipped at 400)")
ax.set_ylabel("papers")
ax.set_title(f"Abstracts are short: median {med:.0f} words, 99% under {words.quantile(0.99):.0f}")
ax.grid(axis="x", visible=False)
viz.save(fig, "fig05_abstract_length", FIG_DIR)

# %%
per_year = (df.dropna(subset=["year"]).query("1991 <= year < @df.year.max()")  # arXiv opened in 1991; last year is partial
            .groupby(["year", "group"]).size().unstack(fill_value=0))
per_year = per_year[[g for g in viz.GROUP_ORDER if g in per_year.columns]]
stats["papers_per_year_last5"] = per_year.sum(axis=1).tail(5).to_dict()

fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.stackplot(per_year.index.astype(int), per_year.T.values, labels=per_year.columns,
             colors=[viz.GROUP_COLORS[g] for g in per_year.columns], edgecolor=viz.SURFACE, linewidth=0.5)
ax.set_ylabel("new papers per year")
ax.yaxis.set_major_formatter(lambda x, _: f"{x / 1e3:.0f}k")
ax.set_title("Submissions keep growing, driven by computer science")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="upper left", fontsize=8)  # same top-to-bottom order as the stack
ax.set_xlim(per_year.index.min(), per_year.index.max())
ax.grid(axis="x", visible=False)
viz.save(fig, "fig06_growth", FIG_DIR)

# %% [markdown]
# ## 5. Preprocessing
# 1. Remove withdrawn papers (abstract or comments say so).
# 2. Clean text: LaTeX math becomes `[MATH]`; citations, URLs, LaTeX commands and extra whitespace are removed.
# 3. Remove abstracts under 20 words and exact duplicate texts.
# 4. Remove papers whose primary is a bare archive that has sub-categories (e.g. pre-2009 `astro-ph`),
#    which would overlap every `astro-ph.*` class.
# 5. Model input = `title. abstract`.

# %%
clean, removed = preprocess.clean_and_filter(df)
stats["preprocessing_removed"] = removed
print(removed)
ex = clean.sample(1, random_state=0).iloc[0]
print("RAW:  ", df.loc[df["id"] == ex["id"], "abstract"].iloc[0][:400])
print("CLEAN:", ex["text"][:400])

del df  # free the raw frame (~10 GB) before sampling
gc.collect()

# %% [markdown]
# ## 6. Sampling and splits
# Fine categories with fewer than 500 primary papers are dropped; each remaining one contributes
# up to 500 papers. Splits are stratified by fine category (80/10/10). The archive and group labels
# are derived from the fine label, so the same split serves every label granularity.

# %%
train, val, test, dropped = preprocess.sample_and_split(clean, MIN_PER_CLASS, CAP_PER_CLASS)
stats["dropped_rare_categories"] = dropped.to_dict()
stats["split_sizes"] = {"train": len(train), "val": len(val), "test": len(test)}
stats["n_classes_sample"] = {lvl: int(train[lvl].nunique()) for lvl in ["group", "archive", "fine"]}
print(stats["split_sizes"], stats["n_classes_sample"])

cols = ["id", "text", "fine", "archive", "group", "year", "n_words"]
for name, part in [("train", train), ("val", val), ("test", test)]:
    part[cols].to_parquet(DATA_DIR / f"{name}.parquet", index=False)

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})
g = train["group"].value_counts()
axes[0].barh(g.index[::-1], g.values[::-1], color=[viz.GROUP_COLORS[x] for x in g.index[::-1]], height=0.65)
axes[0].set_title("Training sample by group")
axes[0].set_xlabel("papers")
axes[0].grid(axis="y", visible=False)
a = train["archive"].value_counts()
axes[1].bar(a.index, a.values, color=[viz.GROUP_COLORS[taxonomy.group_of(x)] for x in a.index], width=0.7)
axes[1].tick_params(axis="x", rotation=90)
axes[1].set_title("…and by archive")
axes[1].grid(axis="x", visible=False)
fig.tight_layout()
viz.save(fig, "fig07_sample_classes", FIG_DIR)

save_json(stats, RESULTS_DIR / "eda_stats.json")
stats
