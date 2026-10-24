"""Choosing the number of categories k from the abstracts themselves.

Two complementary views, both computed on sentence embeddings of `title. abstract`
from encoders that never saw the arXiv labels (no circularity):

* Label-free: k-means on the paper embeddings, sweep k, internal validity indices.
* Taxonomy-guided: Ward-cluster the ~140 fine arXiv categories by their centroid
  embeddings. Cutting the dendrogram at k gives a k-way grouping of real arXiv
  categories (so classes stay tied to subject categories), which we score by how
  separable the papers are (silhouette, Davies-Bouldin, linear-probe macro-F1).
"""
import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, f1_score,
                             normalized_mutual_info_score, silhouette_score)

from .utils import SEED

ENCODERS = {
    # SPECTER2 base: scientific-document encoder, CLS pooling on "title [SEP] abstract".
    "specter2": {"name": "allenai/specter2_base", "pooling": "cls"},
    # General-purpose sentence encoder as a robustness check.
    "minilm": {"name": "sentence-transformers/all-MiniLM-L6-v2", "pooling": "mean"},
}


@torch.no_grad()
def embed(texts, encoder: str, batch_size: int = 128, max_length: int = 256) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    cfg = ENCODERS[encoder]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["name"])
    model = AutoModel.from_pretrained(cfg["name"]).to(device).eval()
    if device == "cuda":
        model.half()
    texts = list(texts)
    order = np.argsort([len(t) for t in texts])  # sort by length -> less padding
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        idx = order[start:start + batch_size]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").to(device)
        hidden = model(**enc).last_hidden_state
        if cfg["pooling"] == "cls":
            vec = hidden[:, 0]
        else:
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            vec = (hidden * mask).sum(1) / mask.sum(1)
        out[idx] = vec.float().cpu().numpy()
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def reduce(X: np.ndarray, dim: int = 64) -> np.ndarray:
    """PCA before clustering: faster k-means and less noisy distances."""
    Z = PCA(n_components=dim, random_state=SEED).fit_transform(X)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def internal_scores(X: np.ndarray, labels: np.ndarray, sample: int = 10000) -> dict:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    return {
        "silhouette": float(silhouette_score(X[idx], labels[idx], metric="cosine")),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


def kmeans_sweep(X: np.ndarray, ks, reference: dict) -> pd.DataFrame:
    """Label-free sweep. `reference` maps a name (e.g. 'group') to arXiv labels, for NMI/ARI only."""
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=4, random_state=SEED).fit(X)
        row = {"k": k, "inertia": float(km.inertia_), **internal_scores(X, km.labels_)}
        for name, y in reference.items():
            row[f"nmi_{name}"] = float(normalized_mutual_info_score(y, km.labels_))
            row[f"ari_{name}"] = float(adjusted_rand_score(y, km.labels_))
        rows.append(row)
        print(f"kmeans k={k:3d} sil={row['silhouette']:.4f} db={row['davies_bouldin']:.3f}", flush=True)
    return pd.DataFrame(rows)


def category_linkage(X: np.ndarray, fine: np.ndarray):
    """Ward linkage over fine-category centroids. Returns (linkage matrix, category names)."""
    cats = np.array(sorted(set(fine)))
    C = np.stack([X[fine == c].mean(0) for c in cats])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return linkage(C, method="ward"), cats


def cut(Z, cats, k: int) -> dict:
    """fine category -> cluster id (0..k-1) for a k-way cut of the category dendrogram."""
    ids = fcluster(Z, t=k, criterion="maxclust")
    return {c: int(i) - 1 for c, i in zip(cats, ids)}


def linear_probe_f1(X_tr, y_tr, X_va, y_va, epochs: int = 30, lr: float = 1e-2) -> float:
    """Macro-F1 of a softmax regression on frozen embeddings (GPU, full-batch Adam)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_cls = int(max(y_tr.max(), y_va.max())) + 1
    Xt = torch.tensor(X_tr, device=device)
    yt = torch.tensor(y_tr, device=device)
    counts = torch.bincount(yt, minlength=n_cls).float()
    weight = (counts.sum() / (n_cls * counts.clamp(min=1)))
    lin = torch.nn.Linear(Xt.shape[1], n_cls).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    for _ in range(epochs * 10):
        opt.zero_grad()
        loss_fn(lin(Xt * 10), yt).backward()  # x10: unit-norm inputs need a larger logit scale
        opt.step()
    with torch.no_grad():
        pred = lin(torch.tensor(X_va, device=device) * 10).argmax(1).cpu().numpy()
    return float(f1_score(y_va, pred, average="macro"))


def taxonomy_sweep(X_tr, fine_tr, X_va, fine_va, Z, cats, ks) -> pd.DataFrame:
    rows = []
    for k in ks:
        mapping = cut(Z, cats, k)
        y_tr = np.array([mapping[c] for c in fine_tr])
        y_va = np.array([mapping[c] for c in fine_va])
        row = {"k": k, **internal_scores(X_tr, y_tr), "probe_macro_f1": linear_probe_f1(X_tr, y_tr, X_va, y_va)}
        rows.append(row)
        print(f"ward k={k:3d} sil={row['silhouette']:.4f} f1={row['probe_macro_f1']:.3f}", flush=True)
    return pd.DataFrame(rows)


def merge_gap_k(Z, k_min: int, k_max: int) -> int:
    """k whose dendrogram cut has the largest gap to the next merge (the most 'stable' cut)."""
    h = Z[:, 2]
    n = len(h) + 1
    gaps = {k: h[n - k] - h[n - k - 1] for k in range(max(k_min, 2), min(k_max, n - 1) + 1)}
    return max(gaps, key=gaps.get)


def knee(ks, values) -> int:
    """Kneedle: point of a monotone curve farthest from the chord between its endpoints."""
    x = np.asarray(ks, dtype=float)
    y = np.asarray(values, dtype=float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min() + 1e-12)
    d = np.abs(yn - (yn[0] + (yn[-1] - yn[0]) * xn))
    return int(x[np.argmax(d)])


def choose_k(kmeans_df: pd.DataFrame, tax_df: pd.DataFrame, Z, k_min: int, k_max: int) -> dict:
    """Each criterion proposes a k in [k_min, k_max]; the reported best k is their median."""
    km = kmeans_df.query("@k_min <= k <= @k_max")
    tx = tax_df.query("@k_min <= k <= @k_max")
    votes = {
        "kmeans_silhouette_max": int(km.loc[km["silhouette"].idxmax(), "k"]),
        "kmeans_davies_bouldin_min": int(km.loc[km["davies_bouldin"].idxmin(), "k"]),
        "taxonomy_silhouette_max": int(tx.loc[tx["silhouette"].idxmax(), "k"]),
        "taxonomy_davies_bouldin_min": int(tx.loc[tx["davies_bouldin"].idxmin(), "k"]),
        "taxonomy_probe_f1_knee": knee(tx["k"], tx["probe_macro_f1"]),
        "dendrogram_merge_gap": merge_gap_k(Z, k_min, k_max),
    }
    best = int(np.median(list(votes.values())))
    return {"votes": votes, "best_k": best}
