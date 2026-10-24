"""Loading, cleaning, labelling and splitting the arXiv metadata snapshot."""
import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .taxonomy import archive_of, canonical, group_of, is_ambiguous_fine
from .utils import SEED

FIELDS = ["id", "title", "abstract", "categories", "comments", "update_date"]


def load_raw(path, limit=None) -> pd.DataFrame:
    """Stream the ~5 GB JSON-lines snapshot, keeping only the fields we use."""
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rec = json.loads(line)
            row = {k: rec.get(k) for k in FIELDS}
            versions = rec.get("versions") or []
            # Year of the first version = year the paper first appeared on arXiv.
            first = versions[0]["created"] if versions else ""
            m = re.search(r"\b(19|20)\d{2}\b", first)
            row["year"] = int(m.group(0)) if m else None
            rows.append(row)
    df = pd.DataFrame(rows)
    df["year"] = df["year"].astype("Int64")
    return df


def word_count(texts: pd.Series) -> pd.Series:
    """Whitespace word count without materialising per-row word lists (matters at 3M rows)."""
    return texts.fillna("").str.count(r"\S+")


# --- text cleaning -----------------------------------------------------------

_MATH = re.compile(
    r"\$\$.+?\$\$|\$.+?\$|\\\[.+?\\\]|\\\(.+?\\\)|\\begin\{(equation|align|eqnarray)\*?\}.+?\\end\{\1\*?\}",
    re.S,
)
_CITE = re.compile(r"\\(cite|ref|eqref|label)\w*\{[^}]*\}")
_CMD_WITH_ARG = re.compile(r"\\[a-zA-Z]+\*?\{([^{}]*)\}")  # \emph{word} -> word
_CMD = re.compile(r"\\[a-zA-Z]+\*?")
_URL = re.compile(r"https?://\S+|www\.\S+")
_BRACKET_CITE = re.compile(r"\[\d+(?:\s*[,-]\s*\d+)*\]")  # [12], [3-5], [1, 4]
_SPACE = re.compile(r"\s+")

MATH_TOKEN = "[MATH]"


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _URL.sub(" ", text)
    text = _CITE.sub(" ", text)
    text = _MATH.sub(f" {MATH_TOKEN} ", text)
    text = _BRACKET_CITE.sub(" ", text)
    for _ in range(2):  # nested commands, e.g. \textbf{\emph{x}}
        text = _CMD_WITH_ARG.sub(r"\1", text)
    text = _CMD.sub(" ", text)
    text = text.replace("{", "").replace("}", "").replace("~", " ")
    text = _SPACE.sub(" ", text).strip()
    # Collapse runs like "[MATH] [MATH]" left by consecutive formulas.
    return re.sub(rf"(?:{re.escape(MATH_TOKEN)}\s*){{2,}}", f"{MATH_TOKEN} ", text).strip()


_WITHDRAWN = re.compile(
    r"\b(this|the)\s+(paper|article|manuscript|submission|preprint|work)\s+(has\s+been|is|was)\s+withdrawn"
    r"|^\s*withdrawn\b|\bwithdrawn\s+by\s+the\s+author",
    re.I,
)


def is_withdrawn(abstract, comments) -> bool:
    return bool(_WITHDRAWN.search(abstract or "")) or bool(re.search(r"\bwithdrawn\b", comments or "", re.I))


# --- labels ------------------------------------------------------------------

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Primary (first-listed) category, canonicalised, plus its archive and group."""
    cats = df["categories"].fillna("").str.split()
    df["n_categories"] = cats.str.len()
    df["fine"] = cats.str[0].map(canonical)
    df["archive"] = df["fine"].map(archive_of)
    df["group"] = df["fine"].map(group_of)
    return df


def clean_and_filter(df: pd.DataFrame, min_words: int = 20):
    """Apply all row filters and text cleaning. Returns (clean_df, counts of rows removed per reason)."""
    # Written for ~3M rows on a 30 GB machine: filter with masks and keep only the columns
    # later steps need, instead of copying the whole frame at every step.
    removed = {"input": len(df)}

    withdrawn = (df["abstract"].fillna("").str.contains(_WITHDRAWN)
                 | df["comments"].fillna("").str.contains(r"\bwithdrawn\b", case=False, regex=True))
    removed["withdrawn"] = int(withdrawn.sum())
    out = df.loc[~withdrawn, ["id", "fine", "archive", "group", "year"]].copy()

    abstract = df.loc[~withdrawn, "abstract"].map(clean_text)
    out["n_words"] = word_count(abstract)
    out["text"] = (df.loc[~withdrawn, "title"].map(clean_text) + ". " + abstract).str.strip()
    del abstract

    short = out["n_words"] < min_words
    removed[f"abstract_under_{min_words}_words"] = int(short.sum())
    out = out[~short]

    # Hash the normalised text: 8 bytes per row instead of a second copy of every abstract.
    key = pd.util.hash_pandas_object(out["text"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True), index=False)
    dup = key.duplicated().to_numpy()
    removed["duplicate_text"] = int(dup.sum())
    out = out[~dup]

    amb = out["fine"].map(is_ambiguous_fine)
    removed["ambiguous_bare_archive_primary"] = int(amb.sum())
    out = out[~amb]

    removed["kept"] = len(out)
    return out.reset_index(drop=True), removed


def sample_and_split(df: pd.DataFrame, min_per_class: int, cap_per_class: int,
                     val_frac: float = 0.1, test_frac: float = 0.1, seed: int = SEED):
    """Drop rare fine categories, cap each remaining one, then stratified train/val/test split.

    Capping keeps the fine-level classes balanced so small fields are not drowned
    out by cs.LG / hep-ph, and keeps GPU time small.
    """
    counts = df["fine"].value_counts()
    keep = counts[counts >= min_per_class].index
    dropped = counts[counts < min_per_class]
    df = df[df["fine"].isin(keep)]

    sampled = df.sample(frac=1, random_state=seed).groupby("fine").head(cap_per_class).reset_index(drop=True)
    train, rest = train_test_split(sampled, test_size=val_frac + test_frac,
                                   stratify=sampled["fine"], random_state=seed)
    val, test = train_test_split(rest, test_size=test_frac / (val_frac + test_frac),
                                 stratify=rest["fine"], random_state=seed)
    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True), dropped)


def token_lengths(texts, tokenizer=None, sample: int = 20000, seed: int = SEED) -> np.ndarray:
    """Approximate token counts (wordpiece if a tokenizer is given, else ~1.3 tokens/word)."""
    texts = pd.Series(texts).sample(n=min(sample, len(texts)), random_state=seed)
    if tokenizer is None:
        return (word_count(texts) * 1.3).to_numpy()
    return np.array([len(ids) for ids in tokenizer(list(texts), add_special_tokens=True)["input_ids"]])
