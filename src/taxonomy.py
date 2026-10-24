"""arXiv category taxonomy: fine category -> archive -> top-level group.

Reference: https://arxiv.org/category_taxonomy
"""

# Pre-1998 archives that were later folded into modern categories.
LEGACY = {
    "acc-phys": "physics.acc-ph",
    "adap-org": "nlin.AO",
    "alg-geom": "math.AG",
    "ao-sci": "physics.ao-ph",
    "atom-ph": "physics.atom-ph",
    "bayes-an": "physics.data-an",
    "chao-dyn": "nlin.CD",
    "chem-ph": "physics.chem-ph",
    "cmp-lg": "cs.CL",
    "comp-gas": "nlin.CG",
    "dg-ga": "math.DG",
    "funct-an": "math.FA",
    "mtrl-th": "cond-mat.mtrl-sci",
    "patt-sol": "nlin.PS",
    "plasm-ph": "physics.plasm-ph",
    "q-alg": "math.QA",
    "solv-int": "nlin.SI",
    "supr-con": "cond-mat.supr-con",
}

# Official aliases: the same category under two names. Map to the canonical one.
ALIASES = {
    "cs.NA": "math.NA",
    "cs.SY": "eess.SY",
    "math.IT": "cs.IT",
    "math.MP": "math-ph",
    "math.ST": "stat.TH",
    "q-fin.EC": "econ.GN",
}

PHYSICS_ARCHIVES = {
    "astro-ph", "cond-mat", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th",
    "math-ph", "nlin", "nucl-ex", "nucl-th", "physics", "quant-ph",
}

GROUPS = {
    "math": "Mathematics",
    "cs": "Computer Science",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "stat": "Statistics",
    "eess": "Electrical Eng. & Systems Sci.",
    "econ": "Economics",
}

# Archives that have sub-categories; a bare "astro-ph" primary (common before 2009)
# is ambiguous at the fine level because it overlaps all astro-ph.* sub-categories.
ARCHIVES_WITH_SUBCATS = {
    "astro-ph", "cond-mat", "cs", "econ", "eess", "math", "nlin", "physics",
    "q-bio", "q-fin", "stat",
}


def canonical(cat: str) -> str:
    cat = LEGACY.get(cat, cat)
    return ALIASES.get(cat, cat)


def archive_of(cat: str) -> str:
    return cat.split(".")[0]


def group_of(cat: str) -> str:
    arch = archive_of(cat)
    return "Physics" if arch in PHYSICS_ARCHIVES else GROUPS.get(arch, "Other")


def is_ambiguous_fine(cat: str) -> bool:
    """True for a bare archive name used as a category when that archive has sub-categories."""
    return "." not in cat and cat in ARCHIVES_WITH_SUBCATS
