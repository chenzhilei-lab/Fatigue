"""
full_interactions.py — complete second-order interaction design (v6.0, Layer 2)
===============================================================================
Reviewer round-3: "only the standard x Rz interaction is included; all other
second-order interactions (mean-stress x standard, mean-stress x roughness,
S-N x standard, ...) are discarded; main-effect shares may be biased."

This script re-runs the full 144-combination AFT decomposition with the
COMPLETE set of second-order interaction terms (all 10 pairwise
products of the 5 factors) added to the design matrix, and quantifies:
  - the total interaction variance share (vs. 1.2% for standard x Rz alone)
  - each pairwise interaction's individual share
  - the corrected main-effect shares

The benchmark (main effects + std x Rz only) is recomputed for comparison.

All numbers in output/full_interactions.json are produced here.
"""
import json, sys, os, itertools
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)

FACTORS = {
    "mean_stress_method": ["Goodman", "Gerber", "Morrow", "SWT"],
    "size_surface_standard": ["ISO_6336", "AGMA_2001", "FKM"],
    "rz_level": ["ground_Rz4", "machined_Rz15", "as_forged_Rz50"],
    "sn_source": ["Waterloo_SMDIdbase_Iter068", "Waterloo_SMDIdbase_Iter066"],
    "damage_method": ["Miner_original", "Miner_modified"],
}
F_ORDER = ["mean_stress_method", "size_surface_standard", "rz_level", "sn_source", "damage_method"]
PAIRS = list(itertools.combinations(F_ORDER, 2))


def entries_from_sweep(sweep_data):
    entries = []
    for r in sweep_data:
        e = {k: r[k] for k in F_ORDER}
        nf = r.get("Nf", float("inf"))
        log_nf = r.get("log10_Nf")
        if isinstance(nf, (int, float)) and np.isfinite(nf) and nf > 0 and log_nf is not None:
            e["logNf"] = float(log_nf); e["censored"] = 0
        else:
            e["logNf"] = None; e["censored"] = 1
        entries.append(e)
    fv = [e["logNf"] for e in entries if e["logNf"] is not None]
    if fv:
        mo = max(fv)
        for e in entries:
            if e["censored"] == 1:
                e["logNf_lower"] = mo
    return entries


def build_main_design(entries, fs=None):
    if fs is None:
        fs = F_ORDER
    cols = []
    for fn in fs:
        lv = FACTORS[fn]
        for j in range(1, len(lv)):
            col = np.array([1.0 if e[fn] == lv[j] else 0.0 for e in entries])
            cols.append(col - 1.0 / len(lv))
    X = np.column_stack(cols) if cols else np.zeros((len(entries), 0))
    return np.column_stack([np.ones(len(entries)), X])


def build_pair_columns(entries, fa, fb):
    """Dummy-coded centred interaction columns for factor pair (fa, fb)."""
    la, lb = FACTORS[fa], FACTORS[fb]
    cols = []
    for i in range(1, len(la)):
        for j in range(1, len(lb)):
            col = np.array([1.0 if e[fa] == la[i] and e[fb] == lb[j] else 0.0 for e in entries])
            cols.append(col)
    Xi = np.column_stack(cols) if cols else np.zeros((len(entries), 0))
    return Xi - Xi.mean(axis=0)


def extract_arrays(entries):
    y = np.array([e.get("logNf", 0) or 0 for e in entries])
    c = np.array([e["censored"] == 1 for e in entries])
    yl = np.array([e.get("logNf_lower", 0) or 0 for e in entries])
    return y, c, yl


from scipy.stats import norm


def em(X, yo, cen, yl, n_iter=30, sigma_fixed=None):
    n, p = X.shape
    om = ~cen
    if om.sum() < p + 2:
        return None, None, float('-inf')
    beta = np.linalg.lstsq(X[om], yo[om], rcond=None)[0]
    sigma = (sigma_fixed if sigma_fixed is not None
             else max(np.std(yo[om] - X[om] @ beta, ddof=p), 0.001))
    ya = yo.copy()
    for _ in range(n_iter):
        mu = X @ beta
        bo, so = beta.copy(), sigma
        if cen.any():
            z = (yl[cen] - mu[cen]) / sigma
            imr = np.where(z > 30, z, np.where(z < -30, 0.0, np.exp(norm.logpdf(z) - norm.logsf(z))))
            ya[cen] = mu[cen] + sigma * imr
        beta = np.linalg.lstsq(X, ya, rcond=None)[0]
        r = ya - X @ beta
        if sigma_fixed is None:
            s2 = np.sum(r[om] ** 2) / max(om.sum(), 1)
            if cen.any():
                z = (yl[cen] - X[cen] @ beta) / so
                imr = np.where(z > 30, z, np.where(z < -30, 0.0, np.exp(norm.logpdf(z) - norm.logsf(z))))
                cv = so ** 2 * (1 + z * imr - imr ** 2)
                cv = np.maximum(cv, 0.0)
                s2 = (np.sum(r[om] ** 2) + np.sum(cv)) / n
            sigma = np.sqrt(max(s2, 0.001))
        sigma_conv = (sigma_fixed is not None) or abs(sigma - so) < 1e-8
        if np.max(np.abs(beta - bo)) < 1e-8 and sigma_conv:
            break
    mu = X @ beta
    ll = np.sum(norm.logpdf((yo[om] - mu[om]) / sigma)) - om.sum() * np.log(sigma)
    if cen.any():
        ll += np.sum(norm.logsf((yl[cen] - mu[cen]) / sigma))
    return beta, sigma, ll


def decompose_full(entries):
    """Full model with all 10 pairwise interactions; return per-term dLL."""
    Xm = build_main_design(entries)
    y, c, yl = extract_arrays(entries)
    bm, sm, llm = em(Xm, y, c, yl)
    # full model = main + all pairs
    Xf = Xm
    for fa, fb in PAIRS:
        Xf = np.column_stack([Xf, build_pair_columns(entries, fa, fb)])
    bf, sf, llf = em(Xf, y, c, yl)
    terms = {}
    # main effects: drop from full model
    for fn in F_ORDER:
        keep_main = [g for g in F_ORDER if g != fn]
        Xd = build_main_design(entries, keep_main)
        for (fa, fb) in PAIRS:
            if fa == fn or fb == fn:
                continue
            Xd = np.column_stack([Xd, build_pair_columns(entries, fa, fb)])
        _, _, ll = em(Xd, y, c, yl, sigma_fixed=sf)
        terms[fn] = llf - ll
    # pairwise interactions: drop one pair from full model
    for (fa, fb) in PAIRS:
        Xd = Xm
        for (ga, gb) in PAIRS:
            if (ga, gb) == (fa, fb):
                continue
            Xd = np.column_stack([Xd, build_pair_columns(entries, ga, gb)])
        _, _, ll = em(Xd, y, c, yl, sigma_fixed=sf)
        terms[f"{fa}x{fb}"] = llf - ll
    total = sum(max(0.0, v) for v in terms.values())
    if total <= 0 or not np.isfinite(total):
        return None
    fracs = {k: round(100.0 * max(0.0, v) / total, 1) for k, v in terms.items()}
    return {"terms": terms, "fractions": fracs, "total_interaction_pct": round(sum(
        v for k, v in fracs.items() if "x" in k), 1)}


def decompose_benchmark(entries):
    """Main + std x Rz only (the paper's benchmark)."""
    Xm = build_main_design(entries)
    y, c, yl = extract_arrays(entries)
    bm, sm, llm = em(Xm, y, c, yl)
    Xb = np.column_stack([Xm, build_pair_columns(entries, "size_surface_standard", "rz_level")])
    bb, sb, llb = em(Xb, y, c, yl)
    terms = {}
    for fn in F_ORDER:
        keep = [g for g in F_ORDER if g != fn]
        Xd = build_main_design(entries, keep)
        Xd = np.column_stack([Xd, build_pair_columns(entries, "size_surface_standard", "rz_level")])
        _, _, ll = em(Xd, y, c, yl, sigma_fixed=sb)
        terms[fn] = llb - ll
    Xd = Xm  # drop the std x rz pair
    _, _, ll = em(Xd, y, c, yl, sigma_fixed=sb)
    terms["size_surface_standardxrz_level"] = llb - ll
    total = sum(max(0.0, v) for v in terms.values())
    if total <= 0:
        return None
    fracs = {k: round(100.0 * max(0.0, v) / total, 1) for k, v in terms.items()}
    return {"terms": terms, "fractions": fracs}


# ============================================================
# Run
# ============================================================
with open("output/sweep.json", encoding="utf-8") as f:
    sweep_orig = json.load(f)
entries = entries_from_sweep(sweep_orig)

print("Benchmark (main + std x Rz)...", flush=True)
bench = decompose_benchmark(entries)

print("Full second-order interaction model (all 10 pairs)...", flush=True)
full = decompose_full(entries)

out = {
    "description": "Complete second-order interaction decomposition: full 144-combination "
                   "AFT with all 10 pairwise interaction terms.",
    "benchmark_main_plus_stdxrz": bench,
    "full_second_order": full,
    "summary": (
        f"Benchmark: std x Rz interaction {bench['fractions'].get('size_surface_standardxrz_level', 0):.1f}%. "
        f"Full second-order model: total interaction share {full['total_interaction_pct']:.1f}% "
        f"(std x Rz {full['fractions'].get('size_surface_standardxrz_level', 0):.1f}%; "
        f"mean-stress x std {full['fractions'].get('mean_stress_methodxsize_surface_standard', 0):.1f}%; "
        f"mean-stress x Rz {full['fractions'].get('mean_stress_methodxrz_level', 0):.1f}%)."
    ),
}
with open("output/full_interactions.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print("\n=== BENCHMARK (main + std x Rz) ===")
for k, v in bench["fractions"].items():
    print(f"  {k:40s}: {v:5.1f}%")
print("\n=== FULL SECOND-ORDER ===")
for k, v in full["fractions"].items():
    print(f"  {k:40s}: {v:5.1f}%")
print(f"\nTotal interaction: {full['total_interaction_pct']:.1f}%")
print("\nSaved output/full_interactions.json")
print("=== Complete ===")
