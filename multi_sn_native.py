# -*- coding: utf-8 -*-
"""
multi_sn_native.py — 5-level S-N sweep under the NATIVE chain (v6.1)
====================================================================
Reviewer round-4 (opinion 04) + audit: the existing 360-combination
multi-SN analysis (multi_sn_sensitivity.py) was run with the OLD
Peterson stress chain (8/5), while the v6.0 primary result uses the
NATIVE chain (AGMA J + FKM official K_f + ISO Y_Sa). Publishing both
side by side is a pipeline-inconsistency error: the 34.7%/33.6%
numbers are not comparable with the 68.5% primary table.

This script re-runs the 5-level S-N sweep (360 combinations) under the
SAME native chain as the primary result:

  - FKM entries: life scaled by official K_f (Siebel & Stieler, Table
    2.3.5 notch gradient, phi=0.44)
  - AGMA entries: life scaled by J-midpoint ratio
  - ISO entries: native Y_Sa (no change)

The native-chain result answers: does the material span (5 heat-
treatment states) compete with the standard choice under the primary
pipeline? 2-curve and 5-curve decompositions are both reported.

All numbers in output/multi_sn_native.json are produced here.
"""
import json, sys, os, itertools
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)

import gear_params
from fatigue_models import compute_fatigue_life
from gear_params import RZ_LEVELS, MATERIAL, GEAR
from aft_variance_decomposition import (
    build_design, extract_arrays, em_censored_normal,
    FACTORS, FACTOR_ORDER,
)

# ---------------------------------------------------------------
# 1. Same 5-level S-N set as multi_sn_sensitivity.py
# ---------------------------------------------------------------
def make_curve(label, uts, sigma_w_1e6, b):
    sigma_f_prime = sigma_w_1e6 * (2e6) ** (-b)
    return {
        "sigma_f_prime": round(sigma_f_prime, 1),
        "b": b,
        "fatigue_strength_at_1e6": round(sigma_w_1e6, 1),
        "uts": uts,
        "source": ("Synthetic 42CrMo4 QT heat-treatment state anchored to "
                   "Ulrich et al. 2025 tempering series + FKM sigma_W=0.45*UTS; "
                   + label),
    }

SYNTHETIC = {
    "Synthetic_42CrMo4_HighStrength": make_curve(
        "low-tempered ~380 C state", uts=1660.0, sigma_w_1e6=0.45*1660.0, b=-0.060),
    "Synthetic_42CrMo4_Classic500": make_curve(
        "tempered ~500 C state", uts=1322.0, sigma_w_1e6=0.45*1322.0, b=-0.060),
    "Synthetic_42CrMo4_Classic560": make_curve(
        "tempered ~560 C state", uts=1151.0, sigma_w_1e6=0.45*1151.0, b=-0.055),
}
for k, v in SYNTHETIC.items():
    gear_params.SN_CURVES[k] = v

SN_SOURCES_5 = [
    "Waterloo_SMDIdbase_Iter068", "Waterloo_SMDIdbase_Iter066",
    "Synthetic_42CrMo4_HighStrength",
    "Synthetic_42CrMo4_Classic500", "Synthetic_42CrMo4_Classic560",
]
SN_SOURCES_2 = ["Waterloo_SMDIdbase_Iter068", "Waterloo_SMDIdbase_Iter066"]
MEAN_STRESS_METHODS = ["Goodman", "Gerber", "Morrow", "SWT"]
DAMAGE_METHODS = ["Miner_original", "Miner_modified"]
SIZE_SURFACE_STANDARDS = ["ISO_6336", "AGMA_2001", "FKM"]
RZ_KEYS = list(RZ_LEVELS.keys())

# ---------------------------------------------------------------
# 2. Native-chain stress multipliers (same as multi_geometry_native.py)
# ---------------------------------------------------------------
def fkm_kf_official(mn):
    aG, bG = 0.90, 1200.0
    Rm = MATERIAL["uts"]
    r_root = GEAR["tip_radius"] * mn
    B = np.pi * mn / 2.0
    hf = 1.25 * mn
    b = B - hf
    tr = hf / r_root
    Bb = B / b
    table = [(1.0, 1.3, 0.08), (1.0, 4.9, 0.19), (3.3, 1.3, 0.21), (3.3, 4.9, 0.44)]
    tr = min(max(tr, 1.0), 3.3); Bb = min(max(Bb, 1.3), 4.9)
    def lerp1d(x, x0, x1, v0, v1):
        if x1 == x0: return v0
        return v0 + (x - x0) / (x1 - x0) * (v1 - v0)
    v00, v01, v10, v11 = 0.08, 0.19, 0.21, 0.44
    phi = lerp1d(tr, 1.0, 3.3, lerp1d(Bb, 1.3, 4.9, v00, v01), lerp1d(Bb, 1.3, 4.9, v10, v11))
    K_t = 1.0 + 0.5 * np.sqrt(B / max(r_root, 0.1)) * (B / (2.25 * mn)) ** 0.3
    G = 2.3 / max(r_root, 1e-3) * (1.0 + phi)
    exp_term = 10.0 ** (-(aG + Rm / bG))
    ns = 1.0 + G ** 0.25 * exp_term
    return float(K_t / ns)

K_FKM_OFFICIAL = fkm_kf_official(GEAR["module_mn"])
RATIO_FKM = K_FKM_OFFICIAL / 1.55      # vs unified Y_Sa
RATIO_AGMA = 1.57 / 1.55               # AGMA J midpoint (same as v6.0 native chain)

# ---------------------------------------------------------------
# 3. Native-chain sweep
# ---------------------------------------------------------------
def run_sweep(sn_sources):
    rows = []
    for sn, ms, dm, ss, rz_key in itertools.product(
            sn_sources, MEAN_STRESS_METHODS, DAMAGE_METHODS,
            SIZE_SURFACE_STANDARDS, RZ_KEYS):
        rz = RZ_LEVELS[rz_key]
        Nf, _ = compute_fatigue_life(
            sn_source=sn, kf_method="Peterson", mean_stress_method=ms,
            damage_method=dm, size_surface_standard=ss,
            rz_um=rz["Rz_um"], verbose=False)
        # native-chain stress multiplier applied AFTER standard's own factors
        if ss == "FKM":
            if np.isfinite(Nf):
                Nf = Nf / RATIO_FKM
        elif ss == "AGMA_2001":
            if np.isfinite(Nf):
                Nf = Nf / RATIO_AGMA
        rows.append({
            "sn_source": sn, "mean_stress_method": ms,
            "damage_method": dm, "size_surface_standard": ss,
            "rz_level": rz_key,
            "log10_Nf": float(np.log10(Nf)) if np.isfinite(Nf) else None,
        })
    return rows


def make_entries(rows):
    entries = []
    for r in rows:
        e = {k: r[k] for k in ("sn_source", "mean_stress_method",
                               "damage_method", "size_surface_standard",
                               "rz_level")}
        if r["log10_Nf"] is not None:
            e["logNf"] = float(r["log10_Nf"]); e["censored"] = 0
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


def decompose(entries):
    X_full = build_design(entries, FACTOR_ORDER)
    Y, CENS, Y_LOW = extract_arrays(entries)
    beta_f, sig_f, ll_f = em_censored_normal(X_full, Y, CENS, Y_LOW)
    if not np.isfinite(ll_f):
        return None, None
    deltas = {}
    for f in FACTOR_ORDER:
        Xd = build_design(entries, [g for g in FACTOR_ORDER if g != f])
        beta, sig, ll = em_censored_normal(Xd, Y, CENS, Y_LOW, sigma_fixed=sig_f)
        deltas[f] = ll_f - ll
    # std x rz interaction (same term as the paper)
    ls = FACTORS["size_surface_standard"]; lr = FACTORS["rz_level"]
    xic = []
    for si in range(1, len(ls)):
        for rj in range(1, len(lr)):
            col = np.array([1.0 if e["size_surface_standard"] == ls[si] and e["rz_level"] == lr[rj]
                            else 0.0 for e in entries])
            xic.append(col)
    Xi = np.column_stack(xic) if xic else np.zeros((len(entries), 0))
    Xi = Xi - Xi.mean(axis=0)
    _, _, lli = em_censored_normal(np.column_stack([X_full, Xi]), Y, CENS, Y_LOW, sigma_fixed=sig_f)
    deltas["size_surface_standard x rz_level"] = lli - ll_f
    total = sum(max(0.0, d) for d in deltas.values())
    if total <= 0 or not np.isfinite(total):
        return None, None
    fracs = {k: 100.0 * max(0.0, d) / total for k, d in deltas.items()}
    return fracs, deltas


# ---------------------------------------------------------------
# 4. Run
# ---------------------------------------------------------------
print("Native-chain sweep: 2 curves (144)...", flush=True)
rows2 = run_sweep(SN_SOURCES_2)
e2 = make_entries(rows2)
f2, d2 = decompose(e2)

print("Native-chain sweep: 5 curves (360)...", flush=True)
rows5 = run_sweep(SN_SOURCES_5)
e5 = make_entries(rows5)
f5, d5 = decompose(e5)

# censoring stats
def cens_stats(entries):
    return sum(1 for e in entries if e["censored"] == 1), len(entries)
c2, n2 = cens_stats(e2)
c5, n5 = cens_stats(e5)

out = {
    "description": "5-level S-N sweep under the NATIVE chain (AGMA J midpoint + FKM official K_f + "
                   "ISO Y_Sa), the same pipeline as the v6.0 primary result; 2-curve (144) and "
                   "5-curve (360) decompositions.",
    "pipeline": "native (FKM K_f=%.4f official, ratio %.4f; AGMA J ratio %.4f)" % (
        K_FKM_OFFICIAL, RATIO_FKM, RATIO_AGMA),
    "note_consistency": "Replaces multi_sn_sensitivity.json (Peterson pipeline, 8/5) for all "
                        "comparisons against the native-chain primary result.",
    "synthetic_curves": {k: {"uts": v["uts"], "sigma_w_1e6": v["fatigue_strength_at_1e6"], "b": v["b"]}
                         for k, v in SYNTHETIC.items()},
    "native_2curves_144": {
        "shares_%": {k: round(v, 2) for k, v in f2.items()},
        "n_censored": c2, "n_total": n2,
    },
    "native_5curves_360": {
        "shares_%": {k: round(v, 2) for k, v in f5.items()},
        "n_censored": c5, "n_total": n5,
    },
    "comparison": {
        "standard_2v_5": [round(f2["size_surface_standard"], 1), round(f5["size_surface_standard"], 1)],
        "sn_2v_5": [round(f2["sn_source"], 1), round(f5["sn_source"], 1)],
        "ranking_2curves": sorted(f2, key=f2.get, reverse=True)[:3],
        "ranking_5curves": sorted(f5, key=f5.get, reverse=True)[:3],
    },
    "summary": (
        "Native chain: 2 curves -> std %.1f%%, S-N %.1f%%; 5 curves -> std %.1f%%, S-N %.1f%%. "
        "Under the native chain the material span (5 heat treatments) %s."
        % (f2["size_surface_standard"], f2["sn_source"],
           f5["size_surface_standard"], f5["sn_source"],
           "overtakes the standard choice" if f5["sn_source"] > f5["size_surface_standard"]
           else "approaches but does not overtake the standard choice")
    ),
}
with open("output/multi_sn_native.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print("\n=== NATIVE CHAIN: 2 curves (144) ===")
for k, v in f2.items():
    print("  %-40s %6.2f%%" % (k, v))
print("=== NATIVE CHAIN: 5 curves (360) ===")
for k, v in f5.items():
    print("  %-40s %6.2f%%" % (k, v))
print("\nSaved output/multi_sn_native.json")
print("=== Complete ===")
