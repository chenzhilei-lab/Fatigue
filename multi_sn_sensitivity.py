"""
multi_sn_sensitivity.py — Multi S-N curve sensitivity (B5 / 0805意见02 二.3)
============================================================================
The full-factorial design uses only two near-identical S-N curves from the
same heat-treatment condition, which structurally caps the S-N source
share at 2.5%. This script tests how that share changes when the design
is extended to five curves spanning realistic quenched-and-tempered
42CrMo4 heat treatments:

  Waterloo #68/#66  (existing; UTS 1356/1537 MPa)
  + Synthetic 42CrMo4 heat-treatment states anchored to the independent
    tempering series of Ulrich et al. (Forsch. Ingenieurwes. 89:59, 2025)
    with fatigue strength at 10^6 taken as sigma_W = 0.45*UTS (FKM rule):
      - High-strength low-tempered  (~380 C,  UTS 1660 MPa, sigma_W 747 MPa)
      - Classic tempered ~500 C    (UTS 1322 MPa, sigma_W 595 MPa)
      - Classic tempered ~560 C    (UTS 1151 MPa, sigma_W 518 MPa)

The sweep is 5 x 4 x 2 x 3 x 3 = 360 combinations at 700 N*m, and the
same drop-one-factor AFT decomposition is applied (with the S-N factor
now spanning 5 levels). The two-curve baseline is reproduced as a check.

Output: output/multi_sn_sensitivity.json
        output/fig8_multi_sn_sensitivity.png
"""

import json
import os
import itertools
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplcache"))

import gear_params
from fatigue_models import compute_fatigue_life
from gear_params import RZ_LEVELS
from aft_variance_decomposition import (
    build_design, extract_arrays, em_censored_normal,
    FACTORS, FACTOR_ORDER,
)

# ---------------------------------------------------------------
# 1. Synthetic S-N curves (runtime patch; gear_params.py untouched)
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
    max_obs = max(e["logNf"] for e in entries if e["logNf"] is not None)
    for e in entries:
        if e["censored"] == 1:
            e["logNf_lower"] = max_obs
    return entries


def decompose(entries, sn_levels):
    FACTORS["sn_source"] = sn_levels
    X_full = build_design(entries, FACTOR_ORDER)
    y, cens, y_low = extract_arrays(entries)
    _, sigma_full, ll_full = em_censored_normal(X_full, y, cens, y_low)
    drop = {}
    for fname in FACTOR_ORDER:
        X_drop = build_design(entries, [f for f in FACTOR_ORDER if f != fname])
        _, _, ll = em_censored_normal(X_drop, y, cens, y_low,
                                      sigma_fixed=sigma_full)
        drop[fname] = ll_full - ll
    levels_std = FACTORS["size_surface_standard"]
    levels_rz = FACTORS["rz_level"]
    int_cols = []
    for si in range(1, len(levels_std)):
        for rj in range(1, len(levels_rz)):
            col = np.array([
                1.0 if e["size_surface_standard"] == levels_std[si]
                     and e["rz_level"] == levels_rz[rj] else 0.0
                for e in entries])
            int_cols.append(col)
    X_int = np.column_stack(int_cols) if int_cols else np.zeros((len(entries), 0))
    X_int = X_int - X_int.mean(axis=0)
    X_w_int = np.column_stack([X_full, X_int])
    _, _, ll_int = em_censored_normal(X_w_int, y, cens, y_low,
                                      sigma_fixed=sigma_full)
    drop["size_surface_standard x rz_level"] = ll_int - ll_full
    total = sum(max(0, d) for d in drop.values())
    shares = {k: 100.0 * max(0, v) / total for k, v in drop.items()}
    n_cens = int(np.sum(cens))
    print(f"  n_censored={n_cens}/{len(entries)} ({100*n_cens/len(entries):.0f}%)  "
          f"sigma={sigma_full:.4f}  ll={ll_full:.2f}")
    for k, v in shares.items():
        print(f"    {k:32s} {v:6.1f}%  (dLL={drop[k]:+8.2f})")
    return {
        "shares_%": {k: round(v, 2) for k, v in shares.items()},
        "delta_logLik": {k: round(v, 3) for k, v in drop.items()},
        "sigma_aft": round(float(sigma_full), 4),
        "logLik_full": round(float(ll_full), 2),
        "n_censored": n_cens, "n_total": len(entries),
    }


# ---------------------------------------------------------------
# 2. Run both designs
# ---------------------------------------------------------------
print("Sweep: 2-curve baseline (144)")
rows2 = run_sweep(SN_SOURCES_2)
print("Sweep: 5-curve design (360)")
rows5 = run_sweep(SN_SOURCES_5)

print("\n=== Baseline reproduction: 2 S-N curves ===")
dec2 = decompose(make_entries(rows2), SN_SOURCES_2)
print("\n=== Extended design: 5 S-N curves ===")
dec5 = decompose(make_entries(rows5), SN_SOURCES_5)

curves_info = {k: {"uts": gear_params.SN_CURVES[k]["uts"],
                   "sigma_w_1e6": gear_params.SN_CURVES[k]["fatigue_strength_at_1e6"],
                   "b": gear_params.SN_CURVES[k]["b"],
                   "sigma_f_prime": gear_params.SN_CURVES[k]["sigma_f_prime"],
                   "synthetic": k.startswith("Synthetic")}
               for k in SN_SOURCES_5}

summary = {
    "design": "5 x 4 x 2 x 3 x 3 = 360 combinations at 700 N*m; "
              "S-N factor spans 5 levels (2 Waterloo + 3 synthetic heat treatments)",
    "synthetic_curves": curves_info,
    "baseline_2_curves": {
        "sn_source_share_%": dec2["shares_%"]["sn_source"],
        "n_censored": dec2["n_censored"], "n_total": dec2["n_total"],
        "sigma_aft": dec2["sigma_aft"],
    },
    "extended_5_curves": {
        "sn_source_share_%": dec5["shares_%"]["sn_source"],
        "n_censored": dec5["n_censored"], "n_total": dec5["n_total"],
        "sigma_aft": dec5["sigma_aft"],
        "shares_%": dec5["shares_%"],
        "delta_logLik": dec5["delta_logLik"],
    },
    "change": {
        "sn_share_pp": round(dec5["shares_%"]["sn_source"]
                             - dec2["shares_%"]["sn_source"], 1),
        "direction": ("higher" if dec5["shares_%"]["sn_source"]
                      > dec2["shares_%"]["sn_source"] else "lower"),
    },
}
with open("output/multi_sn_sensitivity.json", "w") as f:
    json.dump({"summary": summary,
               "baseline_2_curves_decomposition": dec2,
               "extended_5_curves_decomposition": dec5}, f, indent=2)

print("\n=== Summary ===")
print(f"S-N share: 2 curves -> {dec2['shares_%']['sn_source']:.1f}% | "
      f"5 curves -> {dec5['shares_%']['sn_source']:.1f}% "
      f"({summary['change']['sn_share_pp']:+.1f} pp)")
print(f"Censoring: {dec2['n_censored']}/144 -> {dec5['n_censored']}/360 "
      f"({100*dec5['n_censored']/360:.0f}%)")

# ---------------------------------------------------------------
# 3. Figure
# ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

# Panel a: S-N share
ax = axes[0]
shares = [dec2["shares_%"]["sn_source"], dec5["shares_%"]["sn_source"]]
b = ax.bar(["2 S-N curves\n(Waterloo)", "5 S-N curves\n(+3 heat treatments)"],
           shares, color=["#2166ac", "#b2182b"], alpha=0.85, width=0.55)
for bar, v in zip(b, shares):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.1f}%",
            ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("S--N curve source variance share (%)")
ax.set_title("S--N factor contribution vs.\nnumber of S--N curve levels")
ax.set_ylim(0, max(shares) * 1.35)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

# Panel b: fatigue strength at 10^6 for the five curves
ax = axes[1]
names = [k.replace("Waterloo_SMDIdbase_Iter", "W#").replace("Synthetic_42CrMo4_", "")
         for k in SN_SOURCES_5]
labels = ["#68", "#66", "HighStr\n(380 C)", "Classic\n(500 C)", "Classic\n(560 C)"]
vals = [curves_info[k]["sigma_w_1e6"] for k in SN_SOURCES_5]
colors = ["#2166ac", "#2166ac", "#b2182b", "#b2182b", "#b2182b"]
b = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.55)
for bar, v in zip(b, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 4, f"{v:.0f}",
            ha="center", fontsize=8.5)
ax.set_ylabel(r"Fatigue strength at $10^6$ cycles (MPa)")
ax.set_title("S--N curve levels in the extended design")
ax.set_ylim(0, max(vals) * 1.18)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

plt.tight_layout()
fig.savefig("output/fig8_multi_sn_sensitivity.png", dpi=300)
plt.close()
print("Saved output/fig8_multi_sn_sensitivity.png")
print("Saved output/multi_sn_sensitivity.json")
