"""
two_level_variable_amplitude.py — Two-level block-spectrum illustration
========================================================================
Supplementary result for the response to the "variable-amplitude loading
is only qualitative" comment (0805审稿人意见02, item 3).

Design:
  - Same 144-combination full-factorial chain as run_sweep.py.
  - Two-level block spectrum: n1 cycles at T1, n2 cycles at T2 per block.
  - Per-combination fatigue lives N1 (at T1) and N2 (at T2) computed with
    the identical compute_fatigue_life() pipeline (including the
    Dc=1.0/0.7 damage-rule threshold baked into each level's life).
  - Block life via linear Palmgren-Miner:  B = 1/(n1/N1 + n2/N2) blocks,
    total cycles = B*(n1+n2); run-out only if BOTH levels are run-out.
  - Drop-one-factor AFT decomposition (same machinery as
    aft_variance_decomposition.py) is run on (i) the constant-amplitude
    700 N*m data as an internal reproduction check and (ii) the two-level
    block data, so the damage-rule share can be compared directly.

Output: output/two_level_variable_amplitude.json
        output/fig7_two_level_comparison.png
"""

import json
import os
import itertools
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplcache"))

from fatigue_models import compute_fatigue_life
from gear_params import RZ_LEVELS, LOAD, GEAR

# Import the AFT machinery WITHOUT triggering the main analysis
from aft_variance_decomposition import (
    build_design, extract_arrays, em_censored_normal,
    FACTORS, FACTOR_ORDER,
)

# ---------------------------------------------------------------
# 1. Constant-amplitude baseline (reproduction check at 700 N*m)
# ---------------------------------------------------------------
SN_SOURCES = ["Waterloo_SMDIdbase_Iter068", "Waterloo_SMDIdbase_Iter066"]
MEAN_STRESS_METHODS = ["Goodman", "Gerber", "Morrow", "SWT"]
DAMAGE_METHODS = ["Miner_original", "Miner_modified"]
SIZE_SURFACE_STANDARDS = ["ISO_6336", "AGMA_2001", "FKM"]
RZ_KEYS = list(RZ_LEVELS.keys())

COMBOS = list(itertools.product(
    SN_SOURCES, MEAN_STRESS_METHODS, DAMAGE_METHODS,
    SIZE_SURFACE_STANDARDS, RZ_KEYS))


def set_torque(torque_Nm):
    """Override the global load so compute_fatigue_life uses torque T."""
    LOAD["torque_pinion"] = float(torque_Nm)
    LOAD["tangential_force"] = (2 * LOAD["torque_pinion"] * 1000) / GEAR["pitch_diameter_pinion"]
    LOAD["Ft_per_unit_facewidth"] = LOAD["tangential_force"] / GEAR["face_width"]


def life_at(sn, ms, dm, ss, rz_key, torque):
    """Nf at one torque level through the identical pipeline."""
    set_torque(torque)
    rz_info = RZ_LEVELS[rz_key]
    Nf, _ = compute_fatigue_life(
        sn_source=sn, kf_method="Peterson",
        mean_stress_method=ms, damage_method=dm,
        size_surface_standard=ss, rz_um=rz_info["Rz_um"], verbose=False)
    return Nf


def two_level_life(sn, ms, dm, ss, rz_key, T1, T2, n1, n2):
    """Block life under n1 cycles @T1 + n2 cycles @T2 repeated per block.

    N1/N2 already include the damage-rule threshold (Dc), so the Miner
    failure criterion is sum(n_i/N_i) = 1. Returns (N_block_total, N1, N2).
    """
    N1 = life_at(sn, ms, dm, ss, rz_key, T1)
    N2 = life_at(sn, ms, dm, ss, rz_key, T2)
    f1 = np.isfinite(N1)
    f2 = np.isfinite(N2)
    if not f1 and not f2:
        return np.inf, N1, N2
    d1 = n1 / N1 if f1 else 0.0
    d2 = n2 / N2 if f2 else 0.0
    damage_per_block = d1 + d2
    blocks = 1.0 / damage_per_block
    return blocks * (n1 + n2), N1, N2


def make_entries(rows):
    """Convert per-combo dicts into AFT entry list (same format as
    aft_variance_decomposition.py ENTRIES)."""
    entries = []
    for r in rows:
        e = {k: r[k] for k in ("sn_source", "mean_stress_method",
                               "damage_method", "size_surface_standard",
                               "rz_level")}
        log_nf = r.get("log10_N_block")
        if log_nf is not None and np.isfinite(log_nf):
            e["logNf"] = float(log_nf)
            e["censored"] = 0
        else:
            e["logNf"] = None
            e["censored"] = 1
        entries.append(e)
    max_obs = max(e["logNf"] for e in entries if e["logNf"] is not None)
    for e in entries:
        if e["censored"] == 1:
            e["logNf_lower"] = max_obs
    return entries


def decompose(entries, label):
    """Drop-one-factor decomposition (main factors + Std x Rz interaction),
    mirroring aft_variance_decomposition.py."""
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
    delta_int = ll_int - ll_full

    drop["size_surface_standard x rz_level"] = delta_int
    total = sum(max(0, d) for d in drop.values())
    shares = {k: 100.0 * max(0, v) / total for k, v in drop.items()}
    n_cens = int(np.sum(cens))
    print(f"[{label}] n_censored={n_cens}/{len(entries)}  "
          f"sigma={sigma_full:.4f}  ll_full={ll_full:.2f}")
    for k, v in shares.items():
        print(f"    {k:32s} {v:6.1f}%  (dLL={drop[k]:+8.2f})")
    return {"shares_%": {k: round(v, 2) for k, v in shares.items()},
            "delta_logLik": {k: round(v, 3) for k, v in drop.items()},
            "sigma_aft": round(float(sigma_full), 4),
            "logLik_full": round(float(ll_full), 2),
            "n_censored": n_cens, "n_total": len(entries)}


# ---------------------------------------------------------------
# 2. Compute lives
# ---------------------------------------------------------------
T1, T2 = 700.0, 800.0          # reference torque and overload level
N1_BLOCK, N2_BLOCK = 1e3, 1e3   # cycles per level per block

rows = []
for sn, ms, dm, ss, rz_key in COMBOS:
    N_block, N1, N2 = two_level_life(sn, ms, dm, ss, rz_key, T1, T2,
                                     N1_BLOCK, N2_BLOCK)
    rows.append({
        "sn_source": sn, "mean_stress_method": ms, "damage_method": dm,
        "size_surface_standard": ss, "rz_level": rz_key,
        "N1_700": float(N1) if np.isfinite(N1) else None,
        "N2_800": float(N2) if np.isfinite(N2) else None,
        "N_block_total": float(N_block) if np.isfinite(N_block) else None,
        "log10_N_block": float(np.log10(N_block)) if np.isfinite(N_block) else None,
    })

# ---------------------------------------------------------------
# 3. AFT decompositions: constant 700 (reproduction) vs two-level
# ---------------------------------------------------------------
with open("output/sweep.json") as f:
    sweep = json.load(f)
const_entries = []
for r in sweep:
    e = {k: r[k] for k in ("sn_source", "mean_stress_method",
                           "damage_method", "size_surface_standard",
                           "rz_level")}
    log_nf = r.get("log10_Nf")
    if log_nf is not None:
        e["logNf"] = float(log_nf); e["censored"] = 0
    else:
        e["logNf"] = None; e["censored"] = 1
    const_entries.append(e)
max_obs = max(e["logNf"] for e in const_entries if e["logNf"] is not None)
for e in const_entries:
    if e["censored"] == 1:
        e["logNf_lower"] = max_obs

two_entries = make_entries(rows)

print("\n=== Reproduction check: constant-amplitude 700 N*m ===")
const_dec = decompose(const_entries, "constant 700 N*m")
print("\n=== Two-level block spectrum 700/800 N*m ===")
two_dec = decompose(two_entries, "two-level 700/800 N*m")

# ---------------------------------------------------------------
# 4. Summary statistics
# ---------------------------------------------------------------
finite_block = [r["log10_N_block"] for r in rows if r["log10_N_block"] is not None]
finite_const = [r["log10_Nf"] for r in sweep if r["log10_Nf"] is not None]
finite_N1 = [r["log10_N_block"] for r in rows
             if r["log10_N_block"] is not None and r["N1_700"] is not None]

damage_pairs = {}
for (sn, ms, ss, rz) in itertools.product(SN_SOURCES, MEAN_STRESS_METHODS,
                                          SIZE_SURFACE_STANDARDS, RZ_KEYS):
    base = {"sn_source": sn, "mean_stress_method": ms,
            "size_surface_standard": ss, "rz_level": rz}
    m0 = next(r for r in rows if all(r[k] == v for k, v in base.items())
              and r["damage_method"] == "Miner_original")
    m1 = next(r for r in rows if all(r[k] == v for k, v in base.items())
              and r["damage_method"] == "Miner_modified")
    if m0["log10_N_block"] is not None and m1["log10_N_block"] is not None:
        damage_pairs.setdefault("two_level", []).append(
            abs(m0["log10_N_block"] - m1["log10_N_block"]))
    c0 = next(r for r in sweep if all(r[k] == v for k, v in base.items())
              and r["damage_method"] == "Miner_original")
    c1 = next(r for r in sweep if all(r[k] == v for k, v in base.items())
              and r["damage_method"] == "Miner_modified")
    if c0["log10_Nf"] is not None and c1["log10_Nf"] is not None:
        damage_pairs.setdefault("constant", []).append(
            abs(c0["log10_Nf"] - c1["log10_Nf"]))

summary = {
    "design": {
        "T1_Nm": T1, "T2_Nm": T2,
        "n1_cycles_per_block": N1_BLOCK, "n2_cycles_per_block": N2_BLOCK,
        "n_combinations": len(rows),
        "model": "linear Palmgren-Miner block accumulation; "
                 "Dc=1.0/0.7 threshold already embedded in N1,N2",
    },
    "constant_700_Nm": {
        "spread_dex": round(max(finite_const) - min(finite_const), 2),
        "n_finite": len(finite_const), "n_runout": 144 - len(finite_const),
        "decomposition": const_dec,
        "damage_pair_abs_log10_diff": {
            "median": round(float(np.median(damage_pairs["constant"])), 4),
            "min": round(float(np.min(damage_pairs["constant"])), 4),
            "max": round(float(np.max(damage_pairs["constant"])), 4),
        },
    },
    "two_level_700_800_Nm": {
        "spread_dex": round(max(finite_block) - min(finite_block), 2),
        "n_finite": len(finite_block),
        "n_runout": 144 - len(finite_block),
        "decomposition": two_dec,
        "damage_pair_abs_log10_diff": {
            "median": round(float(np.median(damage_pairs["two_level"])), 4),
            "min": round(float(np.min(damage_pairs["two_level"])), 4),
            "max": round(float(np.max(damage_pairs["two_level"])), 4),
        },
    },
}
with open("output/two_level_variable_amplitude.json", "w") as f:
    json.dump({"summary": summary,
               "constant_700_shares_%": const_dec["shares_%"],
               "two_level_shares_%": two_dec["shares_%"],
               "entries": rows}, f, indent=2)

print("\n=== Summary ===")
print(json.dumps({k: v for k, v in summary.items() if k != "design"},
                 indent=2)[:1200])

# ---------------------------------------------------------------
# 5. Figure
# ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

# Panel a: spread comparison (jittered strip + median)
ax = axes[0]
const_sorted = np.sort(finite_const)
two_sorted = np.sort(finite_block)
ax.plot(const_sorted, np.linspace(0, 1, len(const_sorted)),
        drawstyle="steps-post", color="#2166ac", lw=2,
        label=r"constant 700 N$\cdot$m")
ax.plot(two_sorted, np.linspace(0, 1, len(two_sorted)),
        drawstyle="steps-post", color="#b2182b", lw=2,
        label=r"two-level 700/800 N$\cdot$m")
# Annotate the method-induced life-spread difference
ax.annotate("", xy=(max(two_sorted), 0.35), xytext=(max(const_sorted), 0.35),
            arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.4))
ax.text((max(const_sorted) + max(two_sorted)) / 2, 0.42,
        "method spread:\n2.79 $\\rightarrow$ 4.25 dex",
        ha="center", fontsize=8.5, color="0.25")
ax.set_xlabel(r"$\log_{10}(N_f)$")
ax.set_ylabel("empirical CDF (finite-life combos)")
ax.set_title("Method-induced life spread")
ax.legend(fontsize=8.5, loc="lower right")
ax.grid(alpha=0.3, linestyle="--")

# Panel b: variance shares
ax = axes[1]
labels = ["Mean-stress", "Standard", "Roughness", "S–N", "Damage", "Std×Rz"]
const_share = [const_dec["shares_%"][k] for k in
               ("mean_stress_method", "size_surface_standard", "rz_level",
                "sn_source", "damage_method",
                "size_surface_standard x rz_level")]
two_share = [two_dec["shares_%"][k] for k in
             ("mean_stress_method", "size_surface_standard", "rz_level",
              "sn_source", "damage_method",
              "size_surface_standard x rz_level")]
x = np.arange(len(labels))
ax.bar(x - 0.2, const_share, 0.38, color="#2166ac", alpha=0.85, label="constant 700 N$\cdot$m")
ax.bar(x - 0.2, const_share, 0.38, color="#2166ac", alpha=0.85,
       label=r"constant 700 N$\cdot$m")
ax.bar(x + 0.2, two_share, 0.38, color="#b2182b", alpha=0.85,
       label=r"two-level 700/800 N$\cdot$m")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, fontsize=8.5)
ax.set_ylabel("Variance fraction (%)")
ax.set_title("AFT decomposition")
ax.legend(fontsize=8.5)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

plt.tight_layout()
fig.savefig("output/fig7_two_level_comparison.png", dpi=300)
plt.close()
print("\nSaved output/two_level_variable_amplitude.json")
print("Saved output/fig7_two_level_comparison.png")
