"""
variance_decomposition.py — ANOVA + Bootstrap for 6-switch gear fatigue sweep
==============================================================================
Input: output/sweep.json
Output: output/anova.json
"""

import json, os
import numpy as np
from scipy.stats import f as f_dist

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)

with open("output/sweep.json") as f:
    sweep = json.load(f)

valid = [r for r in sweep if r["log10_Nf"] is not None]
y = np.array([r["log10_Nf"] for r in valid])
n_valid = len(y)
print(f"Valid entries: {n_valid} / {len(sweep)}")

factors = [
    "sn_source", "mean_stress_method",
    "damage_method", "size_surface_standard", "rz_level"
]
n_levels = [2, 4, 2, 3, 3]  # 5 factors (Kf switch removed in v2.3)

# Interaction terms to decompose (only the largest expected interaction)
interactions = [
    ("size_surface_standard", "rz_level"),  # Std × Rz interaction
]

# Map categorical to integer
factor_values = {}
for f in factors:
    cats = sorted(set(r[f] for r in valid))
    factor_values[f] = {c: i for i, c in enumerate(cats)}

X = np.zeros((n_valid, len(factors)), dtype=int)
for i, r in enumerate(valid):
    for j, f in enumerate(factors):
        X[i, j] = factor_values[f][r[f]]

grand_mean = y.mean()
SS_total = np.sum((y - grand_mean) ** 2)

# Type-II SS
SS_factors = {}
for j, f in enumerate(factors):
    group_means = {}
    for level in range(n_levels[j]):
        mask = X[:, j] == level
        if mask.sum() > 0:
            group_means[level] = y[mask].mean()
    ss = 0.0
    for level, mean_k in group_means.items():
        n_k = (X[:, j] == level).sum()
        ss += n_k * (mean_k - grand_mean) ** 2
    SS_factors[f] = float(ss)

SS_residual = SS_total - sum(SS_factors.values())
df_total = n_valid - 1
df_residual = df_total - sum(n_levels) + len(factors)

# ── Interaction: size_surface_standard × rz_level ──
# Compute cell means and subtract main effects
SS_interactions = {}
for (f1, f2) in interactions:
    j1, j2 = factors.index(f1), factors.index(f2)
    cell_means = {}
    for l1 in range(n_levels[j1]):
        for l2 in range(n_levels[j2]):
            mask = (X[:, j1] == l1) & (X[:, j2] == l2)
            if mask.sum() > 0:
                cell_means[(l1, l2)] = (y[mask].mean(), mask.sum())

    # Cell SS: variance explained by cell means
    SS_cell = sum(n * (mean_val - grand_mean) ** 2
                  for (mean_val, n) in cell_means.values())

    # Interaction SS = Cell SS - Main effect SS of f1 - Main effect SS of f2
    SS_inter = SS_cell - SS_factors[f1] - SS_factors[f2]
    # Clamp to >= 0 (numerical precision)
    SS_inter = max(SS_inter, 0.0)
    SS_interactions[f"{f1} × {f2}"] = float(SS_inter)
    SS_residual -= SS_inter  # remove interaction from residual
    df_inter = (n_levels[j1] - 1) * (n_levels[j2] - 1)
    df_residual -= df_inter

MS_residual = SS_residual / df_residual if df_residual > 0 else 1.0

anova_table = []
for j, f in enumerate(factors):
    df_f = n_levels[j] - 1
    MS_f = SS_factors[f] / df_f
    F = MS_f / MS_residual if MS_residual > 0 else np.inf
    p = 1.0 - f_dist.cdf(F, df_f, df_residual)
    variance_frac = 100.0 * SS_factors[f] / SS_total
    anova_table.append({
        "factor": f, "SS": float(SS_factors[f]), "df": df_f,
        "MS": float(MS_f), "F": float(F), "p": float(p),
        "variance_frac_%": round(variance_frac, 2),
    })

# Add interaction terms to ANOVA table
for int_name, SS_inter in SS_interactions.items():
    f1, f2 = interactions[0]  # use first (only) interaction pair
    df_inter = (n_levels[factors.index(f1)] - 1) * (n_levels[factors.index(f2)] - 1)
    MS_inter = SS_inter / df_inter
    F_inter = MS_inter / MS_residual if MS_residual > 0 else np.inf
    p_inter = 1.0 - f_dist.cdf(F_inter, df_inter, df_residual)
    var_frac_inter = 100.0 * SS_inter / SS_total
    anova_table.append({
        "factor": int_name, "SS": float(SS_inter), "df": df_inter,
        "MS": float(MS_inter), "F": float(F_inter), "p": float(p_inter),
        "variance_frac_%": round(var_frac_inter, 2),
        "type": "interaction",
    })

print(f"\n{'Factor':<28} {'SS':>8} {'df':>3} {'F':>10} {'p':>8} {'Var%':>7}")
print("-" * 72)
for row in anova_table:
    print(f"{row['factor']:<28} {row['SS']:>8.1f} {row['df']:>3} "
          f"{row['F']:>10.1f} {row['p']:>8.4f} {row['variance_frac_%']:>6.1f}%")
print("-" * 72)
print(f"{'Residual':<28} {SS_residual:>8.1f} {df_residual:>3}")
print(f"{'Total':<28} {SS_total:>8.1f} {df_total:>3}")

# Bootstrap
N_BOOTSTRAP, RNG_SEED = 2000, 42
rng = np.random.RandomState(RNG_SEED)
print(f"\n=== Bootstrap (N={N_BOOTSTRAP}, seed={RNG_SEED}) ===")

boot_fracs = {f: [] for f in factors}
for int_name in SS_interactions:
    boot_fracs[int_name] = []
for b in range(N_BOOTSTRAP):
    idx = rng.choice(n_valid, size=n_valid, replace=True)
    yb, Xb = y[idx], X[idx]
    gmb = yb.mean(); SSt_b = np.sum((yb - gmb)**2)
    for j, f in enumerate(factors):
        gm = {}
        for lv in range(n_levels[j]):
            mk = Xb[:, j] == lv
            if mk.sum(): gm[lv] = yb[mk].mean()
        ssb = sum((Xb[:,j]==lv).sum() * (gm[lv]-gmb)**2 for lv in gm)
        boot_fracs[f].append(100.0 * ssb / SSt_b)

    # Bootstrap interactions
    for (f1, f2) in interactions:
        j1, j2 = factors.index(f1), factors.index(f2)
        # Cell SS
        cell_ss = 0.0
        for l1 in range(n_levels[j1]):
            for l2 in range(n_levels[j2]):
                mk = (Xb[:, j1] == l1) & (Xb[:, j2] == l2)
                if mk.sum() > 0:
                    cell_ss += mk.sum() * (yb[mk].mean() - gmb) ** 2
        # Subtract main effects
        ss_int_b = max(cell_ss - sum(boot_fracs[f][-1] * SSt_b / 100.0 for f in [f1, f2]), 0)
        int_name = f"{f1} × {f2}"
        boot_fracs[int_name].append(100.0 * ss_int_b / SSt_b)

for k in list(boot_fracs.keys()):
    v = np.array(boot_fracs[k])
    label = k if k in factors else k
    print(f"  {k:<28} {v.mean():>6.1f}% ± {v.std():.1f}%  "
          f"CI95=[{np.percentile(v,2.5):.1f}, {np.percentile(v,97.5):.1f}]  "
          f"S/N={v.mean()/max(v.std(),1e-10):.0f}")

# Save
noise_floor = {}
for f in list(boot_fracs.keys()):
    v = np.array(boot_fracs[f])
    noise_floor[f] = {
        "mean": float(v.mean()), "std": float(v.std()),
        "ci_95_lower": float(np.percentile(v, 2.5)),
        "ci_95_upper": float(np.percentile(v, 97.5)),
    }

anova_output = {
    "n_valid": n_valid, "n_total": len(sweep),
    "grand_mean_log10_Nf": float(grand_mean),
    "SS_total": float(SS_total), "SS_residual": float(SS_residual),
    "anova_table": anova_table,
    "interactions": list(SS_interactions.keys()),
    "bootstrap": {"n_bootstrap": N_BOOTSTRAP, "rng_seed": RNG_SEED, "noise_floor": noise_floor},
    "factor_levels": {f: list(factor_values[f].keys()) for f in factors},
}
with open("output/anova.json", "w") as f:
    json.dump(anova_output, f, indent=2)
print("\nSaved output/anova.json")
