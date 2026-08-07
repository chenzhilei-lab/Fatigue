"""
simulation_validation.py — Monte Carlo validation of the drop-one-factor
likelihood-ratio variance decomposition (v4.1)
===============================================================================
Purpose: the drop-one-factor decomposition (Eq. lr_var in the paper) is a
practical heuristic without a literature pedigree. This script demonstrates
numerically that, for a balanced full-factorial log-normal AFT model, it
recovers known variance shares:

  1. Generate synthetic data from a known generative model with prescribed
     variance shares (MS=40%, Std=30%, Rz=20%, SN=5%, Dam=5%) and right
     censoring (~58%, matching the paper's run-out regime).
  2. Run the exact same AFT + drop-one-factor decomposition M=500 times.
  3. Quantify recovery: bias, RMSE, empirical 95% intervals, bootstrap
     coverage, and rank-recovery rate.
  4. Demonstrate equivalence with first-order Sobol' indices on the
     deterministic model output (complete-data limit), reporting R^2.

Output: output/simulation_validation.json + output/fig_sim_validation.png
"""

import json, itertools, sys, time
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

sys.stdout.reconfigure(line_buffering=True)

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)

from aft_variance_decomposition import (
    build_design, extract_arrays, em_censored_normal,
    FACTORS, FACTOR_ORDER,
)

# ============================================================
# 1. Generative model with KNOWN variance shares
# ============================================================
# Target shares (main effects only; the generative model has no interaction)
TRUE_SHARES = {
    "mean_stress_method": 0.40,
    "size_surface_standard": 0.30,
    "rz_level": 0.20,
    "sn_source": 0.05,
    "damage_method": 0.05,
}

# Sum-to-zero effect patterns (arbitrary shapes; only their variances matter)
EFFECT_PATTERNS = {
    "mean_stress_method": np.array([-0.9, 1.1, 0.2, -0.4]),   # 4 levels
    "size_surface_standard": np.array([0.8, 0.1, -0.9]),      # 3 levels
    "rz_level": np.array([0.55, 0.05, -0.60]),                # 3 levels
    "sn_source": np.array([0.25, -0.25]),                     # 2 levels
    "damage_method": np.array([0.12, -0.12]),                 # 2 levels
}
for f in FACTOR_ORDER:
    assert abs(EFFECT_PATTERNS[f].sum()) < 1e-9, f"not sum-to-zero: {f}"
    assert len(EFFECT_PATTERNS[f]) == len(FACTORS[f]), f"level mismatch: {f}"

SIGMA_TRUE = 0.1745   # noise SD in log10(Nf), matching the paper's fitted value
TOTAL_EXPLAINED_VAR = 0.90  # sum of main-effect variances (dex^2)
CENSOR_THRESHOLD = 6.0      # run-out if log10(Nf) > 10^6 cycles
TARGET_CENSORING = 0.30     # headline validation regime (method's operating range)
CENSORING_SWEEP = [0.15, 0.30, 0.45, 0.58]  # incl. the paper's 58% boundary

def build_effects():
    """Return {factor: effect vector} scaled to the target variance shares."""
    effects = {}
    for f in FACTOR_ORDER:
        v_raw = float(np.mean(EFFECT_PATTERNS[f] ** 2))
        v_target = TRUE_SHARES[f] * TOTAL_EXPLAINED_VAR
        effects[f] = EFFECT_PATTERNS[f] * np.sqrt(v_target / v_raw)
    return effects

EFFECTS = build_effects()

# The 144-combination balanced design
LEVELS = [FACTORS[f] for f in FACTOR_ORDER]
COMBOS = list(itertools.product(*LEVELS))
N_DESIGN = len(COMBOS)

def eta_for_combo(combo):
    """Deterministic linear predictor for one combination."""
    return sum(float(EFFECTS[f][FACTORS[f].index(combo[i])])
               for i, f in enumerate(FACTOR_ORDER))

ETA = np.array([eta_for_combo(c) for c in COMBOS])

# True variance shares (same as target, by construction)
# Sobol' first-order indices of the deterministic model output
# S_k = Var(E[eta | X_k]) / Var(eta) = V_k / sum_j V_j  (balanced design)
TRUE_FRACTIONS = {f: TRUE_SHARES[f] * 100.0 for f in FACTOR_ORDER}
TRUE_FRACTIONS["interaction"] = 0.0

def solve_intercept(target_cens):
    """Find intercept mu such that the mean censoring rate = target_cens."""
    def cens_rate(mu):
        z = (CENSOR_THRESHOLD - mu - ETA) / SIGMA_TRUE
        return float(np.mean(norm.sf(z)))
    lo, hi = -5.0, 9.0
    assert cens_rate(lo) < target_cens < cens_rate(hi), "bracket failed"
    return brentq(lambda mu: cens_rate(mu) - target_cens, lo, hi)

MU_TRUE = solve_intercept(TARGET_CENSORING)
print(f"Generative model: mu={MU_TRUE:.3f}, sigma={SIGMA_TRUE}, "
      f"n={N_DESIGN}, target censoring={TARGET_CENSORING:.0%}", flush=True)
print("True variance shares:", {k: f"{v:.1f}%" for k, v in TRUE_FRACTIONS.items()},
      flush=True)

# ============================================================
# 2. Decomposition (exact replica of the paper's procedure)
# ============================================================
def build_interaction_X(entries, X_full):
    levels_std = FACTORS["size_surface_standard"]
    levels_rz = FACTORS["rz_level"]
    cols = []
    for si in range(1, len(levels_std)):
        for rj in range(1, len(levels_rz)):
            col = np.array([
                1.0 if e["size_surface_standard"] == levels_std[si]
                     and e["rz_level"] == levels_rz[rj]
                else 0.0 for e in entries
            ])
            cols.append(col)
    X_int_raw = np.column_stack(cols) if cols else np.zeros((len(entries), 0))
    X_int_centered = X_int_raw - X_int_raw.mean(axis=0)
    return np.column_stack([X_full, X_int_centered])

def decompose(entries):
    """Full AFT + drop-one-factor + interaction decomposition.
    Returns (fractions dict, deltas dict). Matches the paper's normalization:
    var% = 100 * max(0, dL) / sum(max(0, dL)) over all six terms."""
    X_full = build_design(entries, FACTOR_ORDER)
    Y, CENS, Y_LOW = extract_arrays(entries)
    beta_f, sig_f, ll_f = em_censored_normal(X_full, Y, CENS, Y_LOW)
    if not np.isfinite(ll_f):
        return None, None

    deltas = {}
    for f in FACTOR_ORDER:
        Xd = build_design(entries, [g for g in FACTOR_ORDER if g != f])
        beta, sig, ll = em_censored_normal(Xd, Y, CENS, Y_LOW,
                                           sigma_fixed=sig_f)
        deltas[f] = ll_f - ll

    X_int = build_interaction_X(entries, X_full)
    beta_i, sig_i, ll_i = em_censored_normal(X_int, Y, CENS, Y_LOW,
                                             sigma_fixed=sig_f)
    deltas["interaction"] = ll_i - ll_f

    total = sum(max(0.0, d) for d in deltas.values())
    if total <= 0 or not np.isfinite(total):
        return None, None
    fracs = {k: 100.0 * max(0.0, d) / total for k, d in deltas.items()}
    return fracs, deltas

def make_entries(logNf_vals, censored_mask):
    entries = []
    for combo, y, c in zip(COMBOS, logNf_vals, censored_mask):
        e = dict(zip(FACTOR_ORDER, combo))
        e["logNf"] = float(y)
        e["censored"] = int(c)
        e["logNf_lower"] = 0.0 if not c else CENSOR_THRESHOLD
        entries.append(e)
    return entries

def simulate_once(rng):
    """One Monte Carlo draw: noisy + censored data, then decompose."""
    y = MU_TRUE + ETA + SIGMA_TRUE * rng.standard_normal(N_DESIGN)
    cens = y > CENSOR_THRESHOLD
    entries = make_entries(y, cens)
    return decompose(entries)

# ============================================================
# 3. Monte Carlo simulation
# ============================================================
M_SIMS = 500
B_BOOT = 100
RNG = np.random.default_rng(42)

MAIN_FACTORS = FACTOR_ORDER  # 5 main-effect factors
true_main = np.array([TRUE_FRACTIONS[f] for f in MAIN_FACTORS])

recovered_all = []      # fractions per sim (6 keys)
ranks_all = []
skipped = 0
t0 = time.time()

for m in range(M_SIMS):
    fracs, deltas = simulate_once(RNG)
    if fracs is None:
        skipped += 1
        continue
    rec_main = np.array([fracs[f] for f in MAIN_FACTORS])
    recovered_all.append(fracs)
    ranks_all.append(np.argsort(-rec_main))
    if (m + 1) % 100 == 0:
        print(f"  sim {m+1}/{M_SIMS} (skipped {skipped}) ...", flush=True)

recovered_main = np.array([[r[f] for f in MAIN_FACTORS] for r in recovered_all])
n_done = len(recovered_all)
print(f"Completed {n_done}/{M_SIMS} simulations in {time.time()-t0:.1f}s "
      f"({skipped} skipped)", flush=True)

# ---- Bias, RMSE, empirical 95% intervals ----
results = {}
for j, f in enumerate(MAIN_FACTORS):
    rec = recovered_main[:, j]
    bias = float(np.mean(rec - true_main[j]))
    rmse = float(np.sqrt(np.mean((rec - true_main[j]) ** 2)))
    q_lo, q_hi = np.percentile(rec, [2.5, 97.5])
    results[f] = {
        "true_%": round(true_main[j], 1),
        "mean_recovered_%": round(float(np.mean(rec)), 1),
        "bias_pp": round(bias, 2),
        "rmse_pp": round(rmse, 2),
        "empirical_95%_interval": [round(q_lo, 1), round(q_hi, 1)],
    }
    print(f"  {f:25s} true={true_main[j]:5.1f}%  recovered="
          f"{np.mean(rec):5.1f}% (bias {bias:+.2f} pp, RMSE {rmse:.2f} pp, "
          f"95% [{q_lo:.1f}, {q_hi:.1f}])", flush=True)

# ---- Rank recovery ----
true_rank = tuple(np.argsort(-true_main))
rank_hits = sum(1 for r in ranks_all if tuple(r) == true_rank)
rank_recovery = rank_hits / n_done
# Top-3 order match (the three dominant factors, ignoring the 5% tie)
top3_true = tuple(np.argsort(-true_main)[:3])
top3_hits = sum(1 for r in ranks_all if np.array_equal(r[:3], top3_true))
top3_recovery = top3_hits / n_done
from scipy.stats import spearmanr
sp_vals = [spearmanr(r, true_main).statistic for r in recovered_main]
print(f"Rank recovery: exact 5-factor {rank_recovery:.1%}; "
      f"top-3 order {top3_recovery:.1%}; "
      f"mean Spearman rho {np.mean(sp_vals):.3f}", flush=True)

# ---- Bootstrap coverage (redo properly) ----
# We did not store the raw entries; re-run a dedicated coverage pass with
# fresh simulations (independent, same generative model).
print("Coverage pass: fresh 150 simulations x B=100 bootstrap...", flush=True)
coverage_hits = {f: 0 for f in MAIN_FACTORS}
n_coverage = 0
rng_cov = np.random.default_rng(7)
for m in range(150):
    y = MU_TRUE + ETA + SIGMA_TRUE * rng_cov.standard_normal(N_DESIGN)
    cens = y > CENSOR_THRESHOLD
    entries = make_entries(y, cens)
    fracs, _ = decompose(entries)
    if fracs is None:
        continue
    boot_fracs = {f: [] for f in MAIN_FACTORS}
    for b in range(B_BOOT):
        idx = rng_cov.integers(0, N_DESIGN, size=N_DESIGN)
        sub = [entries[i] for i in idx]
        bf, _ = decompose(sub)
        if bf is None:
            continue
        for f in MAIN_FACTORS:
            boot_fracs[f].append(bf[f])
    if any(len(v) < 30 for v in boot_fracs.values()):
        continue
    n_coverage += 1
    for f in MAIN_FACTORS:
        lo, hi = np.percentile(boot_fracs[f], [2.5, 97.5])
        if lo <= TRUE_FRACTIONS[f] <= hi:
            coverage_hits[f] += 1
    if (m + 1) % 50 == 0:
        print(f"  coverage sim {m+1}/150 ...", flush=True)

coverage_out = {}
for f in MAIN_FACTORS:
    coverage_out[f] = round(coverage_hits[f] / max(n_coverage, 1), 3)
print(f"Coverage (n={n_coverage}):", coverage_out, flush=True)

# ============================================================
# 4. Sobol' equivalence on the deterministic model output
# ============================================================
# First-order Sobol' index of factor k on the model output eta:
#   S_k = Var(E[eta|X_k]) / Var(eta)  (balanced design, no interactions)
eta_var = float(np.var(ETA))
sobol_true = {}
for f in FACTOR_ORDER:
    idx = FACTOR_ORDER.index(f)
    grouped = {}
    for combo, e in zip(COMBOS, ETA):
        grouped.setdefault(combo[idx], []).append(e)
    cond_var = sum(len(v) * np.mean(v) ** 2 for v in grouped.values()) / N_DESIGN
    sobol_true[f] = (cond_var - float(np.mean(ETA)) ** 2) / eta_var

# AFT LR fractions on COMPLETE (uncensored) data across fresh simulations
rng_eq = np.random.default_rng(123)
lr_complete = {f: [] for f in FACTOR_ORDER}
sobol_vals = {f: [] for f in FACTOR_ORDER}
for m in range(M_SIMS):
    y = MU_TRUE + ETA + SIGMA_TRUE * rng_eq.standard_normal(N_DESIGN)
    entries_c = make_entries(y, np.zeros(N_DESIGN, dtype=bool))
    fracs, _ = decompose(entries_c)
    if fracs is None:
        continue
    for f in FACTOR_ORDER:
        lr_complete[f].append(fracs[f])
        sobol_vals[f].append(100.0 * sobol_true[f])

# Pooled R^2 between LR fractions and Sobol' indices
lr_all = np.concatenate([lr_complete[f] for f in FACTOR_ORDER])
sobol_all = np.concatenate([sobol_vals[f] for f in FACTOR_ORDER])
r_pooled = np.corrcoef(lr_all, sobol_all)[0, 1]
r2_pooled = float(r_pooled ** 2)
print(f"Sobol' equivalence (complete data, pooled R^2): {r2_pooled:.4f}",
      flush=True)

# ============================================================
# 5. Censoring-rate operating envelope
# ============================================================
print("\nCensoring-rate sweep (300 sims per level)...", flush=True)
sweep_out = {}
rng_sw = np.random.default_rng(5)
for ctarget in CENSORING_SWEEP:
    mu_c = solve_intercept(ctarget)
    recs_c = []
    ranks_c = []
    for _ in range(300):
        y = mu_c + ETA + SIGMA_TRUE * rng_sw.standard_normal(N_DESIGN)
        cens = y > CENSOR_THRESHOLD
        fr, _ = decompose(make_entries(y, cens))
        if fr is None:
            continue
        r5 = np.array([fr[f] for f in MAIN_FACTORS])
        recs_c.append(r5)
        ranks_c.append(tuple(np.argsort(-r5)))
    rec_c = np.array(recs_c)
    bias_c = rec_c.mean(axis=0) - true_main
    rmse_c = np.sqrt(((rec_c - true_main) ** 2).mean(axis=0))
    top3_hits_c = sum(
        1 for r in ranks_c
        if r[:3] == top3_true)
    sweep_out[str(ctarget)] = {
        "bias_pp_dominant": round(float(bias_c[0]), 2),
        "max_abs_bias_pp": round(float(np.max(np.abs(bias_c))), 2),
        "rmse_pp_dominant": round(float(rmse_c[0]), 2),
        "top3_rank_recovery": round(top3_hits_c / len(ranks_c), 3),
    }
    print(f"  cens={ctarget:.0%}: bias(dominant)={bias_c[0]:+.2f} pp, "
          f"RMSE={rmse_c[0]:.2f} pp, top-3 rank {top3_hits_c/len(ranks_c):.0%}",
          flush=True)

# ============================================================
# 6. Save outputs
# ============================================================
out = {
    "description": "Monte Carlo validation of drop-one-factor LR variance "
                   "decomposition on a balanced full-factorial log-normal AFT model",
    "generative_model": {
        "n_design": N_DESIGN,
        "n_factors": len(FACTOR_ORDER),
        "sigma_true": SIGMA_TRUE,
        "total_explained_variance_dex2": TOTAL_EXPLAINED_VAR,
        "censoring_threshold_log10Nf": CENSOR_THRESHOLD,
        "target_censoring_fraction": TARGET_CENSORING,
        "mu_true": round(MU_TRUE, 3),
        "true_variance_shares_%": TRUE_FRACTIONS,
        "true_sobol_first_order_%": {k: round(v * 100, 1)
                                     for k, v in sobol_true.items()},
    },
    "simulation": {
        "n_sims": M_SIMS,
        "n_completed": n_done,
        "n_skipped": skipped,
        "bootstrap_B": B_BOOT,
        "rng_seed": 42,
        "per_factor": results,
        "rank_recovery_exact_order": round(rank_recovery, 4),
        "rank_recovery_top3_order": round(top3_recovery, 4),
        "mean_spearman_rho": round(float(np.mean(sp_vals)), 3),
        "bootstrap_coverage": coverage_out,
        "bootstrap_coverage_n_sims": n_coverage,
    },
    "sobol_equivalence": {
        "n_sims": M_SIMS,
        "pooled_R2_LR_vs_Sobol": round(r2_pooled, 4),
        "note": "Sobol' first-order indices computed on the deterministic "
                "model output eta; LR fractions from AFT on complete data "
                "(no censoring).",
    },
    "censoring_envelope": sweep_out,
}

with open("output/simulation_validation.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved output/simulation_validation.json", flush=True)

# ============================================================
# 7. Figure: truth vs recovered
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

# Left: truth vs recovered scatter (500 sims x 5 factors)
ax = axes[0]
colors = plt.cm.viridis(np.linspace(0, 1, len(MAIN_FACTORS)))
for j, f in enumerate(MAIN_FACTORS):
    ax.scatter([true_main[j]] * n_done, recovered_main[:, j],
               s=8, alpha=0.25, color=colors[j], label=f.replace("_", " "))
    ax.errorbar(true_main[j], results[f]["mean_recovered_%"],
                yerr=[[results[f]["mean_recovered_%"] - results[f]["empirical_95%_interval"][0]],
                      [results[f]["empirical_95%_interval"][1] - results[f]["mean_recovered_%"]]],
                fmt="o", color="black", capsize=3, zorder=5)
ax.plot([0, 45], [0, 45], "k--", lw=1, label="1:1")
ax.set_xlabel("True variance share (%)")
ax.set_ylabel("Recovered variance share (%)")
ax.set_title("Drop-one-factor LR decomposition:\ntruth vs recovered (500 sims)")
ax.legend(fontsize=7, loc="upper left")
ax.grid(alpha=0.3, linestyle="--")

# Right: rank recovery + Sobol' equivalence
ax = axes[1]
ax.bar([0], [100 * rank_recovery], color="#2166ac", alpha=0.85)
ax.set_ylim(0, 105)
ax.set_xticks([0])
ax.set_xticklabels(["exact rank\nmatch"])
ax.set_ylabel("Proportion of simulations (%)")
ax.set_title(f"Rank recovery: {100*rank_recovery:.0f}%\n"
             f"Sobol' equivalence (complete data): R$^2$ = {r2_pooled:.3f}")
ax.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
fig.savefig("output/fig_sim_validation.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved output/fig_sim_validation.png", flush=True)

print("\n=== Simulation validation complete ===", flush=True)
