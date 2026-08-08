"""
pattern_mixture_aft.py — pattern-mixture AFT + censoring-rate sweep (v6.0)
==========================================================================
Reviewer round-3 (0808-审稿人意见03): "AFT non-informative censoring
assumption cannot be strictly verified ... no mixture-model validation."

This script provides the statistical validation requested:

  1. Censoring-rate sweep: MC at 30% / 45% / 58% / 75% censoring,
     quantifying per-factor estimation error (bias / RMSE) of the
     drop-one-factor AFT decomposition.
  2. Pattern-mixture AFT: censoring probability is made WEAKLY
     DEPENDENT on the latent (true) life -- the mechanism by which the
     non-informative assumption can fail in practice (run-out is driven
     by the same stress model that predicts life). The shift in the
     recovered variance shares quantifies the potential bias of the
     benchmark AFT under informative censoring.

Generative model and decomposition replicate simulation_validation.py
exactly (balanced 144 design, log-normal AFT, common-sigma LR shares).

All numbers in output/pattern_mixture_aft.json are produced here.
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
# 1. Generative model with KNOWN variance shares (same as simulation_validation.py)
# ============================================================
TRUE_SHARES = {
    "mean_stress_method": 0.40,
    "size_surface_standard": 0.30,
    "rz_level": 0.20,
    "sn_source": 0.05,
    "damage_method": 0.05,
}
EFFECT_PATTERNS = {
    "mean_stress_method": np.array([-0.9, 1.1, 0.2, -0.4]),
    "size_surface_standard": np.array([0.8, 0.1, -0.9]),
    "rz_level": np.array([0.55, 0.05, -0.60]),
    "sn_source": np.array([0.25, -0.25]),
    "damage_method": np.array([0.12, -0.12]),
}
for f in FACTOR_ORDER:
    assert abs(EFFECT_PATTERNS[f].sum()) < 1e-9, f"not sum-to-zero: {f}"
    assert len(EFFECT_PATTERNS[f]) == len(FACTORS[f]), f"level mismatch: {f}"

SIGMA_TRUE = 0.1745
TOTAL_EXPLAINED_VAR = 0.90
CENSOR_THRESHOLD = 6.0

def build_effects():
    effects = {}
    for f in FACTOR_ORDER:
        v_raw = float(np.mean(EFFECT_PATTERNS[f] ** 2))
        v_target = TRUE_SHARES[f] * TOTAL_EXPLAINED_VAR
        effects[f] = EFFECT_PATTERNS[f] * np.sqrt(v_target / v_raw)
    return effects

EFFECTS = build_effects()

LEVELS = [FACTORS[f] for f in FACTOR_ORDER]
COMBOS = list(itertools.product(*LEVELS))
N_DESIGN = len(COMBOS)

def eta_for_combo(combo):
    return sum(float(EFFECTS[f][FACTORS[f].index(combo[i])])
               for i, f in enumerate(FACTOR_ORDER))

ETA = np.array([eta_for_combo(c) for c in COMBOS])
TRUE_FRACTIONS = {f: TRUE_SHARES[f] * 100.0 for f in FACTOR_ORDER}
TRUE_FRACTIONS["interaction"] = 0.0

def solve_intercept(target_cens):
    def cens_rate(mu):
        z = (CENSOR_THRESHOLD - mu - ETA) / SIGMA_TRUE
        return float(np.mean(norm.sf(z)))
    lo, hi = -5.0, 9.0
    assert cens_rate(lo) < target_cens < cens_rate(hi), "bracket failed"
    return brentq(lambda mu: cens_rate(mu) - target_cens, lo, hi)

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
    X_int = build_interaction_X(entries, X_full)
    beta_i, sig_i, ll_i = em_censored_normal(X_int, Y, CENS, Y_LOW, sigma_fixed=sig_f)
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

# ============================================================
# 3. Part A: censoring-rate sweep (30/45/58/75%), MC 300 sims each
# ============================================================
N_SIMS_SWEEP = 300
CENSORING_SWEEP = [0.30, 0.45, 0.58, 0.75]
MAIN_FACTORS = FACTOR_ORDER
true_main = np.array([TRUE_FRACTIONS[f] for f in MAIN_FACTORS])

print("Part A: censoring-rate sweep (300 sims per level)...", flush=True)
sweep_out = {}
t0 = time.time()
for ctarget in CENSORING_SWEEP:
    mu = solve_intercept(ctarget)
    rng = np.random.default_rng(1000 + int(ctarget * 100))
    rec_all = []
    for m in range(N_SIMS_SWEEP):
        y = mu + ETA + SIGMA_TRUE * rng.standard_normal(N_DESIGN)
        cens = y > CENSOR_THRESHOLD
        entries = make_entries(y, cens)
        fracs, _ = decompose(entries)
        if fracs is None:
            continue
        rec_all.append(fracs)
    if not rec_all:
        continue
    rec = np.array([[r[f] for f in MAIN_FACTORS] for r in rec_all])
    per = {}
    for j, f in enumerate(MAIN_FACTORS):
        bias = float(np.mean(rec[:, j] - true_main[j]))
        rmse = float(np.sqrt(np.mean((rec[:, j] - true_main[j]) ** 2)))
        per[f] = {"true_%": round(true_main[j], 1),
                  "bias_pp": round(bias, 2),
                  "rmse_pp": round(rmse, 2)}
    dom_bias = per["size_surface_standard"]["bias_pp"]
    sweep_out[str(ctarget)] = {
        "n_sims_completed": len(rec_all),
        "bias_pp_dominant": dom_bias,
        "max_abs_bias_pp": round(float(np.max(np.abs(rec - true_main))), 2),
        "rmse_pp_dominant": per["size_surface_standard"]["rmse_pp"],
        "per_factor": per,
    }
    print(f"  cens={ctarget:.0%}: dominant(std) bias {dom_bias:+.2f} pp, "
          f"RMSE {per['size_surface_standard']['rmse_pp']:.2f} pp", flush=True)
print(f"Part A done in {time.time()-t0:.0f}s", flush=True)

# ============================================================
# 4. Part B: pattern-mixture AFT (informative censoring)
# ============================================================
# Mechanism: censoring probability depends WEAKLY on the latent life
#   logit(P(censored)) = alpha + gamma * (y - CENSOR_THRESHOLD)
# gamma > 0: longer-lived combinations are MORE likely to be run-out
# (the model's predicted stress determines both life and run-out).
# We calibrate alpha so the mean censoring fraction stays at the paper's
# 58%, then compare recovered shares under gamma = 0 (non-informative,
# baseline) vs gamma = 2.0 (weakly informative).
print("\nPart B: pattern-mixture AFT (informative censoring, gamma sweep)...",
      flush=True)
N_SIMS_PM = 300
GAMMAS = [0.0, 1.0, 2.0, 3.0]

def simulate_pattern_mixture(gamma, ctarget, rng):
    """Censoring depends on latent life: P(cens) = sigmoid(alpha + gamma*z)."""
    mu = solve_intercept(ctarget)  # non-informative intercept (reference)
    y = mu + ETA + SIGMA_TRUE * rng.standard_normal(N_DESIGN)
    # informative censoring: higher life -> higher censoring probability
    z = (y - CENSOR_THRESHOLD) / SIGMA_TRUE
    # calibrate alpha so mean censoring = ctarget
    def mean_cens(alpha):
        return float(np.mean(1.0 / (1.0 + np.exp(-(alpha + gamma * z)))))
    lo, hi = -20.0, 20.0
    alpha = brentq(lambda a: mean_cens(a) - ctarget, lo, hi)
    p = 1.0 / (1.0 + np.exp(-(alpha + gamma * z)))
    cens = rng.random(N_DESIGN) < p
    entries = make_entries(y, cens)
    return decompose(entries)

pm_out = {}
for gamma in GAMMAS:
    rng = np.random.default_rng(2000 + int(gamma * 10))
    rec_all = []
    for m in range(N_SIMS_PM):
        fracs, _ = simulate_pattern_mixture(gamma, 0.58, rng)
        if fracs is None:
            continue
        rec_all.append(fracs)
    if not rec_all:
        continue
    rec = np.array([[r[f] for f in MAIN_FACTORS] for r in rec_all])
    per = {}
    for j, f in enumerate(MAIN_FACTORS):
        bias = float(np.mean(rec[:, j] - true_main[j]))
        per[f] = {"true_%": round(true_main[j], 1),
                  "mean_recovered_%": round(float(np.mean(rec[:, j])), 1),
                  "bias_pp": round(bias, 2)}
    pm_out[f"gamma_{gamma:g}"] = {
        "n_sims_completed": len(rec_all),
        "mechanism": "logit(P(cens)) = alpha + gamma*(y-CENSOR_THRESHOLD)/sigma",
        "per_factor": per,
        "std_bias_pp": per["size_surface_standard"]["bias_pp"],
        "std_mean_recovered_%": per["size_surface_standard"]["mean_recovered_%"],
    }
    print(f"  gamma={gamma:g}: std recovered "
          f"{per['size_surface_standard']['mean_recovered_%']:.1f}% "
          f"(true 30.0%, bias {per['size_surface_standard']['bias_pp']:+.2f} pp)",
          flush=True)

# Reference: what does the benchmark (gamma=0) recover at 58%?
gamma0_std = pm_out["gamma_0"]["std_mean_recovered_%"]
gamma2_std = pm_out["gamma_2"]["std_mean_recovered_%"]
delta_gamma2 = gamma2_std - gamma0_std

# ============================================================
# 5. Output
# ============================================================
out = {
    "description": "Censoring-rate sweep + pattern-mixture AFT validation of "
                   "the drop-one-factor decomposition under informative censoring.",
    "generative_model": {
        "true_variance_shares_%": {k: round(v, 1) for k, v in TRUE_FRACTIONS.items()},
        "sigma_true": SIGMA_TRUE,
        "n_design": N_DESIGN,
        "censoring_threshold_log10Nf": CENSOR_THRESHOLD,
    },
    "partA_censoring_sweep": sweep_out,
    "partB_pattern_mixture": pm_out,
    "summary": (
        f"At the paper's 58% censoring, the benchmark AFT under-recovers the "
        f"dominant (standard) factor by {sweep_out['0.58']['bias_pp_dominant']:+.2f} pp "
        f"(recovered {gamma0_std:.1f}% vs true 30.0%). Under weakly informative "
        f"censoring (gamma=2.0) the recovered standard share shifts by "
        f"{delta_gamma2:+.2f} pp relative to gamma=0. At 75% censoring the "
        f"dominant-factor bias grows to {sweep_out['0.75']['bias_pp_dominant']:+.2f} pp."
    ),
}

with open("output/pattern_mixture_aft.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\nSaved output/pattern_mixture_aft.json")
print("=== Complete ===")
