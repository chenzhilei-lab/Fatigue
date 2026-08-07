"""
run_weibull_comparison.py — Log-normal AFT vs Weibull AFT distribution comparison
===================================================================================
Fits both log-normal and Weibull AFT models to a 72-combination subset
(2 SN × 4 MS × 3 Std × 3 Rz) at 700 Nm. Reports AIC/BIC to justify
the log-normal choice.

Output: output/weibull_aft.json
"""

import json, os, numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 1. Load data, build 72-combination subset
# ============================================================
with open('output/sweep.json') as f:
    sweep = json.load(f)

# Subset: 2 SN × 4 MS × 3 Std × 3 Rz = 72 combos (fix damage to Miner_original)
subset = [r for r in sweep if r['damage_method'] == 'Miner_original']
print(f"Subset entries: {len(subset)}")

# Build AFT entries
entries = []
for r in subset:
    nf = r.get('Nf', float('inf'))
    ln = r.get('log10_Nf')
    e = {}
    if isinstance(nf, (int, float)) and np.isfinite(nf) and nf > 0 and ln is not None:
        e['logNf'] = float(ln)
        e['censored'] = 0
    else:
        e['logNf'] = None
        e['censored'] = 1
    entries.append(e)

MAX_OBS = max(e['logNf'] for e in entries if e['logNf'] is not None)
for e in entries:
    if e['censored'] == 1:
        e['logNf_lower'] = MAX_OBS

n_censored = sum(e['censored'] for e in entries)
n_observed = len(entries) - n_censored
print(f"Observed: {n_observed}, Censored: {n_censored}")

# ============================================================
# 2. Build factor design matrix (sum-to-zero contrasts)
# ============================================================
FACTOR_LEVELS = {
    'sn_source': ['Waterloo_SMDIdbase_Iter068', 'Waterloo_SMDIdbase_Iter066'],
    'mean_stress_method': ['Goodman', 'Gerber', 'Morrow', 'SWT'],
    'size_surface_standard': ['ISO_6336', 'AGMA_2001', 'FKM'],
    'rz_level': ['as_forged_Rz50', 'ground_Rz4', 'machined_Rz15'],
}
FACTOR_ORDER = ['mean_stress_method', 'size_surface_standard', 'rz_level', 'sn_source']

def build_design(entries):
    cols = []
    for fname in FACTOR_ORDER:
        levels = FACTOR_LEVELS[fname]
        for j in range(1, len(levels)):
            col = np.array([1.0 if e[fname] == levels[j] else 0.0 for e in entries])
            cols.append(col - 1.0/len(levels))
    X = np.column_stack(cols) if cols else np.zeros((len(entries), 0))
    X = np.column_stack([np.ones(len(entries)), X])
    return X

# Encode factor info
for e, r in zip(entries, subset):
    e['sn_source'] = r['sn_source']
    e['mean_stress_method'] = r['mean_stress_method']
    e['size_surface_standard'] = r['size_surface_standard']
    e['rz_level'] = r['rz_level']

X = build_design(entries)
n_params_beta = X.shape[1]
print(f"Design matrix: {X.shape[0]} × {X.shape[1]}")

y_obs = np.array([e.get('logNf', 0.0) or 0.0 for e in entries])
cens = np.array([e['censored'] == 1 for e in entries], dtype=bool)
y_lower = np.array([e.get('logNf_lower', 0.0) or 0.0 for e in entries])

# ============================================================
# 3. Log-normal AFT MLE
# ============================================================
def lognormal_neg_loglik(params):
    beta = params[:-1]
    log_sigma = params[-1]
    sigma = np.exp(log_sigma)
    mu = X @ beta

    obs_mask = ~cens
    resid_obs = (y_obs[obs_mask] - mu[obs_mask]) / sigma
    ll = np.sum(norm.logpdf(resid_obs)) - obs_mask.sum() * np.log(sigma)

    if cens.any():
        z_cens = (y_lower[cens] - mu[cens]) / sigma
        ll += np.sum(norm.logsf(z_cens))
    return -ll

# Initial guess from OLS on observed
from numpy.linalg import lstsq
beta0 = lstsq(X[~cens], y_obs[~cens], rcond=None)[0]
sigma0 = np.std(y_obs[~cens] - X[~cens] @ beta0)
params0 = np.append(beta0, np.log(max(sigma0, 0.01)))

res_ln = minimize(lognormal_neg_loglik, params0, method='L-BFGS-B', options={'maxiter': 5000})
ll_ln = -res_ln.fun
sigma_ln = np.exp(res_ln.x[-1])
n_params = n_params_beta + 1  # betas + sigma

# ============================================================
# 4. Weibull AFT MLE (extreme value / Gumbel errors on log-T)
# ============================================================
def weibull_neg_loglik(params):
    """
    Weibull AFT: log(T) = Xβ + σ·W, W ~ standard Gumbel (extreme value type I)
    Gumbel pdf: f(w) = exp(-w - exp(-w))
    Gumbel sf:  S(w) = exp(-exp(-w))
    """
    beta = params[:-1]
    log_sigma = params[-1]
    sigma = np.exp(log_sigma)
    mu = X @ beta

    ll = 0.0
    for i in range(len(entries)):
        if cens[i] == 0:
            w = (y_obs[i] - mu[i]) / sigma
            # Gumbel log-pdf: -w - exp(-w) - log(sigma)
            ll += -w - np.exp(-w) - np.log(sigma)
        else:
            w = (y_lower[i] - mu[i]) / sigma
            # Gumbel log-survival: log(1 - F(w)) = -exp(-w)
            ll += -np.exp(-w)
    return -ll

# Initial guess same as log-normal
res_w = minimize(weibull_neg_loglik, params0, method='L-BFGS-B', options={'maxiter': 5000})
ll_w = -res_w.fun
sigma_w = np.exp(res_w.x[-1])

# ============================================================
# 5. AIC / BIC comparison
# ============================================================
aic_ln = 2 * n_params - 2 * ll_ln
bic_ln = n_params * np.log(n_observed) - 2 * ll_ln
aic_w = 2 * n_params - 2 * ll_w
bic_w = n_params * np.log(n_observed) - 2 * ll_w

print(f"\n=== Distribution Comparison (72-combo subset, 700 Nm) ===")
print(f"{'Model':<20s} {'logLik':>10s} {'sigma':>8s} {'AIC':>10s} {'BIC':>10s} {'n_params':>8s}")
print("-" * 60)
print(f"{'Log-normal AFT':<20s} {ll_ln:>10.1f} {sigma_ln:>8.4f} {aic_ln:>10.1f} {bic_ln:>10.1f} {n_params:>8d}")
print(f"{'Weibull AFT':<20s} {ll_w:>10.1f} {sigma_w:>8.4f} {aic_w:>10.1f} {bic_w:>10.1f} {n_params:>8d}")

delta_aic = aic_w - aic_ln
delta_bic = bic_w - bic_ln
print(f"\nΔAIC (Weibull - Lognormal): {delta_aic:+.1f}")
print(f"ΔBIC (Weibull - Lognormal): {delta_bic:+.1f}")

if delta_aic > 0:
    print("=> Log-normal AFT preferred (lower AIC)")
else:
    print(f"=> Weibull AFT has lower AIC by {abs(delta_aic):.1f}")

# ============================================================
# 6. Save
# ============================================================
output = {
    'description': 'Log-normal vs Weibull AFT distribution comparison (72-combo subset, 700 Nm)',
    'n_entries': len(entries),
    'n_observed': n_observed,
    'n_censored': n_censored,
    'n_parameters': n_params,
    'log_normal': {
        'logLik': round(float(ll_ln), 2),
        'sigma': round(float(sigma_ln), 4),
        'AIC': round(float(aic_ln), 1),
        'BIC': round(float(bic_ln), 1),
    },
    'weibull': {
        'logLik': round(float(ll_w), 2),
        'sigma': round(float(sigma_w), 4),
        'AIC': round(float(aic_w), 1),
        'BIC': round(float(bic_w), 1),
    },
    'delta_AIC_weibull_minus_lognormal': round(float(delta_aic), 1),
    'delta_BIC_weibull_minus_lognormal': round(float(delta_bic), 1),
    'preferred_model': 'Log-normal AFT' if delta_aic > 0 else 'Weibull AFT',
    'note': 'The comparison uses a 72-combo subset (Miner_original only) with full '
            'factorial design matrix (10 beta parameters + 1 sigma). The log-normal '
            'distribution is retained for the main analysis because (i) log10(Nf) is '
            'the natural scale for Basquin-type S-N comparisons and (ii) the likelihood-'
            'ratio decomposition under log-normality is equivalent to Sobol first-order '
            'indices for balanced factorial designs.',
}

with open('output/weibull_aft.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nSaved output/weibull_aft.json")
