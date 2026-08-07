"""
aft_residual_diagnostics.py — AFT model residual diagnostics (v2)
==================================================================
Proper survival-analysis diagnostics for the log-normal AFT model:
  - Cox-Snell residuals (all 144 entries, incl. censored) → Exp(1) check
  - Martingale residuals (all 144 entries) → outlier/pattern detection
  - Standardized residuals on observed entries → Q-Q normality check
  - Levene test on observed residuals

Generates:
  output/fig_diag_qq.png            — 3-panel diagnostic plot
  output/aft_diagnostics.json       — full diagnostic results
"""
import json, sys, os
import numpy as np
from scipy.stats import norm, levene, shapiro, probplot, expon

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)

# ============================================================
# 1. Load sweep data
# ============================================================
print("Loading sweep data...", flush=True)
with open("output/sweep.json") as f:
    sweep = json.load(f)

ENTRIES = []
for r in sweep:
    e = {
        "sn_source": r["sn_source"],
        "mean_stress_method": r["mean_stress_method"],
        "damage_method": r["damage_method"],
        "size_surface_standard": r["size_surface_standard"],
        "rz_level": r["rz_level"],
    }
    nf = r.get("Nf", float("inf"))
    log_nf = r.get("log10_Nf")
    if isinstance(nf, (int, float)) and np.isfinite(nf) and nf > 0 and log_nf is not None:
        e["logNf"] = float(log_nf)
        e["censored"] = 0
    else:
        e["logNf"] = None
        e["censored"] = 1
    ENTRIES.append(e)

MAX_OBS = max(e["logNf"] for e in ENTRIES if e["logNf"] is not None)
for e in ENTRIES:
    if e["censored"] == 1:
        e["logNf_lower"] = MAX_OBS

N_TOTAL = len(ENTRIES)
N_CENSORED = sum(1 for e in ENTRIES if e["censored"] == 1)
N_OBSERVED = N_TOTAL - N_CENSORED
print(f"Total: {N_TOTAL}, Observed: {N_OBSERVED}, Censored: {N_CENSORED}", flush=True)

# ============================================================
# 2. Factor encoding
# ============================================================
FACTORS = {
    "sn_source": ["Waterloo_SMDIdbase_Iter066", "Waterloo_SMDIdbase_Iter068"],
    "mean_stress_method": ["Goodman", "Gerber", "Morrow", "SWT"],
    "damage_method": ["Miner_original", "Miner_modified"],
    "size_surface_standard": ["ISO_6336", "AGMA_2001", "FKM"],
    "rz_level": ["as_forged_Rz50", "ground_Rz4", "machined_Rz15"],
}

FACTOR_ORDER = [
    "mean_stress_method",
    "size_surface_standard",
    "rz_level",
    "sn_source",
    "damage_method",
]

FACTOR_LABELS = {
    "mean_stress_method": "Mean-Stress Correction",
    "size_surface_standard": "Size/Surface Standard",
    "rz_level": "Surface Roughness Rz",
    "sn_source": "S-N Curve Source",
    "damage_method": "Cumulative Damage Rule",
}


def build_design(entries, factors_subset=None):
    if factors_subset is None:
        factors_subset = FACTOR_ORDER
    cols = []
    for fname in factors_subset:
        levels = FACTORS[fname]
        for j in range(1, len(levels)):
            col = np.array([1.0 if e[fname] == levels[j] else 0.0 for e in entries])
            cols.append(col - 1.0 / len(levels))
    X = np.column_stack(cols) if cols else np.zeros((len(entries), 0))
    X = np.column_stack([np.ones(len(entries)), X])
    return X


def extract_arrays(entries):
    y = np.array([e.get("logNf", 0.0) or 0.0 for e in entries])
    cens = np.array([e["censored"] == 1 for e in entries], dtype=bool)
    y_low = np.array([e.get("logNf_lower", 0.0) or 0.0 for e in entries])
    return y, cens, y_low


# ============================================================
# 3. EM algorithm
# ============================================================
def em_censored_normal(X, y_obs, censored, y_lower, n_iter=30, tol=1e-8):
    n, p = X.shape
    obs_mask = ~censored
    beta = np.linalg.lstsq(X[obs_mask], y_obs[obs_mask], rcond=None)[0]
    resid_obs = y_obs[obs_mask] - X[obs_mask] @ beta
    sigma = np.std(resid_obs, ddof=p)
    y_aug = y_obs.copy()
    for iteration in range(n_iter):
        mu = X @ beta
        beta_old = beta.copy()
        sigma_old = sigma
        if censored.any():
            z = (y_lower[censored] - mu[censored]) / sigma
            log_phi = norm.logpdf(z)
            log_sf = norm.logsf(z)
            imr = np.where(z > 30, z, np.where(z < -30, 0.0, np.exp(log_phi - log_sf)))
            y_aug[censored] = mu[censored] + sigma * imr
        beta = np.linalg.lstsq(X, y_aug, rcond=None)[0]
        resid = y_aug - X @ beta
        sigma2 = np.sum(resid[obs_mask]**2) / len(obs_mask)
        if censored.any():
            z = (y_lower[censored] - X[censored] @ beta) / sigma_old
            imr = np.where(z > 30, z, np.where(z < -30, 0.0, np.exp(norm.logpdf(z) - norm.logsf(z))))
            cond_var = sigma_old**2 * (1 + z * imr - imr**2)
            cond_var = np.maximum(cond_var, 0.0)
            sigma2 = (np.sum(resid[obs_mask]**2) + np.sum(cond_var)) / n
        sigma = np.sqrt(max(sigma2, 0.001))
        if np.max(np.abs(beta - beta_old)) < tol and abs(sigma - sigma_old) < tol:
            break
    mu = X @ beta
    ll = 0.0
    resid_obs_final = (y_obs[obs_mask] - mu[obs_mask]) / sigma
    ll += np.sum(norm.logpdf(resid_obs_final)) - obs_mask.sum() * np.log(sigma)
    if censored.any():
        z_cens = (y_lower[censored] - mu[censored]) / sigma
        ll += np.sum(norm.logsf(z_cens))
    return beta, sigma, ll, mu, y_aug


# ============================================================
# 4. Fit model
# ============================================================
print("Fitting full AFT model...", flush=True)
X_FULL = build_design(ENTRIES, FACTOR_ORDER)
Y_OBS, CENS, Y_LOWER = extract_arrays(ENTRIES)
beta_full, sigma_full, ll_full, mu_full, y_aug = em_censored_normal(X_FULL, Y_OBS, CENS, Y_LOWER)

obs_mask = ~CENS

# ============================================================
# 5. Cox-Snell residuals (all 144 entries, including censored)
# ============================================================
# For log-normal AFT: S(t|mu,sigma) = 1 - Phi((log(t) - mu)/sigma)
# Cox-Snell residual: r_CS = -log(S(t)) for observed
#                     r_CS = -log(S(c)) for censored
# Under correct specification, r_CS ~ Exp(1)
# -> Cumulative hazard of r_CS should be ~ straight line with slope 1

print("\n=== Cox-Snell Residuals ===", flush=True)
cs_residuals = np.zeros(N_TOTAL)
for i in range(N_TOTAL):
    if CENS[i]:
        # Censored: survival at censoring threshold
        z_cens = (Y_LOWER[i] - mu_full[i]) / sigma_full
        S_cens = norm.sf(z_cens)  # P(T > c)
        cs_residuals[i] = -np.log(max(S_cens, 1e-15))
    else:
        # Observed: survival at event time
        z_obs = (Y_OBS[i] - mu_full[i]) / sigma_full
        S_obs = norm.sf(z_obs)
        cs_residuals[i] = -np.log(max(S_obs, 1e-15))

print(f"  Cox-Snell residuals: mean={np.mean(cs_residuals):.4f} (target: 1.0 for Exp(1))", flush=True)
print(f"  Range: [{np.min(cs_residuals):.4f}, {np.max(cs_residuals):.4f}]", flush=True)

# Nelson-Aalen cumulative hazard for CS residuals
cs_sorted = np.sort(cs_residuals)
n_at_risk = np.arange(N_TOTAL, 0, -1)
cumhaz_cs = np.cumsum(1.0 / n_at_risk)

# KS test for Exp(1)
# Sort CS residuals, compute empirical CDF, compare to Exp(1) CDF
ecdf_y = np.arange(1, N_TOTAL + 1) / N_TOTAL
exp_cdf = expon.cdf(cs_sorted)
ks_stat = np.max(np.abs(ecdf_y - exp_cdf))
print(f"  KS statistic vs Exp(1): {ks_stat:.4f}", flush=True)

# ============================================================
# 6. Martingale residuals (all 144 entries)
# ============================================================
# M_i = delta_i - r_CS_i
# delta_i = 1 for observed, 0 for censored
# Range: (-inf, 1], mean ~ 0 for well-specified model

print("\n=== Martingale Residuals ===", flush=True)
delta = np.where(CENS, 0.0, 1.0)
martingale = delta - cs_residuals

print(f"  Martingale residuals: mean={np.mean(martingale):.4f} (target: 0)", flush=True)
print(f"  Range: [{np.min(martingale):.4f}, {np.max(martingale):.4f}]", flush=True)
print(f"  Censored mean: {np.mean(martingale[CENS]):.4f}", flush=True)
print(f"  Observed mean:  {np.mean(martingale[obs_mask]):.4f}", flush=True)

# Check for extreme outliers (|M| > 2 is suspicious)
n_outliers = np.sum(np.abs(martingale) > 2)
if n_outliers > 0:
    print(f"  Extreme Martingale outliers (|M|>2): {n_outliers}/{N_TOTAL}", flush=True)
else:
    print(f"  No extreme Martingale outliers (all |M| <= 2)", flush=True)

# ============================================================
# 7. Standardized residuals on observed entries
# ============================================================
resid_std = (Y_OBS[obs_mask] - mu_full[obs_mask]) / sigma_full
sw_stat, sw_p = shapiro(resid_std)

print(f"\n=== Standardized Residuals (observed, n={N_OBSERVED}) ===", flush=True)
print(f"  Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4f}", flush=True)
print(f"  Residual mean:  {np.mean(resid_std):.4f}", flush=True)
print(f"  Residual std:   {np.std(resid_std):.4f}", flush=True)
print(f"  Skewness:       {np.mean(resid_std**3):.4f}", flush=True)

# ============================================================
# 8. Q-Q Plot — 3 panels
# ============================================================
print("\n=== Generating Diagnostic Plot ===", flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(12, 10))
gs = GridSpec(2, 2, figure=fig)

# Panel A: Cox-Snell cumulative hazard (model adequacy check)
ax = fig.add_subplot(gs[0, 0])
ax.plot(cs_sorted, cumhaz_cs, 'b-', linewidth=2, label='Cox-Snell residuals')
ax.plot([0, max(cs_sorted)], [0, max(cs_sorted)], 'r--', linewidth=1.5, label='Exp(1) reference (slope=1)')
ax.fill_between(cs_sorted, cumhaz_cs - 1.96/np.sqrt(n_at_risk),
                cumhaz_cs + 1.96/np.sqrt(n_at_risk), alpha=0.15, color='blue')
ax.set_xlabel('Cox-Snell Residual', fontsize=10)
ax.set_ylabel('Cumulative Hazard', fontsize=10)
ax.set_title('Cox-Snell Residual Diagnostics\n(All 144 entries; n_censored=92)', fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.text(0.95, 0.05, f'KS(Exp(1))={ks_stat:.3f}',
        transform=ax.transAxes, fontsize=9, ha='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_xlim(0, max(cs_sorted) * 1.05)
ax.set_ylim(0, max(cumhaz_cs) * 1.05)

# Panel B: Martingale residuals vs fitted values
ax = fig.add_subplot(gs[0, 1])
colors = ['#d62728' if c else '#1f77b4' for c in CENS]
ax.scatter(mu_full, martingale, c=colors, alpha=0.6, edgecolors='black', linewidth=0.3)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
# Loess-like smooth
sort_idx = np.argsort(mu_full)
window = 20
smooth_y = np.convolve(martingale[sort_idx], np.ones(window)/window, mode='same')
ax.plot(mu_full[sort_idx], smooth_y, 'k-', linewidth=2, label=f'Running mean (w={window})')
ax.set_xlabel('Fitted log10(Nf)', fontsize=10)
ax.set_ylabel('Martingale Residual', fontsize=10)
ax.set_title('Martingale Residuals vs Fitted Values\n(All 144 entries)', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
# Legend for colors
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#d62728', label=f'Censored (n={N_CENSORED})'),
                   Patch(facecolor='#1f77b4', label=f'Observed (n={N_OBSERVED})')]
ax.legend(handles=legend_elements, fontsize=8, loc='lower left')

# Panel C: Q-Q plot of standardized residuals (observed only)
ax = fig.add_subplot(gs[1, :])
(osm, osr), (slope, intercept, r) = probplot(resid_std, dist="norm", plot=ax)
ax.set_title(f'Q-Q Plot: Standardized Residuals (Observed, n={N_OBSERVED})\n'
             f'Shapiro-Wilk W={sw_stat:.4f}, p={sw_p:.4f}  |  '
             f'Cox-Snell KS(Exp(1))={ks_stat:.3f}  |  '
             f'Martingale mean={np.mean(martingale):.3f}',
             fontsize=11, fontweight='bold')
ax.set_xlabel("Theoretical Quantiles", fontsize=10)
ax.set_ylabel("Sample Quantiles", fontsize=10)
ax.text(0.05, 0.95, f'Q-Q R² = {r**2:.3f}', transform=ax.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(pad=2)
plt.savefig("output/fig_diag_qq.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved output/fig_diag_qq.png", flush=True)

# ============================================================
# 9. Levene's Test
# ============================================================
print("\n=== Levene's Test ===", flush=True)
obs_indices = np.where(obs_mask)[0]
levene_results = {}

for fname in FACTOR_ORDER:
    levels = FACTORS[fname]
    groups = []
    for level in levels:
        group_resid = []
        for j, idx in enumerate(obs_indices):
            if ENTRIES[idx][fname] == level:
                group_resid.append(float(resid_std[j]))
        groups.append(group_resid)

    valid_groups = [g for g in groups if len(g) >= 2]
    if len(valid_groups) >= 2:
        stat, p = levene(*valid_groups, center='median')
        verdict = "Homoscedastic OK" if p > 0.05 else "Heteroscedastic (p<0.05)"
        print(f"  {FACTOR_LABELS[fname]:30s}: Levene stat={stat:.4f}, p={p:.4f}  {verdict}", flush=True)
        levene_results[fname] = {
            "factor": FACTOR_LABELS[fname],
            "levene_stat": round(float(stat), 4),
            "p_value": round(float(p), 4),
            "homoscedastic": bool(p > 0.05),
            "group_sizes": [len(g) for g in groups],
            "group_variances": [round(float(np.var(g, ddof=1)), 4) for g in groups],
        }

# ============================================================
# 10. Save results
# ============================================================
diagnostics = {
    "description": "AFT model residual diagnostics — Cox-Snell, Martingale, standardized residuals",
    "model": {
        "distribution": "Log-normal AFT with right-censoring (EM estimation)",
        "n_total": N_TOTAL,
        "n_observed": N_OBSERVED,
        "n_censored": N_CENSORED,
        "sigma_aft": round(float(sigma_full), 4),
        "logLik": round(float(ll_full), 2),
    },
    "cox_snell": {
        "description": "Cox-Snell residuals. Under correct model, should follow Exp(1).",
        "n": N_TOTAL,
        "mean": round(float(np.mean(cs_residuals)), 4),
        "mean_target": 1.0,
        "ks_statistic_vs_exp1": round(float(ks_stat), 4),
        "interpretation": "KS statistic measures departure from Exp(1). Lower is better. Formal test not applied due to censoring; visual inspection of cumulative hazard vs identity line is the primary diagnostic."
    },
    "martingale": {
        "description": "Martingale residuals. M_i = delta_i - r_CS_i. Range (-inf, 1]. Mean ~ 0 for correct model.",
        "n": N_TOTAL,
        "mean": round(float(np.mean(martingale)), 4),
        "mean_target": 0.0,
        "range": [round(float(np.min(martingale)), 4), round(float(np.max(martingale)), 4)],
        "n_extreme_outliers": int(n_outliers),
    },
    "standardized_residuals_observed": {
        "n": N_OBSERVED,
        "mean": round(float(np.mean(resid_std)), 4),
        "std": round(float(np.std(resid_std)), 4),
        "skewness": round(float(np.mean(resid_std**3)), 4),
    },
    "shapiro_wilk": {
        "statistic": round(float(sw_stat), 4),
        "p_value": round(float(sw_p), 4),
        "normal_at_0.05": bool(sw_p > 0.05),
    },
    "levene_test": levene_results,
    "qq_plot_r_squared": round(float(r**2), 4),
}

with open("output/aft_diagnostics.json", "w") as f:
    json.dump(diagnostics, f, indent=2)
print("\nSaved output/aft_diagnostics.json", flush=True)

# Summary
print("\n=== Summary ===", flush=True)
print(f"  Cox-Snell residuals: mean={np.mean(cs_residuals):.3f}, KS(Exp(1))={ks_stat:.3f}", flush=True)
print(f"  Martingale residuals: mean={np.mean(martingale):.3f}", flush=True)
print(f"  Shapiro-Wilk (observed): W={sw_stat:.4f}, p={sw_p:.4f}", flush=True)
print(f"  Levene: {sum(1 for v in levene_results.values() if not v.get('homoscedastic', True))}/{len(levene_results)} heteroscedastic", flush=True)
print("\n=== Complete ===", flush=True)
