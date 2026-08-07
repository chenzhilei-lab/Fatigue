"""
multi_aft_comparison.py — Multi-distribution AFT model comparison (强制项⑨)
==========================================================================
Fits log-normal, log-logistic, and Gamma AFT models to the same 108-combo
3-method dataset. Computes AIC for each and reports variance decomposition
under each distribution to quantify model-assumption sensitivity.
"""
import json, sys, os
import numpy as np
from scipy.stats import norm, logistic
from scipy.optimize import minimize

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open("output/sweep.json") as f:
    sweep = json.load(f)

# 3-method baseline: exclude Goodman
entries_raw = [r for r in sweep if r["mean_stress_method"] != "Goodman"]

FACTORS = {
    "sn_source": ["Waterloo_SMDIdbase_Iter066", "Waterloo_SMDIdbase_Iter068"],
    "mean_stress_method": ["Gerber", "Morrow", "SWT"],
    "damage_method": ["Miner_original", "Miner_modified"],
    "size_surface_standard": ["ISO_6336", "AGMA_2001", "FKM"],
    "rz_level": ["as_forged_Rz50", "ground_Rz4", "machined_Rz15"],
}
F_ORDER = ["mean_stress_method","size_surface_standard","rz_level","sn_source","damage_method"]

E = []
for r in entries_raw:
    e = {k: r[k] for k in ["sn_source","mean_stress_method","damage_method","size_surface_standard","rz_level"]}
    nf = r.get("Nf", float("inf"))
    log_nf = r.get("log10_Nf")
    if isinstance(nf,(int,float)) and np.isfinite(nf) and nf>0 and log_nf is not None:
        e["y"]=float(log_nf); e["c"]=0
    else:
        e["y"]=None; e["c"]=1
    E.append(e)
mv = max(e["y"] for e in E if e["y"] is not None)
for e in E:
    if e["c"]==1: e["yl"]=mv

def build_X(entries):
    cols = []
    for fn in F_ORDER:
        lv = FACTORS[fn]
        for j in range(1,len(lv)):
            col = np.array([1.0 if e[fn]==lv[j] else 0.0 for e in entries])
            cols.append(col - 1.0/len(lv))
    X = np.column_stack(cols) if cols else np.zeros((len(entries),0))
    return np.column_stack([np.ones(len(entries)), X])

Xf = build_X(E)
yo = np.array([e.get("y",0)or 0 for e in E])
c = np.array([e["c"]==1 for e in E], dtype=bool)
yl = np.array([e.get("yl",0)or 0 for e in E])
om = ~c
n, p = Xf.shape

# ============================================================
# 1. Log-normal AFT (existing; EM algorithm)
# ============================================================
def em_lognormal(X, yo, c, yl, n_iter=30, sigma_fixed=None):
    beta = np.linalg.lstsq(X[om], yo[om], rcond=None)[0]
    sigma = (sigma_fixed if sigma_fixed is not None
             else max(np.std(yo[om]-X[om]@beta, ddof=p), 0.001))
    ya = yo.copy()
    for _ in range(n_iter):
        mu = X@beta; so = sigma
        if c.any():
            z = (yl[c]-mu[c])/sigma
            imr = np.where(z>30,z,np.where(z<-30,0.0,np.exp(norm.logpdf(z)-norm.logsf(z))))
            ya[c] = mu[c]+sigma*imr
        beta = np.linalg.lstsq(X, ya, rcond=None)[0]
        r = ya - X@beta
        if sigma_fixed is None:
            s2 = np.sum(r[om]**2)/max(om.sum(),1)
            if c.any():
                z = (yl[c]-X[c]@beta)/so
                imr = np.where(z>30,z,np.where(z<-30,0.0,np.exp(norm.logpdf(z)-norm.logsf(z))))
                cv = so**2*(1+z*imr-imr**2); cv = np.maximum(cv,0.0)
                s2 = (np.sum(r[om]**2)+np.sum(cv))/n
            sigma = np.sqrt(max(s2,0.001))
    mu = X@beta
    ll = np.sum(norm.logpdf((yo[om]-mu[om])/sigma))-om.sum()*np.log(sigma)
    if c.any(): ll += np.sum(norm.logsf((yl[c]-mu[c])/sigma))
    k = p + 1  # betas + sigma
    aic = -2*ll + 2*k
    return beta, sigma, ll, aic

# ============================================================
# 2. Log-logistic AFT (MLE via optimization)
# ============================================================
def nll_loglogistic(params, X, yo, c, yl):
    beta = params[:-1]; sigma = np.exp(params[-1])
    mu = X@beta
    ll = 0.0
    # Observed: log-logistic density
    z_obs = (yo[om]-mu[om])/sigma
    ll += np.sum(z_obs - 2*np.log(1+np.exp(z_obs)) - np.log(sigma))
    # Censored: log-logistic survival
    if c.any():
        z_cens = (yl[c]-mu[c])/sigma
        ll += np.sum(-np.log(1+np.exp(z_cens)))
    return -ll

beta0 = np.linalg.lstsq(Xf[om], yo[om], rcond=None)[0]
s0 = np.log(max(np.std(yo[om]-Xf[om]@beta0), 0.01))
res_ll = minimize(nll_loglogistic, np.append(beta0, s0), args=(Xf, yo, c, yl),
                  method='L-BFGS-B', options={'maxiter':2000})
beta_ll, sigma_ll = res_ll.x[:-1], np.exp(res_ll.x[-1])
ll_ll = -res_ll.fun
k_ll = p+1
aic_ll = -2*ll_ll + 2*k_ll

print(f"Log-logistic: sigma={sigma_ll:.4f}, logLik={ll_ll:.2f}, AIC={aic_ll:.1f}")

# ============================================================
# 3. Gamma AFT (MLE via optimization)
# ============================================================
# Gamma AFT: T ~ Gamma(shape=1/sigma^2, scale=sigma^2 * exp(mu))
# log(T) = mu + sigma * log(G)
# Survival: S(t) = 1 - F_gamma(t; shape, scale)
from scipy.special import gammaln, gammainc, gammaincc

def nll_gamma(params, X, yo, c, yl):
    beta = params[:-1]
    sigma = np.exp(params[-1])  # sigma > 0
    mu = X@beta
    shape = 1.0/(sigma*sigma)
    ll = 0.0
    for i in range(n):
        scale = sigma*sigma * np.exp(mu[i])
        if c[i]:
            # Censored: survival
            t = 10.0**yl[i]
            if t <= 0: t = 1e-10
            S = gammaincc(shape, t/scale) if t/scale > 0 else 1.0
            S = max(S, 1e-15)
            ll += np.log(S)
        else:
            # Observed: density
            t = 10.0**yo[i]
            if t <= 0: t = 1e-10
            ll += (shape-1)*np.log(t) - t/scale - shape*np.log(scale) - gammaln(shape)
    return -ll

# Simpler approach: fit Gamma to exp(mu)*epsilon where epsilon~Gamma
# log(T) = mu + log(epsilon)
# Mean of log(T) = mu + psi(shape) - log(1/sigma^2) = mu + psi(1/s^2) + 2*log(s)
# Not trivial. Use MLE on time scale directly.
res_g = minimize(nll_gamma, np.append(beta0, s0), args=(Xf, yo, c, yl),
                 method='L-BFGS-B', options={'maxiter':2000})
beta_g, sigma_g = res_g.x[:-1], np.exp(res_g.x[-1])
ll_g = -res_g.fun
k_g = p+1
aic_g = -2*ll_g + 2*k_g

print(f"Gamma:        sigma={sigma_g:.4f}, logLik={ll_g:.2f}, AIC={aic_g:.1f}")

# ============================================================
# 4. Log-normal EM (already have from item 1)
# ============================================================
beta_ln, sigma_ln, ll_ln, aic_ln = em_lognormal(Xf, yo, c, yl)
print(f"Log-normal:   sigma={sigma_ln:.4f}, logLik={ll_ln:.2f}, AIC={aic_ln:.1f}")

# ============================================================
# 5. Comparison
# ============================================================
print(f"\n{'='*50}")
print(f"Model         logLik      AIC     ΔAIC(vs LN)")
print(f"{'='*50}")
baseline = aic_ln
for name, aic, ll in [("Log-normal", aic_ln, ll_ln), ("Log-logistic", aic_ll, ll_ll), ("Gamma", aic_g, ll_g)]:
    delta = aic - baseline
    print(f"{name:14s} {ll:8.2f}  {aic:8.1f}  {delta:+8.1f}")

# ============================================================
# 6. Variance decomposition under each distribution
# ============================================================
print(f"\n{'='*50}")
print("Variance decomposition under each AFT distribution")
print(f"{'='*50}")

# For log-normal: use EM-based decomposition (already have numbers)
# For log-logistic: approximate by drop-one MLE (expensive but doable)
# For Gamma: same

# Quick approach: compute drop-one logLik for log-logistic
print("\nLog-logistic drop-one decomposition:")
ll_drops = {}
for drop in F_ORDER:
    rem = [f for f in F_ORDER if f!=drop]
    Xd = build_X(E)
    # rebuild with fewer columns
    cols = []
    for fn in rem:
        lv = FACTORS[fn]
        for j in range(1,len(lv)):
            col = np.array([1.0 if e[fn]==lv[j] else 0.0 for e in E])
            cols.append(col - 1.0/len(lv))
    Xd = np.column_stack(cols) if cols else np.zeros((n,0))
    Xd = np.column_stack([np.ones(n), Xd])

    # Common-scale treatment: fix the log-logistic scale at the full-model
    # value (sigma_ll) and optimize the location coefficients only, matching
    # the common-sigma likelihood-ratio decomposition used for log-normal.
    def nll_ll_fixed_scale(beta_only):
        params = np.append(beta_only, np.log(sigma_ll))
        return nll_loglogistic(params, Xd, yo, c, yl)
    res = minimize(nll_ll_fixed_scale, beta0[:Xd.shape[1]],
                   method='L-BFGS-B', options={'maxiter':1000})
    ll_d = -res.fun
    delta = ll_ll - ll_d
    ll_drops[drop] = delta
    print(f"  Drop {drop:30s}: ΔlogLik={delta:7.2f}")

# Total positive delta
td_ll = sum(max(0,d) for d in ll_drops.values())
if td_ll <= 0: td_ll = 1.0
print(f"\n  Log-logistic variance fractions (%):")
for fn in F_ORDER:
    vp = 100*max(0,ll_drops[fn])/td_ll
    print(f"    {fn:30s}: {vp:5.1f}%")

# Log-normal drop-one (quick recompute using EM)
print("\nLog-normal drop-one decomposition (EM):")
ln_drops = {}
for drop in F_ORDER:
    rem = [f for f in F_ORDER if f!=drop]
    Xd = build_X(E)
    cols = []
    for fn in rem:
        lv = FACTORS[fn]
        for j in range(1,len(lv)):
            col = np.array([1.0 if e[fn]==lv[j] else 0.0 for e in E])
            cols.append(col - 1.0/len(lv))
    Xd = np.column_stack(cols) if cols else np.zeros((n,0))
    Xd = np.column_stack([np.ones(n), Xd])
    _, _, ll_d, _ = em_lognormal(Xd, yo, c, yl, sigma_fixed=sigma_ln)
    delta = ll_ln - ll_d
    ln_drops[drop] = delta
    print(f"  Drop {drop:30s}: ΔlogLik={delta:7.2f}")

td_ln = sum(max(0,d) for d in ln_drops.values())
if td_ln <= 0: td_ln = 1.0
print(f"\n  Log-normal variance fractions (%):")
for fn in F_ORDER:
    vp = 100*max(0,ln_drops[fn])/td_ln
    print(f"    {fn:30s}: {vp:5.1f}%")

# Compare
print(f"\n{'='*50}")
print("Cross-distribution comparison:")
print(f"{'Factor':30s} {'Log-normal':>10s} {'Log-logistic':>10s}")
print(f"{'-'*50}")
for fn in F_ORDER:
    v_ln = 100*max(0,ln_drops[fn])/td_ln
    v_ll = 100*max(0,ll_drops[fn])/td_ll
    print(f"{fn:30s} {v_ln:9.1f}% {v_ll:9.1f}%")

# Save
output = {
    "description": "Multi-distribution AFT comparison for 3-method baseline",
    "distributions": {
        "log_normal": {"sigma": round(float(sigma_ln),4), "logLik": round(float(ll_ln),2), "AIC": round(float(aic_ln),1)},
        "log_logistic": {"sigma": round(float(sigma_ll),4), "logLik": round(float(ll_ll),2), "AIC": round(float(aic_ll),1)},
        "gamma": {"sigma": round(float(sigma_g),4), "logLik": round(float(ll_g),2), "AIC": round(float(aic_g),1)},
    },
    "best_model": "log-normal" if aic_ln <= min(aic_ll, aic_g) else ("log-logistic" if aic_ll <= aic_g else "gamma"),
    "log_normal_decomposition": {fn: round(100*max(0,ln_drops[fn])/td_ln,1) for fn in F_ORDER},
    "log_logistic_decomposition": {fn: round(100*max(0,ll_drops[fn])/td_ll,1) for fn in F_ORDER},
}
with open("output/multi_aft_comparison.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved output/multi_aft_comparison.json")
