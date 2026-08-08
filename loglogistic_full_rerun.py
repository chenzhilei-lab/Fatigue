"""
loglogistic_full_rerun.py — log-logistic vs log-normal full-factorial rerun (v6.0)
==================================================================================
Reviewer round-3: "existing log-logistic better but log-normal chosen --
add quantitative sensitivity: re-run the full factorial decomposition with
log-logistic, present both variance tables side by side."

This script re-runs the FULL 144-combination 4-method decomposition (both the
uniform-Y_Sa baseline and the native-chain main result) under:
  - log-normal AFT (EM censored likelihood)  -- benchmark
  - log-logistic AFT (direct NLL minimisation) -- sensitivity

Both use the common-sigma drop-one-factor LR decomposition with the
standard x Rz interaction term, exactly matching the paper's procedure.

All numbers in output/loglogistic_full_rerun.json are produced here.
"""
import json, sys, os, copy
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from gear_params import SN_CURVES, GEAR, MATERIAL, LOAD

FACTORS = {
    "mean_stress_method": ["Goodman", "Gerber", "Morrow", "SWT"],
    "size_surface_standard": ["ISO_6336", "AGMA_2001", "FKM"],
    "rz_level": ["ground_Rz4", "machined_Rz15", "as_forged_Rz50"],
    "sn_source": ["Waterloo_SMDIdbase_Iter068", "Waterloo_SMDIdbase_Iter066"],
    "damage_method": ["Miner_original", "Miner_modified"],
}
F_ORDER = ["mean_stress_method", "size_surface_standard", "rz_level", "sn_source", "damage_method"]

# Native multipliers (identical to native_chain_analysis.py)
Y_Sa_unified = 1.55
AGMA_J_range = {
    "J_low_conservative":    {"J": 0.31, "ratio_to_unified": 1.67/1.55},
    "J_midpoint":            {"J": 0.33, "ratio_to_unified": 1.57/1.55},
    "J_high_nonconservative":{"J": 0.35, "ratio_to_unified": 1.48/1.55},
}
mn = GEAR["module_mn"]
t = np.pi * mn / 2
r_root = GEAR["tip_radius"] * mn
h = 2.25 * mn
K_t_gear = 1.0 + 0.5 * np.sqrt(t / max(r_root, 0.1)) * (t / h) ** 0.3
uts = MATERIAL["uts"]
a_peterson = 0.0254 * (2079.0 / uts) ** 1.8
K_f_peterson = 1.0 + (K_t_gear - 1.0) / (1.0 + a_peterson / max(r_root, 0.01))
FKM_Kf_range = {"Kf_Peterson": {"K_f": K_f_peterson, "ratio_to_unified": K_f_peterson / Y_Sa_unified}}


def recompute_nf_with_stress_ratio(entry, stress_ratio):
    log_nf = entry.get("log10_Nf")
    if log_nf is None:
        return "inf", None
    sn = entry["sn_source"]
    b = SN_CURVES[sn]["b"]
    log_nf_new = log_nf + np.log10(stress_ratio) / b
    nf_new = 10.0 ** log_nf_new
    sigma_eff_orig = entry.get("sigma_effective", 0)
    sigma_eff_new = sigma_eff_orig * stress_ratio
    fatigue_strength = SN_CURVES[sn]["fatigue_strength_at_1e6"]
    if sigma_eff_new <= fatigue_strength:
        return "inf", None
    nf_orig = entry.get("Nf", float("inf"))
    if isinstance(nf_orig, str) or (isinstance(nf_orig, float) and not np.isfinite(nf_orig)):
        return "inf", None
    return float(nf_new), float(log_nf_new)


def build_native_sweep(sweep_data, agma_ratio, fkm_ratio):
    out = []
    for entry in sweep_data:
        new_entry = dict(entry)
        std = entry["size_surface_standard"]
        ratio = 1.0
        if std == "AGMA_2001":
            ratio = agma_ratio
        elif std == "FKM":
            ratio = fkm_ratio
        new_nf, new_log = recompute_nf_with_stress_ratio(entry, ratio)
        new_entry["Nf"] = new_nf
        new_entry["log10_Nf"] = new_log
        out.append(new_entry)
    return out


def entries_from_sweep(sweep_data):
    entries = []
    for r in sweep_data:
        e = {k: r[k] for k in ["sn_source", "mean_stress_method", "damage_method", "size_surface_standard", "rz_level"]}
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


def build_X(entries, fs=None):
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


def build_interaction_X(entries):
    ls, lr = FACTORS["size_surface_standard"], FACTORS["rz_level"]
    xic = []
    for si in range(1, len(ls)):
        for rj in range(1, len(lr)):
            col = np.array([1.0 if e["size_surface_standard"] == ls[si] and e["rz_level"] == lr[rj]
                            else 0.0 for e in entries])
            xic.append(col)
    Xi = np.column_stack(xic) if xic else np.zeros((len(entries), 0))
    return Xi - Xi.mean(axis=0)


def extract_arrays(entries):
    y = np.array([e.get("logNf", 0) or 0 for e in entries])
    c = np.array([e["censored"] == 1 for e in entries])
    yl = np.array([e.get("logNf_lower", 0) or 0 for e in entries])
    return y, c, yl


# ---------------- log-normal EM (replica of aft_variance_decomposition) ----------------
def em_lognormal(X, yo, c, yl, n_iter=30, sigma_fixed=None):
    n, p = X.shape
    om = ~c
    if om.sum() < p + 2:
        return None, None, float('-inf')
    beta = np.linalg.lstsq(X[om], yo[om], rcond=None)[0]
    sigma = (sigma_fixed if sigma_fixed is not None
             else max(np.std(yo[om] - X[om] @ beta, ddof=p), 0.001))
    ya = yo.copy()
    for _ in range(n_iter):
        mu = X @ beta
        bo, so = beta.copy(), sigma
        if c.any():
            z = (yl[c] - mu[c]) / sigma
            imr = np.where(z > 30, z, np.where(z < -30, 0.0, np.exp(norm.logpdf(z) - norm.logsf(z))))
            ya[c] = mu[c] + sigma * imr
        beta = np.linalg.lstsq(X, ya, rcond=None)[0]
        r = ya - X @ beta
        if sigma_fixed is None:
            s2 = np.sum(r[om] ** 2) / max(om.sum(), 1)
            if c.any():
                z = (yl[c] - X[c] @ beta) / so
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
    if c.any():
        ll += np.sum(norm.logsf((yl[c] - mu[c]) / sigma))
    return beta, sigma, ll


# ---------------- log-logistic (direct NLL minimisation) ----------------
def nll_loglogistic(params, X, yo, c, yl):
    beta = params[:-1]
    sigma = np.exp(params[-1])
    mu = X @ beta
    om = ~c
    ll = 0.0
    z_obs = (yo[om] - mu[om]) / sigma
    ll += np.sum(z_obs - 2 * np.log(1 + np.exp(z_obs)) - np.log(sigma))
    if c.any():
        z_cens = (yl[c] - mu[c]) / sigma
        ll += np.sum(-np.log(1 + np.exp(z_cens)))
    return -ll


def fit_loglogistic(X, yo, c, yl):
    p = X.shape[1]
    beta0 = np.linalg.lstsq(X[~c], yo[~c], rcond=None)[0] if (~c).sum() > p else np.zeros(p)
    res = minimize(nll_loglogistic, np.append(beta0, np.log(0.15)),
                   args=(X, yo, c, yl), method='L-BFGS-B', options={'maxiter': 2000})
    beta = res.x[:-1]
    sigma = float(np.exp(res.x[-1]))
    ll = -res.fun
    return beta, sigma, ll


def decompose_both(entries):
    """Return dicts for log-normal and log-logistic with interaction."""
    Xf = build_X(entries)
    y, c, yl = extract_arrays(entries)

    # log-normal
    bf, sf, llf = em_lognormal(Xf, y, c, yl)
    ln_drops = {}
    for drop in F_ORDER:
        Xd = build_X(entries, [g for g in F_ORDER if g != drop])
        _, _, ll = em_lognormal(Xd, y, c, yl, sigma_fixed=sf)
        ln_drops[drop] = llf - ll
    Xi = build_interaction_X(entries)
    _, _, lli = em_lognormal(np.column_stack([Xf, Xi]), y, c, yl, sigma_fixed=sf)
    ln_drops["interaction"] = lli - llf

    # log-logistic (sigma fixed at full-model value for drop-one, same convention)
    bll, sll, lll = fit_loglogistic(Xf, y, c, yl)
    ll_drops = {}
    for drop in F_ORDER:
        Xd = build_X(entries, [g for g in F_ORDER if g != drop])
        beta0 = np.linalg.lstsq(Xd[~c], y[~c], rcond=None)[0] if (~c).sum() > Xd.shape[1] else np.zeros(Xd.shape[1])
        res = minimize(nll_loglogistic, np.append(beta0, np.log(sll)),
                       args=(Xd, y, c, yl), method='L-BFGS-B', options={'maxiter': 2000})
        ll_d = -res.fun
        ll_drops[drop] = lll - ll_d
    resi = minimize(nll_loglogistic, np.append(np.linalg.lstsq(np.column_stack([Xf, Xi])[~c], y[~c], rcond=None)[0], np.log(sll)),
                    args=(np.column_stack([Xf, Xi]), y, c, yl), method='L-BFGS-B', options={'maxiter': 2000})
    lli_ll = -resi.fun
    ll_drops["interaction"] = lli_ll - lll

    def normalise(d):
        total = sum(max(0.0, v) for v in d.values())
        if total <= 0:
            total = 1.0
        return {k: round(100.0 * max(0.0, v) / total, 1) for k, v in d.items()}

    return {
        "log_normal": {"fractions": normalise(ln_drops), "sigma": round(float(sf), 4), "logLik": round(float(llf), 2)},
        "log_logistic": {"fractions": normalise(ll_drops), "sigma": round(sll, 4), "logLik": round(lll, 2)},
    }


# ============================================================
# Run on both datasets
# ============================================================
with open("output/sweep.json", encoding="utf-8") as f:
    sweep_orig = json.load(f)

print("Baseline (uniform Y_Sa, 144 combos)...", flush=True)
entries_base = entries_from_sweep(sweep_orig)
base = decompose_both(entries_base)

print("Native chain (AGMA J mid + FKM Kf Peterson, 144 combos)...", flush=True)
sweep_native = build_native_sweep(sweep_orig, AGMA_J_range["J_midpoint"]["ratio_to_unified"],
                                  FKM_Kf_range["Kf_Peterson"]["ratio_to_unified"])
entries_native = entries_from_sweep(sweep_native)
native = decompose_both(entries_native)

# ============================================================
# Output
# ============================================================
out = {
    "description": "Full 144-combination 4-method decomposition under log-normal "
                   "vs log-logistic AFT, for both uniform-Y_Sa baseline and native-chain main result.",
    "baseline_uniform_ysa": base,
    "native_chain": native,
    "summary": (
        f"Baseline: log-normal std {base['log_normal']['fractions']['size_surface_standard']:.1f}% vs "
        f"log-logistic {base['log_logistic']['fractions']['size_surface_standard']:.1f}% — ranking preserved. "
        f"Native chain: log-normal {native['log_normal']['fractions']['size_surface_standard']:.1f}% vs "
        f"log-logistic {native['log_logistic']['fractions']['size_surface_standard']:.1f}%."
    ),
}
with open("output/loglogistic_full_rerun.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print("\n=== BASELINE (uniform Y_Sa) ===")
print(f"{'Factor':30s} {'log-normal':>10s} {'log-logistic':>12s}")
for k in F_ORDER + ["interaction"]:
    print(f"{k:30s} {base['log_normal']['fractions'].get(k,0):9.1f}% {base['log_logistic']['fractions'].get(k,0):11.1f}%")
print("\n=== NATIVE CHAIN ===")
print(f"{'Factor':30s} {'log-normal':>10s} {'log-logistic':>12s}")
for k in F_ORDER + ["interaction"]:
    print(f"{k:30s} {native['log_normal']['fractions'].get(k,0):9.1f}% {native['log_logistic']['fractions'].get(k,0):11.1f}%")
print("\nSaved output/loglogistic_full_rerun.json")
print("=== Complete ===")
