"""
native_chain_analysis.py — v6.0 native three-standard chain (main result)
=========================================================================
Reviewer round-3 (0808-审稿人意见03) requirement: the uniform-Y_Sa baseline
was criticised as artificially decoupling the three standards' native stress
calculation chains. This script re-runs the full 144-combination AFT
decomposition with each standard's native stress-concentration treatment:

  - ISO 6336: Y_Sa = 1.55 (native, unchanged from baseline)
  - AGMA 2001: J-factor geometry factor (J = 0.31 / 0.33 / 0.35 per
    AGMA 2001-D04 Fig. 8), applied as a stress multiplier relative to
    the unified Y_Sa
  - FKM: K_f via Neuber/Peterson notch sensitivity (K_f = 1.84 Peterson /
    1.75 Neuber for this gear), applied as a stress multiplier

The native-chain result is the PRIMARY decomposition of v6.0; the
uniform-Y_Sa baseline is retained as a sensitivity auxiliary.

All numbers in output/native_chain.json are produced by this script.
"""
import json, sys, os, copy
import numpy as np
from scipy.stats import norm

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from gear_params import SN_CURVES, GEAR, MATERIAL, LOAD

# ============================================================
# 1. Native stress multipliers per standard
# ============================================================
Y_Sa_unified = 1.55

# AGMA J-factor for 20-tooth pinion / 60-tooth gear, 20 deg PA
# Source: AGMA 2001-D04 Figure 8 / ANSI-AGMA 908-B89 (J ~ 0.31-0.35)
# Effective stress multiplier relative to unified Y_Sa = 1.55:
#   AGMA multiplier = (1/J) / (Y_Fa * Y_eps)  vs  ISO multiplier = Y_Sa
AGMA_J_range = {
    "J_low_conservative":    {"J": 0.31, "ratio_to_unified": 1.67/1.55},
    "J_midpoint":            {"J": 0.33, "ratio_to_unified": 1.57/1.55},
    "J_high_nonconservative":{"J": 0.35, "ratio_to_unified": 1.48/1.55},
}

# FKM K_f via Neuber/Peterson (same computation as native_stress_sensitivity.py)
mn = GEAR["module_mn"]
t = np.pi * mn / 2
r_root = GEAR["tip_radius"] * mn
h = 2.25 * mn
K_t_gear = 1.0 + 0.5 * np.sqrt(t / max(r_root, 0.1)) * (t / h) ** 0.3

uts = MATERIAL["uts"]
a_peterson = 0.0254 * (2079.0 / uts) ** 1.8
K_f_peterson = 1.0 + (K_t_gear - 1.0) / (1.0 + a_peterson / max(r_root, 0.01))
a_neuber = 0.01524 * (2079.0 / uts) ** 1.8
K_f_neuber = 1.0 + (K_t_gear - 1.0) / (1.0 + np.sqrt(a_neuber / max(r_root, 0.01)))

FKM_Kf_range = {
    "Kf_Peterson": {"K_f": K_f_peterson, "ratio_to_unified": K_f_peterson / Y_Sa_unified},
    "Kf_Neuber":   {"K_f": K_f_neuber,   "ratio_to_unified": K_f_neuber / Y_Sa_unified},
}

# ============================================================
# 2. Load sweep, recompute Nf under native stress chains
# ============================================================
with open("output/sweep.json", encoding="utf-8") as f:
    sweep_orig = json.load(f)


def recompute_nf_with_stress_ratio(entry, stress_ratio):
    """Recompute Nf with effective stress scaled by stress_ratio."""
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
    """Rebuild the 144-combination sweep with native stress chains."""
    out = []
    for entry in sweep_data:
        new_entry = dict(entry)
        std = entry["size_surface_standard"]
        ratio = 1.0
        if std == "AGMA_2001":
            ratio = agma_ratio
        elif std == "FKM":
            ratio = fkm_ratio
        # ISO keeps Y_Sa native -> ratio 1.0
        new_nf, new_log = recompute_nf_with_stress_ratio(entry, ratio)
        new_entry["Nf"] = new_nf
        new_entry["log10_Nf"] = new_log
        out.append(new_entry)
    return out


# ============================================================
# 3. AFT decomposition (reused from native_stress_sensitivity.py)
# ============================================================
FACTORS = {
    "mean_stress_method": ["Goodman", "Gerber", "Morrow", "SWT"],
    "size_surface_standard": ["ISO_6336", "AGMA_2001", "FKM"],
    "rz_level": ["ground_Rz4", "machined_Rz15", "as_forged_Rz50"],
    "sn_source": ["Waterloo_SMDIdbase_Iter068", "Waterloo_SMDIdbase_Iter066"],
    "damage_method": ["Miner_original", "Miner_modified"],
}
F_ORDER = ["mean_stress_method", "size_surface_standard", "rz_level", "sn_source", "damage_method"]


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


def build_design(entries, fs=None):
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


def extract_arrays(entries):
    y = np.array([e.get("logNf", 0) or 0 for e in entries])
    c = np.array([e["censored"] == 1 for e in entries])
    yl = np.array([e.get("logNf_lower", 0) or 0 for e in entries])
    return y, c, yl


def em(X, yo, cen, yl, n_iter=30, sigma_fixed=None):
    n, p = X.shape
    om = ~cen
    if om.sum() < p + 2:
        return None, None, float('-inf')
    beta = np.linalg.lstsq(X[om], yo[om], rcond=None)[0]
    sigma = (sigma_fixed if sigma_fixed is not None
             else max(np.std(yo[om] - X[om] @ beta, ddof=p), 0.001))
    ya = yo.copy()
    for _ in range(n_iter):
        mu = X @ beta
        bo, so = beta.copy(), sigma
        if cen.any():
            z = (yl[cen] - mu[cen]) / sigma
            imr = np.where(z > 30, z, np.where(z < -30, 0.0, np.exp(norm.logpdf(z) - norm.logsf(z))))
            ya[cen] = mu[cen] + sigma * imr
        beta = np.linalg.lstsq(X, ya, rcond=None)[0]
        r = ya - X @ beta
        if sigma_fixed is None:
            s2 = np.sum(r[om] ** 2) / max(om.sum(), 1)
            if cen.any():
                z = (yl[cen] - X[cen] @ beta) / so
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
    if cen.any():
        ll += np.sum(norm.logsf((yl[cen] - mu[cen]) / sigma))
    return beta, sigma, ll


def run_aft(entries):
    ntot = len(entries)
    ncen = sum(1 for e in entries if e["censored"] == 1)
    Xf = build_design(entries)
    y, c, yl = extract_arrays(entries)
    bf, sf, llf = em(Xf, y, c, yl)
    if bf is None:
        return {"error": "insufficient observed", "n_censored": ncen}
    dr = {}
    for drop in F_ORDER:
        rem = [f for f in F_ORDER if f != drop]
        Xd = build_design(entries, rem)
        _, _, ll = em(Xd, y, c, yl, sigma_fixed=sf)
        dr[drop] = {"dLL": llf - ll, "df": len(FACTORS[drop]) - 1}
    ls, lr = FACTORS["size_surface_standard"], FACTORS["rz_level"]
    xic = []
    for si in range(1, len(ls)):
        for rj in range(1, len(lr)):
            col = np.array([1.0 if e["size_surface_standard"] == ls[si] and e["rz_level"] == lr[rj] else 0.0
                            for e in entries])
            xic.append(col)
    Xi = np.column_stack(xic) if xic else np.zeros((ntot, 0))
    Xi = Xi - Xi.mean(axis=0)
    _, _, lli = em(np.column_stack([Xf, Xi]), y, c, yl, sigma_fixed=sf)
    di = lli - llf
    td = sum(max(0, d["dLL"]) for d in dr.values()) + max(0, di)
    if td <= 0:
        td = 1.0
    vf = {}
    for fn in F_ORDER:
        vf[fn] = 100.0 * max(0, dr[fn]["dLL"]) / td
    vf["interaction"] = 100.0 * max(0, di) / td
    # normalise to 100
    s = sum(vf.values())
    if s > 0:
        vf = {k: 100.0 * v / s for k, v in vf.items()}
    return {
        "n_total": ntot, "n_censored": ncen,
        "sigma": round(float(sf), 4),
        "logLik": round(float(llf), 2),
        "variance_fractions": {k: round(v, 1) for k, v in vf.items()},
        "dLL": {k: round(v["dLL"], 1) for k, v in dr.items()},
    }


# ============================================================
# 4. Run: baseline (uniform Y_Sa) + native chain scenarios
# ============================================================
print("Step 1/4: baseline (uniform Y_Sa, 144 combos) ...", flush=True)
entries_base = entries_from_sweep(sweep_orig)
base = run_aft(entries_base)

print("Step 2/4: AGMA native J (midpoint) only ...", flush=True)
sweep_agma = build_native_sweep(sweep_orig, AGMA_J_range["J_midpoint"]["ratio_to_unified"], 1.0)
res_agma = run_aft(entries_from_sweep(sweep_agma))

print("Step 3/4: FKM native K_f (Peterson) only ...", flush=True)
sweep_fkm = build_native_sweep(sweep_orig, 1.0, FKM_Kf_range["Kf_Peterson"]["ratio_to_unified"])
res_fkm = run_aft(entries_from_sweep(sweep_fkm))

print("Step 4/4: combined native chain (AGMA J midpoint + FKM K_f Peterson) ...", flush=True)
sweep_native = build_native_sweep(sweep_orig, AGMA_J_range["J_midpoint"]["ratio_to_unified"],
                                  FKM_Kf_range["Kf_Peterson"]["ratio_to_unified"])
res_native = run_aft(entries_from_sweep(sweep_native))

# Sensitivity: AGMA J low/high, FKM Neuber
print("Sensitivity: AGMA J low / high, FKM Neuber ...", flush=True)
sweep_jlow = build_native_sweep(sweep_orig, AGMA_J_range["J_low_conservative"]["ratio_to_unified"],
                                FKM_Kf_range["Kf_Peterson"]["ratio_to_unified"])
res_jlow = run_aft(entries_from_sweep(sweep_jlow))
sweep_jhigh = build_native_sweep(sweep_orig, AGMA_J_range["J_high_nonconservative"]["ratio_to_unified"],
                                 FKM_Kf_range["Kf_Peterson"]["ratio_to_unified"])
res_jhigh = run_aft(entries_from_sweep(sweep_jhigh))
sweep_neuber = build_native_sweep(sweep_orig, AGMA_J_range["J_midpoint"]["ratio_to_unified"],
                                  FKM_Kf_range["Kf_Neuber"]["ratio_to_unified"])
res_neuber = run_aft(entries_from_sweep(sweep_neuber))

# ============================================================
# 5. Summary output
# ============================================================
def frac(res):
    return res.get("variance_fractions", {})


std_base = frac(base).get("size_surface_standard")
std_native = frac(res_native).get("size_surface_standard")
std_agma = frac(res_agma).get("size_surface_standard")
std_fkm = frac(res_fkm).get("size_surface_standard")

output = {
    "description": "Native three-standard chain AFT decomposition (v6.0 main result) — "
                   "AGMA J-factor and FKM K_f replace the unified Y_Sa for those standards; ISO keeps native Y_Sa",
    "gear": {"module_mn": GEAR["module_mn"], "teeth": GEAR["teeth_pinion"],
             "torque_Nm": LOAD["torque_pinion"], "R": 0.0},
    "native_multipliers": {
        "iso_y_sa": Y_Sa_unified,
        "agma_j_range": {k: {"J": v["J"], "ratio_to_unified": round(v["ratio_to_unified"], 4)}
                         for k, v in AGMA_J_range.items()},
        "fkm_kf_range": {k: {"K_f": round(v["K_f"], 2), "ratio_to_unified": round(v["ratio_to_unified"], 4)}
                         for k, v in FKM_Kf_range.items()},
    },
    "baseline_uniform_ysa": base,
    "native_agma_only": res_agma,
    "native_fkm_only": res_fkm,
    "native_chain_main": res_native,
    "sensitivity": {
        "agma_J_low": res_jlow,
        "agma_J_high": res_jhigh,
        "fkm_neuber": res_neuber,
    },
    "deltas_pp": {
        "native_vs_baseline_standard": round(std_native - std_base, 1),
        "agma_only_vs_baseline_standard": round(std_agma - std_base, 1),
        "fkm_only_vs_baseline_standard": round(std_fkm - std_base, 1),
    },
    "summary": (
        f"Uniform-Y_Sa baseline: standard variance {std_base:.1f}%. "
        f"Native three-standard chain (AGMA J=0.33 midpoint + FKM K_f(Peterson)={K_f_peterson:.2f}): "
        f"standard variance {std_native:.1f}% (Δ={std_native - std_base:+.1f}pp). "
        f"AGMA J sensitivity {frac(res_jlow).get('size_surface_standard'):.1f}--{frac(res_jhigh).get('size_surface_standard'):.1f}%; "
        f"FKM Neuber alternative {frac(res_neuber).get('size_surface_standard'):.1f}%."
    ),
}

with open("output/native_chain.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("\n=== NATIVE CHAIN MAIN RESULT ===")
print(f"Baseline (uniform Y_Sa):           standard = {std_base:.1f}%")
print(f"Native chain (J mid + Kf Peters):  standard = {std_native:.1f}% (Δ={std_native - std_base:+.1f}pp)")
for fn in F_ORDER:
    print(f"  {fn:30s}: {frac(res_native).get(fn):5.1f}%")
print(f"  interaction                    : {frac(res_native).get('interaction'):5.1f}%")
print("=== SENSITIVITY ===")
print(f"AGMA J low  (J=0.31): {frac(res_jlow).get('size_surface_standard'):.1f}%")
print(f"AGMA J high (J=0.35): {frac(res_jhigh).get('size_surface_standard'):.1f}%")
print(f"FKM Neuber:           {frac(res_neuber).get('size_surface_standard'):.1f}%")
print("\nSaved output/native_chain.json")
print("=== Complete ===")
