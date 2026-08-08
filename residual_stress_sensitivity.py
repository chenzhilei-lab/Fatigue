"""
residual_stress_sensitivity.py — residual stress sensitivity (v6.0, Layer 2)
=============================================================================
Reviewer round-3: "residual stress completely ignored; no quantitative
sensitivity simulation; 18.2% mean-stress share is an ideal zero-residual
baseline."

This script re-runs the full 144-combination AFT decomposition under
three manufacturing residual-stress states applied to the mean stress:

  - ground:      sigma_res = -300 MPa (compressive, reduces effective mean)
  - machined:    sigma_res =   0 MPa (near zero; reference baseline)
  - as-forged:   sigma_res = +150 MPa (tensile, raises effective mean)

Mechanism: sigma_m' = sigma_m + sigma_res. The mean-stress correction
converts (sigma_a, sigma_m') to an equivalent fully-reversed amplitude
sigma_ar', and the life is recomputed on the Basquin curve. The
resulting variance fractions bracket how the 18.2% mean-stress share
moves under realistic manufacturing stress states.

All numbers in output/residual_stress_sensitivity.json are produced here.
"""
import json, sys, os
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)

from gear_params import SN_CURVES

RESIDUAL_STATES = {
    "ground_-300MPa": {"sigma_res": -300.0, "label": "ground (-300 MPa compressive)"},
    "machined_0MPa":  {"sigma_res": 0.0,    "label": "machined (0 MPa, baseline)"},
    "as_forged_+150MPa": {"sigma_res": 150.0, "label": "as-forged (+150 MPa tensile)"},
}


def mean_stress_correction_ratio(sigma_max, R, method, sn_source, sigma_res):
    """Ratio sigma_ar(sigma_res)/sigma_ar(0) for the given correction method."""
    sigma_a = sigma_max * (1.0 - R) / 2.0
    sigma_m = sigma_max * (1.0 + R) / 2.0
    uts = SN_CURVES[sn_source]["uts"]
    sfp = SN_CURVES[sn_source]["sigma_f_prime"]

    def sar(sm):
        if method == "Goodman":
            return sigma_a / max(1.0 - sm / uts, 0.01)
        if method == "Gerber":
            return sigma_a / max(1.0 - (sm / uts) ** 2, 0.01)
        if method == "Morrow":
            return sigma_a / max(1.0 - sm / sfp, 0.01)
        if method == "SWT":
            return np.sqrt(sigma_max * sigma_a)
        raise ValueError(method)

    base = sar(sigma_m)
    shifted = sar(sigma_m + sigma_res)
    if base <= 0:
        return 1.0
    return max(float(shifted / base), 1e-3)


def recompute_nf(entry, ratio, sn):
    log_nf = entry.get("log10_Nf")
    if log_nf is None:
        return "inf", None
    b = SN_CURVES[sn]["b"]
    log_nf_new = log_nf + np.log10(ratio) / b
    nf_new = 10.0 ** log_nf_new
    sigma_eff_orig = entry.get("sigma_effective", 0)
    sigma_eff_new = sigma_eff_orig * ratio
    fatigue_strength = SN_CURVES[sn]["fatigue_strength_at_1e6"]
    if sigma_eff_new <= fatigue_strength:
        return "inf", None
    nf_orig = entry.get("Nf", float("inf"))
    if isinstance(nf_orig, str) or (isinstance(nf_orig, float) and not np.isfinite(nf_orig)):
        return "inf", None
    return float(nf_new), float(log_nf_new)


def build_state_sweep(sweep_data, sigma_res):
    out = []
    for entry in sweep_data:
        new_entry = dict(entry)
        sn = entry["sn_source"]
        method = entry["mean_stress_method"]
        sigma_max = entry.get("sigma_nominal", 0)  # peak stress incl. mean
        if sigma_max <= 0:
            ratio = 1.0
        else:
            ratio = mean_stress_correction_ratio(sigma_max, 0.0, method, sn, sigma_res)
        new_nf, new_log = recompute_nf(entry, ratio, sn)
        new_entry["Nf"] = new_nf
        new_entry["log10_Nf"] = new_log
        out.append(new_entry)
    return out


# ---- AFT decomposition (same as native_chain_analysis.py) ----
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


from scipy.stats import norm


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
        dr[drop] = llf - ll
    ls, lr = FACTORS["size_surface_standard"], FACTORS["rz_level"]
    xic = []
    for si in range(1, len(ls)):
        for rj in range(1, len(lr)):
            col = np.array([1.0 if e["size_surface_standard"] == ls[si] and e["rz_level"] == lr[rj]
                            else 0.0 for e in entries])
            xic.append(col)
    Xi = np.column_stack(xic) if xic else np.zeros((ntot, 0))
    Xi = Xi - Xi.mean(axis=0)
    _, _, lli = em(np.column_stack([Xf, Xi]), y, c, yl, sigma_fixed=sf)
    di = lli - llf
    td = sum(max(0, d) for d in dr.values()) + max(0, di)
    if td <= 0:
        td = 1.0
    vf = {}
    for fn in F_ORDER:
        vf[fn] = 100.0 * max(0, dr[fn]) / td
    vf["interaction"] = 100.0 * max(0, di) / td
    s = sum(vf.values())
    if s > 0:
        vf = {k: 100.0 * v / s for k, v in vf.items()}
    return {
        "n_total": ntot, "n_censored": ncen,
        "sigma": round(float(sf), 4),
        "logLik": round(float(llf), 2),
        "variance_fractions": {k: round(v, 1) for k, v in vf.items()},
    }


# ============================================================
# Run
# ============================================================
with open("output/sweep.json", encoding="utf-8") as f:
    sweep_orig = json.load(f)

print("Baseline (0 MPa) ...", flush=True)
base = run_aft(entries_from_sweep(sweep_orig))

states_out = {}
for key, cfg in RESIDUAL_STATES.items():
    print(f"State {key} ...", flush=True)
    sweep_s = build_state_sweep(sweep_orig, cfg["sigma_res"])
    res = run_aft(entries_from_sweep(sweep_s))
    states_out[key] = {
        "sigma_res_MPa": cfg["sigma_res"],
        "label": cfg["label"],
        "result": res,
        "mean_stress_var_pct": res["variance_fractions"].get("mean_stress_method"),
    }
    print(f"  mean-stress share: {res['variance_fractions'].get('mean_stress_method'):.1f}%",
          flush=True)

mean_vars = [v["mean_stress_var_pct"] for v in states_out.values()]
out = {
    "description": "Residual stress sensitivity: full 144-combination AFT decomposition "
                   "under ground/machined/as-forged residual stress states applied to mean stress.",
    "baseline_0MPa": base,
    "states": states_out,
    "mean_stress_share_range_pct": [min(mean_vars), max(mean_vars)],
    "summary": (
        f"Mean-stress variance share: 0 MPa baseline {base['variance_fractions']['mean_stress_method']:.1f}%; "
        f"ground -300 MPa {states_out['ground_-300MPa']['mean_stress_var_pct']:.1f}%; "
        f"as-forged +150 MPa {states_out['as_forged_+150MPa']['mean_stress_var_pct']:.1f}%. "
        f"Range across manufacturing states: {min(mean_vars):.1f}--{max(mean_vars):.1f} pp."
    ),
}
with open("output/residual_stress_sensitivity.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\nSaved output/residual_stress_sensitivity.json")
print("=== Complete ===")
