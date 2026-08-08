"""
multi_load_sensitivity.py — multi load-coefficient sensitivity (v6.0, Layer 2)
===============================================================================
Reviewer round-3: "only K_A is tested; K_V (dynamic) and K_Fb (face-load)
are missing; industrial off-load and variable-speed conditions absent."

This script re-runs the full 144-combination AFT decomposition under
joint load-coefficient scenarios. The total load multiplier applied to
the nominal stress is K = K_A * K_V * K_Fb, with:

  K_A  (application factor): 1.0 / 1.25 / 1.5
  K_V  (dynamic factor):     1.0 / 1.3 / 1.6
  K_Fb (face-load factor):   1.0 / 1.4

Six representative scenarios cover steady, moderate-shock, and
industrial misaligned/variable-speed operation.

All numbers in output/multi_load_sensitivity.json are produced here.
"""
import json, sys, os
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)

from gear_params import SN_CURVES

SCENARIOS = {
    "steady_baseline": {"K_A": 1.0, "K_V": 1.0, "K_Fb": 1.0, "label": "steady, baseline (K=1.0)"},
    "moderate_dynamic": {"K_A": 1.0, "K_V": 1.3, "K_Fb": 1.0, "label": "moderate dynamic, K_V=1.3"},
    "heavy_dynamic": {"K_A": 1.0, "K_V": 1.6, "K_Fb": 1.0, "label": "heavy dynamic, K_V=1.6"},
    "misaligned_face": {"K_A": 1.0, "K_V": 1.0, "K_Fb": 1.4, "label": "face misalignment, K_Fb=1.4"},
    "shock_plus_dynamic": {"K_A": 1.25, "K_V": 1.3, "K_Fb": 1.0, "label": "shock + dynamic (K=1.625)"},
    "industrial_combined": {"K_A": 1.25, "K_V": 1.3, "K_Fb": 1.4, "label": "industrial combined (K=2.275)"},
}


def recompute_nf_with_k(entry, k_total):
    """Scale effective stress by k_total; recompute Nf via Basquin."""
    sn = entry["sn_source"]
    sigma_eff_orig = entry.get("sigma_effective", 0)
    sigma_eff_new = sigma_eff_orig * k_total
    fatigue_strength = SN_CURVES[sn]["fatigue_strength_at_1e6"]
    if sigma_eff_new <= fatigue_strength:
        return "inf", None
    sigma_f_prime = SN_CURVES[sn]["sigma_f_prime"]
    b = SN_CURVES[sn]["b"]
    nf_new = 0.5 * (sigma_eff_new / sigma_f_prime) ** (1.0 / b)
    log_nf_new = np.log10(nf_new)
    return float(nf_new), float(log_nf_new)


def build_scenario_sweep(sweep_data, k_total):
    out = []
    for entry in sweep_data:
        new_entry = dict(entry)
        new_nf, new_log = recompute_nf_with_k(entry, k_total)
        new_entry["Nf"] = new_nf
        new_entry["log10_Nf"] = new_log
        out.append(new_entry)
    return out


# ---- AFT decomposition ----
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

scen_out = {}
for key, cfg in SCENARIOS.items():
    k = cfg["K_A"] * cfg["K_V"] * cfg["K_Fb"]
    print(f"Scenario {key} (K={k:.3f}) ...", flush=True)
    sweep_s = build_scenario_sweep(sweep_orig, k)
    res = run_aft(entries_from_sweep(sweep_s))
    if "error" in res:
        scen_out[key] = {"label": cfg["label"], "K_total": round(k, 3), "result": res,
                         "note": "AFT not identified"}
        print(f"  {res.get('error')}", flush=True)
        continue
    scen_out[key] = {"label": cfg["label"], "K_total": round(k, 3), "result": res}
    vf = res["variance_fractions"]
    print(f"  std {vf.get('size_surface_standard'):.1f}%  rz {vf.get('rz_level'):.1f}%  "
          f"ms {vf.get('mean_stress_method'):.1f}%", flush=True)

out = {
    "description": "Multi load-coefficient sensitivity: joint K_A x K_V x K_Fb scenarios "
                   "over the full 144-combination AFT decomposition.",
    "scenarios": scen_out,
    "summary": "Factor ranking and variance shares under six load-coefficient scenarios; "
               "see per-scenario variance_fractions.",
}
with open("output/multi_load_sensitivity.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\nSaved output/multi_load_sensitivity.json")
print("=== Complete ===")
