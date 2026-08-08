"""
rzra_range_sensitivity.py — Rz/Ra conversion ratio sweep (v6.0, Layer 2)
=========================================================================
Reviewer round-3: "Rz/Ra = 1/6 fixed conversion is a simplification with
no quantitative calibration; traverse the 4-20 range and redo the
decomposition."

AGMA 2001-D04 Clause 16 expresses the surface condition factor C_f in
terms of R_a (microinches); ISO and FKM use R_z. This script sweeps the
Rz/Ra conversion ratio over the process-dependent range
{4, 6, 10, 20} and re-runs the full 144-combination AFT decomposition,
quantifying how the 22.6% roughness share moves.

All numbers in output/rzra_range_sensitivity.json are produced here.
"""
import json, sys, os
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)

from gear_params import SN_CURVES

RATIOS = [4, 6, 10, 20]


def agma_cf(Rz_um, ratio, sn_source):
    """AGMA 2001-D04 Clause 16 C_f with Rz/Ra = ratio."""
    if Rz_um <= 8:
        return 1.0  # ground gears
    Ra_uin = (Rz_um / ratio) * 39.37
    uts_ksi = SN_CURVES[sn_source]["uts"] / 6.895
    cf = 1.0 - 0.0058 * np.log(Ra_uin) * np.log(uts_ksi)
    return max(float(cf), 0.7)


def recompute_nf(entry, ratio):
    """Recompute Nf with AGMA C_f re-evaluated at the given Rz/Ra ratio."""
    sn = entry["sn_source"]
    std = entry["size_surface_standard"]
    log_nf = entry.get("log10_Nf")
    if std != "AGMA_2001" or log_nf is None:
        return entry.get("Nf", "inf"), log_nf
    Rz_um = entry.get("rz_um", 0)
    cf_old = agma_cf(Rz_um, 6.0, sn)      # baseline ratio = 6
    cf_new = agma_cf(Rz_um, ratio, sn)
    # stress multiplier ratio: sigma_eff ~ 1/Cf
    stress_ratio = cf_old / max(cf_new, 1e-3)
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


def build_ratio_sweep(sweep_data, ratio):
    out = []
    for entry in sweep_data:
        new_entry = dict(entry)
        new_nf, new_log = recompute_nf(entry, ratio)
        new_entry["Nf"] = new_nf
        new_entry["log10_Nf"] = new_log
        out.append(new_entry)
    return out


# ---- AFT decomposition (same core as other Layer-2 scripts) ----
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

print("Baseline (ratio=6) ...", flush=True)
base = run_aft(entries_from_sweep(sweep_orig))

ratios_out = {}
for ratio in RATIOS:
    print(f"Ratio Rz/Ra = {ratio} ...", flush=True)
    sweep_r = build_ratio_sweep(sweep_orig, ratio)
    res = run_aft(entries_from_sweep(sweep_r))
    ratios_out[str(ratio)] = {
        "ratio": ratio,
        "result": res,
        "rz_var_pct": res["variance_fractions"].get("rz_level"),
        "standard_var_pct": res["variance_fractions"].get("size_surface_standard"),
    }
    print(f"  Rz share: {res['variance_fractions'].get('rz_level'):.1f}%  "
          f"Std share: {res['variance_fractions'].get('size_surface_standard'):.1f}%",
          flush=True)

rz_vars = [v["rz_var_pct"] for v in ratios_out.values()]
out = {
    "description": "Rz/Ra conversion ratio sweep: full 144-combination AFT decomposition "
                   "with AGMA C_f re-evaluated at Rz/Ra = 4/6/10/20.",
    "baseline_ratio_6": base,
    "ratios": ratios_out,
    "rz_share_range_pct": [min(rz_vars), max(rz_vars)],
    "summary": (
        f"Roughness variance share: baseline (Rz/Ra=6) {base['variance_fractions']['rz_level']:.1f}%; "
        f"range across Rz/Ra = 4--20: {min(rz_vars):.1f}--{max(rz_vars):.1f}%. "
        f"Standard share range: {min(v['standard_var_pct'] for v in ratios_out.values()):.1f}--"
        f"{max(v['standard_var_pct'] for v in ratios_out.values()):.1f}%."
    ),
}
with open("output/rzra_range_sensitivity.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\nSaved output/rzra_range_sensitivity.json")
print("=== Complete ===")
