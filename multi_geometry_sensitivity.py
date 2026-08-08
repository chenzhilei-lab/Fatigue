"""
multi_geometry_sensitivity.py — multi-geometry sensitivity (v6.0, Layer 2)
===========================================================================
Reviewer round-3: "single geometry (m_n=3, z_1=20); no multi-geometry
simulation; ranking is only valid for this specific gear."

This script re-runs the full 144-combination sweep + AFT decomposition
for two additional gear geometries:

  - baseline:  m_n = 3 mm,  z_1 = 20   (the paper's gear)
  - large module: m_n = 8 mm, z_1 = 20 (ISO size factor Y_X departs from 1)
  - low tooth count: m_n = 3 mm, z_1 = 14 (AGMA J-factor shifts substantially)

For each geometry the fatigue life pipeline is re-evaluated (module and
tooth count enter the nominal stress, the ISO Y_X size factor, the AGMA
Ks/J factors, and the FKM Kd factor), and the factor ranking is compared.

All numbers in output/multi_geometry_sensitivity.json are produced here.
"""
import json, sys, os, itertools
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)

import gear_params as gp
from fatigue_models import compute_fatigue_life

GEOMETRIES = {
    "baseline_mn3_z20": {"module_mn": 3.0, "teeth_pinion": 20, "label": "m_n=3 mm, z=20 (baseline)"},
    "large_module_mn8_z20": {"module_mn": 8.0, "teeth_pinion": 20, "label": "m_n=8 mm, z=20 (ISO Y_X != 1)"},
    "mid_module_mn5_z20": {"module_mn": 5.0, "teeth_pinion": 20, "label": "m_n=5 mm, z=20 (intermediate)"},
    "low_teeth_mn3_z14": {"module_mn": 3.0, "teeth_pinion": 14, "label": "m_n=3 mm, z=14 (AGMA J shifts)"},
}

SN_SOURCES = ["Waterloo_SMDIdbase_Iter068", "Waterloo_SMDIdbase_Iter066"]
MEAN_STRESS_METHODS = ["Goodman", "Gerber", "Morrow", "SWT"]
DAMAGE_METHODS = ["Miner_original", "Miner_modified"]
SIZE_SURFACE_STANDARDS = ["ISO_6336", "AGMA_2001", "FKM"]
RZ_KEYS = list(gp.RZ_LEVELS.keys())

combinations = list(itertools.product(
    SN_SOURCES, MEAN_STRESS_METHODS, DAMAGE_METHODS, SIZE_SURFACE_STANDARDS, RZ_KEYS))

TORQUE_NM = 700.0


def run_sweep_for_geometry(module, teeth):
    """Patch gear_params GEAR/LOAD for the given geometry and run 144 combos."""
    gp.GEAR["module_mn"] = module
    gp.GEAR["teeth_pinion"] = teeth
    gp.GEAR["teeth_gear"] = teeth * 3
    gp.GEAR["pitch_diameter_pinion"] = module * teeth
    gp.GEAR["pitch_diameter_gear"] = module * teeth * 3
    gp.GEAR["center_distance"] = 0.5 * module * teeth * 4
    gp.LOAD["torque_pinion"] = TORQUE_NM
    gp.LOAD["tangential_force"] = (2 * TORQUE_NM * 1000.0) / (module * teeth)
    gp.LOAD["Ft_per_unit_facewidth"] = gp.LOAD["tangential_force"] / gp.GEAR["face_width"]

    results = []
    for sn, ms, dm, ss, rz_key in combinations:
        rz_um = gp.RZ_LEVELS[rz_key]["Rz_um"]
        Nf, diag = compute_fatigue_life(
            sn_source=sn, kf_method="Peterson",
            mean_stress_method=ms, damage_method=dm,
            size_surface_standard=ss, rz_um=rz_um, verbose=False)
        results.append({
            "sn_source": sn, "mean_stress_method": ms, "damage_method": dm,
            "size_surface_standard": ss, "rz_level": rz_key, "rz_um": rz_um,
            "Nf": float(Nf) if np.isfinite(Nf) else "inf",
            "log10_Nf": float(np.log10(Nf)) if (np.isfinite(Nf) and Nf > 0) else None,
        })
    return results


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
        "variance_fractions": {k: round(v, 1) for k, v in vf.items()},
    }


# ============================================================
# Run
# ============================================================
geo_out = {}
for key, cfg in GEOMETRIES.items():
    print(f"Geometry {key} ...", flush=True)
    sweep = run_sweep_for_geometry(cfg["module_mn"], cfg["teeth_pinion"])
    res = run_aft(entries_from_sweep(sweep))
    if "error" in res:
        geo_out[key] = {"label": cfg["label"], "result": res,
                        "note": "AFT decomposition not identified (likely all run-out)"}
        print(f"  {res.get('error')} (n_censored={res.get('n_censored')})", flush=True)
        continue
    geo_out[key] = {"label": cfg["label"], "result": res}
    vf = res["variance_fractions"]
    print(f"  std {vf.get('size_surface_standard'):.1f}%  rz {vf.get('rz_level'):.1f}%  "
          f"ms {vf.get('mean_stress_method'):.1f}%  sn {vf.get('sn_source'):.1f}%  "
          f"dam {vf.get('damage_method'):.1f}%  int {vf.get('interaction'):.1f}%", flush=True)

# restore baseline geometry
gp.GEAR["module_mn"] = 3.0
gp.GEAR["teeth_pinion"] = 20
gp.GEAR["teeth_gear"] = 60

out = {
    "description": "Multi-geometry sensitivity: full 144-combination sweep + AFT "
                   "decomposition for m_n=3/z=20 (baseline), m_n=8/z=20, m_n=3/z=14.",
    "geometries": geo_out,
    "summary": "Factor ranking and variance shares for three gear geometries; "
               "see per-geometry variance_fractions.",
}
with open("output/multi_geometry_sensitivity.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\nSaved output/multi_geometry_sensitivity.json")
print("=== Complete ===")
