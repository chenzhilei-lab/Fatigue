"""
Y_Sa_sensitivity_full.py — Full-range Y_Sa perturbation (1.2–1.9)
=================================================================
Vary Y_Sa uniformly for ALL three standards, re-run AFT point estimates.
Tests: how much does the standard-choice variance change across the
plausible range of gear-tooth stress concentration factors?
Output: output/ysa_sensitivity_full.json
"""

import json, os, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from fatigue_models import (
    apply_mean_stress_correction, apply_size_surface_factor,
    compute_fatigue_life_sn, apply_damage_rule,
)
from gear_params import GEAR, LOAD, LOAD_SPECTRUM, RZ_LEVELS
from aft_variance_decomposition import (
    em_censored_normal, build_design, extract_arrays, FACTORS, FACTOR_ORDER,
)

# ============================================================
# 1. Load data
# ============================================================
with open("output/sweep.json") as f:
    sweep = json.load(f)

z = GEAR["teeth_pinion"]
Y_Sa_nominal = 1.58 - 0.0015 * z  # 1.55
sigma0 = sweep[0]["sigma_nominal"]
print(f"Y_Sa_nominal = {Y_Sa_nominal:.4f}, sigma0 = {sigma0:.1f} MPa")

# ============================================================
# 2. Recompute all entries with perturbed Y_Sa
# ============================================================
def recompute_all(ysa_new):
    """Recompute ALL 144 entries with new Y_Sa, keeping other params."""
    sigma_F0_new = sigma0 * (ysa_new / Y_Sa_nominal)
    R = LOAD_SPECTRUM["stress_ratio_R"]
    new_entries = []

    for entry in sweep:
        sigma_ar = apply_mean_stress_correction(
            sigma_F0_new, R, method=entry["mean_stress_method"],
            sn_source=entry["sn_source"]
        )
        sigma_eff = apply_size_surface_factor(
            sigma_ar, standard=entry["size_surface_standard"],
            Rz_um=RZ_LEVELS[entry["rz_level"]]["Rz_um"],
            sn_source=entry["sn_source"]
        )
        Nf_raw = compute_fatigue_life_sn(sigma_eff, sn_source=entry["sn_source"])
        Nf_final = apply_damage_rule(Nf_raw, method=entry["damage_method"])

        new_entry = {
            "sn_source": entry["sn_source"],
            "mean_stress_method": entry["mean_stress_method"],
            "damage_method": entry["damage_method"],
            "size_surface_standard": entry["size_surface_standard"],
            "rz_level": entry["rz_level"],
            "sigma_nominal": round(float(sigma_F0_new), 2),
            "sigma_effective": round(float(sigma_eff), 2),
        }
        if np.isfinite(Nf_final) and Nf_final > 0:
            new_entry["Nf"] = float(Nf_final)
            new_entry["log10_Nf"] = round(float(np.log10(Nf_final)), 6)
        else:
            new_entry["Nf"] = float("inf")
            new_entry["log10_Nf"] = None
        new_entries.append(new_entry)
    return new_entries


# ============================================================
# 3. AFT point estimates
# ============================================================
def build_entries_list(raw_entries):
    entries = []
    for e in raw_entries:
        entry = {k: e[k] for k in ["sn_source","mean_stress_method",
                                     "damage_method","size_surface_standard","rz_level"]}
        nf = e.get("Nf", float("inf"))
        log_nf = e.get("log10_Nf")
        if isinstance(nf,(int,float)) and np.isfinite(nf) and nf>0 and log_nf is not None:
            entry["logNf"]=float(log_nf); entry["censored"]=0
        else:
            entry["logNf"]=None; entry["censored"]=1
        entries.append(entry)
    MAX = max(e["logNf"] for e in entries if e["logNf"] is not None)
    for e in entries:
        if e["censored"]==1:
            e["logNf_lower"]=MAX
    return entries


def run_aft(entries):
    Y,C,L = extract_arrays(entries)
    Xf = build_design(entries, FACTOR_ORDER)
    bf,sf,llf = em_censored_normal(Xf,Y,C,L)

    drops = {}
    for drop_f in FACTOR_ORDER:
        rem = [f for f in FACTOR_ORDER if f!=drop_f]
        Xd = build_design(entries, rem)
        bd,sd,ld = em_censored_normal(Xd,Y,C,L)
        drops[drop_f] = llf - ld

    # Interaction
    ls = FACTORS["size_surface_standard"]
    lr = FACTORS["rz_level"]
    Xc=[]
    for si in range(1,len(ls)):
        for rj in range(1,len(lr)):
            Xc.append(np.array([1.0 if e["size_surface_standard"]==ls[si] and e["rz_level"]==lr[rj] else 0.0 for e in entries]))
    Xi=np.column_stack(Xc) if Xc else np.zeros((len(entries),0))
    Xi-=Xi.mean(axis=0); Xwi=np.column_stack([Xf,Xi])
    bi,si,lli = em_censored_normal(Xwi,Y,C,L)
    di = max(0,lli-llf)

    total = sum(drops.values()) + di
    rows = {}
    for f in FACTOR_ORDER:
        rows[f] = round(100*drops[f]/total, 1)
    rows["interaction"] = round(100*di/total, 1)

    return {
        "n_total": len(entries),
        "n_censored": int(sum(C)),
        "sigma_aft": round(float(sf), 4),
        "logLik": round(float(llf), 2),
        "std_var_pct": rows["size_surface_standard"],
        "mean_stress_pct": rows["mean_stress_method"],
        "rz_pct": rows["rz_level"],
        "sn_pct": rows["sn_source"],
        "damage_pct": rows["damage_method"],
        "interaction_pct": rows["interaction"],
    }


# ============================================================
# 4. Run across Y_Sa range
# ============================================================
ysa_values = [1.9, 1.7, 1.55, 1.4, 1.3, 1.2]
results = {}

print(f"\n{'Y_Sa':>8s} {'N_cens':>7s} {'Std%':>7s} {'MS%':>7s} {'Rz%':>7s} {'SN%':>7s} {'Dmg%':>7s} {'Int%':>7s}")
print("-" * 60)

for ysa in ysa_values:
    raw = recompute_all(ysa)
    entries = build_entries_list(raw)
    r = run_aft(entries)
    results[ysa] = r
    print(f"{ysa:>8.2f} {r['n_censored']:>7d} {r['std_var_pct']:>7.1f} {r['mean_stress_pct']:>7.1f} "
          f"{r['rz_pct']:>7.1f} {r['sn_pct']:>7.1f} {r['damage_pct']:>7.1f} {r['interaction_pct']:>7.1f}")

# ============================================================
# 5. Summary
# ============================================================
baseline = results[1.55]
std_values = [results[y]["std_var_pct"] for y in ysa_values]
std_mean = np.mean(std_values)
std_range = (min(std_values), max(std_values))
std_delta_max = max(abs(std_values[0] - baseline["std_var_pct"]),
                    abs(std_values[-1] - baseline["std_var_pct"]))

ms_values = [results[y]["mean_stress_pct"] for y in ysa_values]
rz_values = [results[y]["rz_pct"] for y in ysa_values]

print(f"\n=== Sensitivity Summary ===")
print(f"Y_Sa range: {min(ysa_values):.1f} – {max(ysa_values):.1f}")
print(f"Standard-choice variance: {baseline['std_var_pct']:.1f}% (baseline), "
      f"range [{min(std_values):.1f}, {max(std_values):.1f}]%, "
      f"max delta = {std_delta_max:.1f} pp")
print(f"Mean-stress variance:      {baseline['mean_stress_pct']:.1f}% (baseline), "
      f"range [{min(ms_values):.1f}, {max(ms_values):.1f}]%")
print(f"Rz variance:               {baseline['rz_pct']:.1f}% (baseline), "
      f"range [{min(rz_values):.1f}, {max(rz_values):.1f}]%")

# Check three-factor stability
three_factor_sum = [results[y]["std_var_pct"] + results[y]["mean_stress_pct"] + results[y]["rz_pct"] for y in ysa_values]
print(f"Three-factor sum range: [{min(three_factor_sum):.1f}, {max(three_factor_sum):.1f}]%")

# Ranking stability
print(f"\nRanking across Y_Sa range:")
for y in ysa_values:
    r = results[y]
    factors = [("MS", r["mean_stress_pct"]), ("Std", r["std_var_pct"]),
               ("Rz", r["rz_pct"]), ("SN", r["sn_pct"])]
    factors.sort(key=lambda x: -x[1])
    ranking = " > ".join(f"{name}({val:.0f})" for name, val in factors)
    print(f"  Y_Sa={y:.2f}: {ranking}")

output = {
    "description": "Full-range Y_Sa perturbation for ALL three standards",
    "Y_Sa_nominal": Y_Sa_nominal,
    "baseline_std_var_pct": baseline["std_var_pct"],
    "std_var_range": [min(std_values), max(std_values)],
    "std_var_max_delta_pp": round(std_delta_max, 1),
    "conclusion": "Standard-choice variance stable within ±X pp across Y_Sa ∈ [1.2, 1.9]",
    "results": {str(y): results[y] for y in ysa_values},
}

with open("output/ysa_sensitivity_full.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved output/ysa_sensitivity_full.json")
