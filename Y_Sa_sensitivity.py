"""
Y_Sa_sensitivity.py — Sensitivity test for uniform Y_Sa assumption (v2)
=========================================================================
Corrected direction: ISO keeps Y_Sa=1.55 (native), AGMA and FKM reduced.
This tests the reviewer's concern: "AGMA and FKM don't use Y_Sa — how much
of the standard-choice variance is attributable to this modeling choice?"

Method:
  - Keep ISO at Y_Sa=1.55 (always)
  - Reduce Y_Sa for AGMA + FKM entries: 1.55→1.45→1.35→1.15→1.0
  - Run AFT point estimates for each scenario
  - Report standard-choice variance trend

Output: output/ysa_sensitivity.json
"""

import json, sys, os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from fatigue_models import (
    apply_mean_stress_correction,
    apply_size_surface_factor,
    compute_fatigue_life_sn,
    apply_damage_rule,
)
from gear_params import GEAR, LOAD, LOAD_SPECTRUM, RZ_LEVELS

from aft_variance_decomposition import (
    em_censored_normal, build_design, extract_arrays,
    FACTORS, FACTOR_ORDER,
)

# ============================================================
# 1. Load data
# ============================================================
with open("output/sweep.json") as f:
    sweep = json.load(f)

iso_entries = [e for e in sweep if e["size_surface_standard"] == "ISO_6336"]
agma_entries_orig = [e for e in sweep if e["size_surface_standard"] == "AGMA_2001"]
fkm_entries_orig = [e for e in sweep if e["size_surface_standard"] == "FKM"]
print(f"Loaded: {len(iso_entries)} ISO, {len(agma_entries_orig)} AGMA, {len(fkm_entries_orig)} FKM")

z = GEAR["teeth_pinion"]
Y_Sa_iso = 1.58 - 0.0015 * z
sigma0 = iso_entries[0]["sigma_nominal"]
print(f"Y_Sa_ISO = {Y_Sa_iso:.4f}, baseline sigma_nominal = {sigma0:.1f} MPa")

# ============================================================
# 2. Recompute AGMA/FKM entries with reduced Y_Sa
# ============================================================
def recompute_entries(orig_entries, ysa_new, standard_label):
    """Recompute entries with modified baseline stress (new Y_Sa)."""
    sigma_F0_new = sigma0 * (ysa_new / Y_Sa_iso)
    R = LOAD_SPECTRUM["stress_ratio_R"]

    new_entries = []
    for entry in orig_entries:
        sigma_ar = apply_mean_stress_correction(
            sigma_F0_new, R, method=entry["mean_stress_method"],
            sn_source=entry["sn_source"]
        )
        sigma_eff = apply_size_surface_factor(
            sigma_ar, standard=standard_label,
            Rz_um=RZ_LEVELS[entry["rz_level"]]["Rz_um"],
            sn_source=entry["sn_source"]
        )
        Nf_raw = compute_fatigue_life_sn(sigma_eff, sn_source=entry["sn_source"])
        Nf_final = apply_damage_rule(Nf_raw, method=entry["damage_method"])

        new_entry = {
            "sn_source": entry["sn_source"],
            "mean_stress_method": entry["mean_stress_method"],
            "damage_method": entry["damage_method"],
            "size_surface_standard": standard_label,
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
# 3. Build AFT entries + run decomposition
# ============================================================
def build_entries_list(iso, agma, fkm):
    entries = []
    for e in iso + agma + fkm:
        entry = {
            "sn_source": e["sn_source"],
            "mean_stress_method": e["mean_stress_method"],
            "damage_method": e["damage_method"],
            "size_surface_standard": e["size_surface_standard"],
            "rz_level": e["rz_level"],
        }
        nf = e.get("Nf", float("inf"))
        log_nf = e.get("log10_Nf")
        if isinstance(nf, (int, float)) and np.isfinite(nf) and nf > 0 and log_nf is not None:
            entry["logNf"] = float(log_nf)
            entry["censored"] = 0
        else:
            entry["logNf"] = None
            entry["censored"] = 1
        entries.append(entry)

    MAX_OBS = max(e["logNf"] for e in entries if e["logNf"] is not None)
    for e in entries:
        if e["censored"] == 1:
            e["logNf_lower"] = MAX_OBS
    return entries


def run_aft_point(entries):
    Y_OBS, CENS, Y_LOWER = extract_arrays(entries)
    N_TOTAL = len(entries)

    X_FULL = build_design(entries, FACTOR_ORDER)
    beta_full, sigma_full, ll_full = em_censored_normal(X_FULL, Y_OBS, CENS, Y_LOWER)

    drop_results = {}
    for drop_factor in FACTOR_ORDER:
        remaining = [f for f in FACTOR_ORDER if f != drop_factor]
        X_drop = build_design(entries, remaining)
        beta, sigma, ll = em_censored_normal(X_drop, Y_OBS, CENS, Y_LOWER)
        drop_results[drop_factor] = {
            "delta_logLik": round(ll_full - ll, 4),
            "df": len(FACTORS[drop_factor]) - 1,
        }

    # Interaction
    levels_std = FACTORS["size_surface_standard"]
    levels_rz = FACTORS["rz_level"]
    X_int_cols = []
    for si in range(1, len(levels_std)):
        for rj in range(1, len(levels_rz)):
            X_int_cols.append(np.array([
                1.0 if e["size_surface_standard"] == levels_std[si]
                     and e["rz_level"] == levels_rz[rj]
                else 0.0 for e in entries
            ]))
    X_int_raw = np.column_stack(X_int_cols) if X_int_cols else np.zeros((N_TOTAL, 0))
    X_int_centered = X_int_raw - X_int_raw.mean(axis=0)
    X_W_INT = np.column_stack([X_FULL, X_int_centered])
    beta_int, sigma_int, ll_int = em_censored_normal(X_W_INT, Y_OBS, CENS, Y_LOWER)
    delta_int = max(0, ll_int - ll_full)

    total_delta = sum(d["delta_logLik"] for d in drop_results.values()) + delta_int

    anova_rows = {}
    for fname in FACTOR_ORDER:
        anova_rows[fname] = round(100 * drop_results[fname]["delta_logLik"] / total_delta, 1)
    anova_rows["size_surface_standard x rz_level"] = round(100 * delta_int / total_delta, 1)

    return {
        "n_total": N_TOTAL,
        "n_censored": int(sum(CENS)),
        "sigma_aft": round(float(sigma_full), 4),
        "logLik_full": round(float(ll_full), 2),
        "standard_choice_var_pct": anova_rows["size_surface_standard"],
        "anova_rows": anova_rows,
    }


# ============================================================
# 4. Run scenarios: reduce Y_Sa for AGMA + FKM, keep ISO fixed
# ============================================================
scenarios = [
    # (label, Y_Sa_for_AGMA_FKM)
    ("baseline",   Y_Sa_iso),
    ("AGMA/FKM=1.45", 1.45),
    ("AGMA/FKM=1.35", 1.35),
    ("AGMA/FKM=1.15", 1.15),
    ("AGMA/FKM=1.00", 1.00),
]

results = {}
print(f"\n{'Scenario':<20s} {'Y_Sa_AGFM':>10s} {'N_cens':>7s} {'Std Var%':>10s} {'sigma_aft':>10s}")
print("-" * 62)

for label, ysa_agfm in scenarios:
    if label == "baseline":
        agma_data = agma_entries_orig
        fkm_data = fkm_entries_orig
    else:
        agma_data = recompute_entries(agma_entries_orig, ysa_agfm, "AGMA_2001")
        fkm_data = recompute_entries(fkm_entries_orig, ysa_agfm, "FKM")

    entries = build_entries_list(iso_entries, agma_data, fkm_data)
    result = run_aft_point(entries)
    results[label] = result
    print(f"{label:<20s} {ysa_agfm:>10.3f} {result['n_censored']:>7d} "
          f"{result['standard_choice_var_pct']:>10.1f}% {result['sigma_aft']:>10.4f}")

# ============================================================
# 5. Summary
# ============================================================
b = results["baseline"]
print(f"\n=== Full Factor Trend ===")
print(f"{'Factor':30s} ", end="")
for label, _, in scenarios:
    print(f"{label:<16s}", end="")
print()
print("-" * (30 + 16 * len(scenarios)))
for fname in FACTOR_ORDER + ["size_surface_standard x rz_level"]:
    print(f"{fname:30s} ", end="")
    for label, _, in scenarios:
        v = results[label]["anova_rows"][fname]
        marker = " ←" if fname == "size_surface_standard" else ""
        print(f"{v:>5.1f}%{'':10s}", end="")
    print()

# Delta
base_std = b["standard_choice_var_pct"]
print(f"\n=== Standard-Choice Sensitivity ===")
for label, ysa_val in scenarios[1:]:
    r = results[label]
    delta = base_std - r["standard_choice_var_pct"]
    print(f"  {label}: {r['standard_choice_var_pct']:.1f}% (Δ = {delta:+.1f} pp vs baseline)")

# Interpretation
y10 = results["AGMA/FKM=1.00"]["standard_choice_var_pct"]
delta_max = base_std - y10
robust = delta_max < 10.0

print(f"\n  Max attributable to Y_Sa: {delta_max:.1f} pp")
print(f"  Conclusion robust: {'YES' if robust else 'NO'}")

output = {
    "description": "Sensitivity of standard-choice variance to uniform Y_Sa assumption",
    "method": "ISO kept at Y_Sa=1.55 (native); AGMA and FKM reduced to test reviewer concern",
    "scenarios": {
        label: {
            "Y_Sa_AGMA_FKM": ysa_val,
            "standard_choice_var_pct": results[label]["standard_choice_var_pct"],
            "sigma_aft": results[label]["sigma_aft"],
            "n_censored": results[label]["n_censored"],
            "n_total": results[label]["n_total"],
        }
        for label, ysa_val in scenarios
    },
    "interpretation": {
        "delta_var_pp": round(delta_max, 1),
        "conclusion_robust": bool(robust),
    },
}

with open("output/ysa_sensitivity.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved output/ysa_sensitivity.json")
