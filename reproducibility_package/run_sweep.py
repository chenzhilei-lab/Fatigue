"""
run_sweep.py — Full factorial sweep: 144 combinations
2×4×2×3×3 = 144 combos (Kf switch removed in v2.3).
Output: output/sweep.json
"""

import json
import os
import time
import itertools
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)

from fatigue_models import compute_fatigue_life
from gear_params import RZ_LEVELS

# Switch values (6 factors)
SN_SOURCES = ["Waterloo_SMDIdbase_Iter068", "Waterloo_SMDIdbase_Iter066"]
MEAN_STRESS_METHODS = ["Goodman", "Gerber", "Morrow", "SWT"]
DAMAGE_METHODS = ["Miner_original", "Miner_modified"]
SIZE_SURFACE_STANDARDS = ["ISO_6336", "AGMA_2001", "FKM"]
RZ_KEYS = list(RZ_LEVELS.keys())

# Kf switch removed in v2.3 — YSa already handles stress concentration (ISO 6336)
combinations = list(itertools.product(
    SN_SOURCES, MEAN_STRESS_METHODS,
    DAMAGE_METHODS, SIZE_SURFACE_STANDARDS, RZ_KEYS
))

N_COMBOS = len(combinations)
print(f"Running {N_COMBOS} combinations (2×4×2×3×3)...")
print(f"Test gear: module=3.0mm, 20 teeth, 42CrMo4, 700 N·m")
print()

results = []
runouts = 0
low_cycle = 0

t0 = time.time()

for sn, ms, dm, ss, rz_key in combinations:
    rz_info = RZ_LEVELS[rz_key]
    rz_um = rz_info["Rz_um"]

    Nf, diag = compute_fatigue_life(
        sn_source=sn,
        kf_method="Peterson",  # unused — Kf switch removed in v2.3
        mean_stress_method=ms,
        damage_method=dm,
        size_surface_standard=ss,
        rz_um=rz_um,
        verbose=False,
    )

    entry = {
        "sn_source": sn,
        "mean_stress_method": ms,
        "damage_method": dm,
        "size_surface_standard": ss,
        "rz_level": rz_key,
        "rz_um": rz_um,
        "Nf": float(Nf) if np.isfinite(Nf) else "inf",
        "log10_Nf": float(np.log10(Nf)) if (np.isfinite(Nf) and Nf > 0) else None,
        "sigma_nominal": float(diag["sigma_nominal"]),
        "sigma_ar": float(diag["sigma_ar"]),
        "sigma_effective": float(diag["sigma_effective"]),
    }
    results.append(entry)

    if not np.isfinite(Nf):
        runouts += 1
    elif Nf < 1e4:
        low_cycle += 1

elapsed = time.time() - t0

# Stats
finite_Nf = [r["log10_Nf"] for r in results if r["log10_Nf"] is not None]
print(f"Completed in {elapsed:.1f}s")
print(f"  Run-outs (Nf=inf): {runouts}/{N_COMBOS}")
print(f"  Low-cycle fatigue (<10^4): {low_cycle}/{N_COMBOS}")
if finite_Nf:
    print(f"  Min log10(Nf): {min(finite_Nf):.2f}  ({10**min(finite_Nf):.1e} cycles)")
    print(f"  Max log10(Nf): {max(finite_Nf):.2f}  ({10**max(finite_Nf):.1e} cycles)")
    print(f"  Spread (dex): {max(finite_Nf) - min(finite_Nf):.2f}")

# Save
with open("output/sweep.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} entries to output/sweep.json")

summary = {
    "n_combinations": N_COMBOS,
    "n_runouts": runouts,
    "n_low_cycle": low_cycle,
    "n_valid": len(finite_Nf),
    "min_log10_Nf": float(min(finite_Nf)) if finite_Nf else None,
    "max_log10_Nf": float(max(finite_Nf)) if finite_Nf else None,
    "spread_dex": float(max(finite_Nf) - min(finite_Nf)) if finite_Nf else None,
    "gear_module_mm": 3.0, "teeth": 20,
    "material": "42CrMo4 / AISI 4140",
    "torque_Nm": 700, "speed_rpm": 1500,
}
with open("output/sweep_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved sweep_summary.json")
