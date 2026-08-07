"""
benchmark_wang2023.py — Run our 5-switch pipeline on Wang et al. (2023) test gear
==================================================================================
Wang J.J., Pei W.C., Ji H.C., Long H.Y., Wang Z.T. (2023)
"Research on bending fatigue test based on 42CrMo gear" (in Chinese)
J. Mech. Strength 45(2): 474-480. doi:10.16579/j.issn.1001.9669.2023.02.030

Test gear (Tab. 1 of the original): m=6 mm, z=20, b=25 mm, carburized 42CrMo,
HRC 58-62, R=0.1.
5 stress levels (Tab. 2): 292.95, 257.12, 221.14, 186.02, 150.74 MPa
Run-out at 3×10^6 cycles
Fatigue limits at 10^7 cycles (Tab. 6): 167.23 (50%), 151.72 (90%),
149.22 (95%), 144.10 (99%) MPa  →  range 144-167 MPa
Life scatter at fixed stress (Tab. 3, levels I-IV): max/min ratio 2.3-4.3,
i.e. ~0.4-0.6 dex.

Our approach:
1. Adapt our gear_params to Wang's geometry
2. Run the 144-combo sweep at Wang's 5 stress levels
3. Compare predicted life spread vs measured scatter
4. Output benchmark summary
"""

import json, os, itertools, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import gear_params
from fatigue_models import compute_fatigue_life
from gear_params import RZ_LEVELS

# ============================================================
# 1. Configure Wang's test gear
# ============================================================
# Wang's gear (Tab. 1): m=6, z=20, b=25mm, carburized 42CrMo, R=0.1
# Wang reports nominal bending stress directly — we need to match it

SN = ['Waterloo_SMDIdbase_Iter068', 'Waterloo_SMDIdbase_Iter066']
MEAN = ['Goodman', 'Gerber', 'Morrow', 'SWT']
DAM = ['Miner_original', 'Miner_modified']
STD = ['ISO_6336', 'AGMA_2001', 'FKM']
RZ = list(RZ_LEVELS.keys())

def run_sweep_for_gear(torque_Nm, module_mn, teeth, face_width, stress_ratio):
    """Run the 144-combo sweep with custom gear parameters."""
    old = {
        'torque': gear_params.LOAD['torque_pinion'],
        'Ft': gear_params.LOAD['tangential_force'],
        'Ftw': gear_params.LOAD['Ft_per_unit_facewidth'],
        'mn': gear_params.GEAR['module_mn'],
        'z': gear_params.GEAR['teeth_pinion'],
        'b': gear_params.GEAR['face_width'],
        'R': gear_params.LOAD_SPECTRUM['stress_ratio_R'],
        'pd': gear_params.GEAR['pitch_diameter_pinion'],
    }

    try:
        gear_params.LOAD['torque_pinion'] = torque_Nm
        gear_params.GEAR['module_mn'] = module_mn
        gear_params.GEAR['teeth_pinion'] = teeth
        gear_params.GEAR['face_width'] = face_width
        gear_params.LOAD_SPECTRUM['stress_ratio_R'] = stress_ratio
        pd = module_mn * teeth
        gear_params.GEAR['pitch_diameter_pinion'] = pd
        Ft = 2 * torque_Nm * 1000 / pd
        gear_params.LOAD['tangential_force'] = Ft
        gear_params.LOAD['Ft_per_unit_facewidth'] = Ft / face_width

        results = []
        for sn, ms, dm, std, rz in itertools.product(SN, MEAN, DAM, STD, RZ):
            rz_um = RZ_LEVELS[rz]['Rz_um']
            Nf, diag = compute_fatigue_life(sn, 'Peterson', ms, dm, std, rz_um)
            entry = {
                'sn_source': sn, 'mean_stress_method': ms,
                'damage_method': dm, 'size_surface_standard': std,
                'rz_level': rz, 'rz_um': rz_um,
                'sigma_nominal': round(float(diag.get('sigma_nominal', 0)), 2),
                'sigma_ar': round(float(diag.get('sigma_ar', 0)), 2),
                'sigma_effective': round(float(diag.get('sigma_effective', 0)), 2),
            }
            if np.isfinite(Nf) and Nf > 0:
                entry['Nf'] = float(Nf)
                entry['log10_Nf'] = round(float(np.log10(Nf)), 6)
            else:
                entry['Nf'] = float('inf')
                entry['log10_Nf'] = None
            results.append(entry)
        return results
    finally:
        gear_params.LOAD['torque_pinion'] = old['torque']
        gear_params.LOAD['tangential_force'] = old['Ft']
        gear_params.LOAD['Ft_per_unit_facewidth'] = old['Ftw']
        gear_params.GEAR['module_mn'] = old['mn']
        gear_params.GEAR['teeth_pinion'] = old['z']
        gear_params.GEAR['face_width'] = old['b']
        gear_params.LOAD_SPECTRUM['stress_ratio_R'] = old['R']
        gear_params.GEAR['pitch_diameter_pinion'] = old['pd']


def calibrate_torque(target_sigma, module_mn, teeth, face_width, stress_ratio,
                     initial_T=500.0, tol=0.2):
    """Find the torque that reproduces a target nominal root stress."""
    T = initial_T
    for _ in range(12):
        sweep_data = run_sweep_for_gear(T, module_mn, teeth, face_width,
                                        stress_ratio)
        actual_sigma = sweep_data[0]['sigma_nominal']
        if abs(actual_sigma - target_sigma) <= tol:
            return T, sweep_data, actual_sigma
        # linear scaling: sigma ∝ torque
        T = T * target_sigma / actual_sigma
    return T, sweep_data, actual_sigma


# ============================================================
# 2. Run Wang's gear at the 5 reported stress levels
# ============================================================
# Wang Tab. 2: loads 25/22/19/16/13 kN → σ_F = 292.95/257.12/221.14/186.02/150.74
wang_stress_levels = [292.95, 257.12, 221.14, 186.02, 150.74]  # MPa (R=0-equivalent)

print("=== Wang et al. (2023) Benchmark ===")
print("Wang gear (Tab. 1): m=6 mm, z=20, b=25 mm, carburized 42CrMo, HRC 58-62, R=0.1")
print(f"Wang stress levels (MPa): {wang_stress_levels}")
print()

results_by_level = {}

for target_sigma in wang_stress_levels:
    best_T, sweep_data, actual_sigma = calibrate_torque(
        target_sigma, 6, 20, 25, 0.1)
    print(f"Target σ={target_sigma} MPa, T={best_T:.0f} Nm → actual σ_F0={actual_sigma:.2f} MPa")

    finite = [r for r in sweep_data if r['log10_Nf'] is not None]
    n_finite = len(finite)
    n_runout = 144 - n_finite

    if n_finite > 0:
        logNf = np.array([r['log10_Nf'] for r in finite])
        p5 = np.percentile(logNf, 5)
        p50 = np.percentile(logNf, 50)
        p95 = np.percentile(logNf, 95)
        spread = p95 - p5
        print(f"  Finite={n_finite}, Run-out={n_runout}")
        print(f"  log10(Nf): median={p50:.2f}, 5%={p5:.2f}, 95%={p95:.2f}, spread={spread:.2f} dex")
        print(f"  Nf range: {10**p5:.0f} – {10**p95:.0f} cycles")
    else:
        print(f"  ALL RUN-OUT ({n_runout}/144)")
        p50, spread = None, None

    results_by_level[target_sigma] = {
        'torque_Nm': round(best_T),
        'sigma_F0_actual': round(actual_sigma, 2),
        'n_finite': n_finite,
        'n_runout': n_runout,
        'median_logNf': round(float(p50), 2) if p50 else None,
        'spread_dex': round(float(spread), 2) if spread else None,
    }
    print()

# ============================================================
# 3. Compare with Wang's reported scatter
# ============================================================
# Wang Tab. 3 (levels I-IV): max/min life ratio 2.3-4.3 → ~0.4-0.6 dex.
# Level V (150.74 MPa): all run-out at 3×10^6 cycles.

print("=== Comparison ===")
print("Wang reports (Tab. 3): ~0.4-0.6 dex scatter at a fixed stress level;")
print("                       level V all run-out at 3e6 cycles")
print("Our pipeline method-induced spread:")
for sigma, r in results_by_level.items():
    if r['spread_dex']:
        print(f"  σ={sigma} MPa: method spread = {r['spread_dex']:.1f} dex "
              f"({r['n_finite']}/144 finite, {r['n_runout']} run-out)")
    else:
        print(f"  σ={sigma} MPa: all run-out (method spread undefined)")

print()
print("Interpretation: At stress levels where both material scatter and")
print("method-induced spread are quantifiable, we can compare their magnitudes.")
print("If method spread ≈ material scatter, then engineering choices matter as")
print("much as the material itself — a powerful validation of our framework.")

# Save
output = {
    'description': 'Wang et al. (2023) benchmark: 5-switch pipeline on published test gear',
    'wang_gear': {'m': 6, 'z': 20, 'b': 25, 'material': '42CrMo (carburized, HRC 58-62)',
                  'R': 0.1},
    'wang_stress_levels_MPa': wang_stress_levels,
    'wang_fatigue_limit_MPa_at_1e7': [167.23, 151.72, 149.22, 144.10],
    'wang_reported_scatter_dex': '0.4-0.6 (Tab. 3, levels I-IV)',
    'results': {str(round(k, 2)): v for k, v in results_by_level.items()},
}

with open('output/benchmark_wang2023.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nSaved output/benchmark_wang2023.json")
