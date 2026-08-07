"""
variable_amplitude_demo.py — Two-level block-loading damage comparison
========================================================================
Simple demonstration: Miner (with Haibach extension for sub-endurance cycles)
vs Modified Miner (ignores sub-endurance cycles).

Two-block spectrum:
  Block 1: 100% torque × 10^4 cycles (above endurance)
  Block 2:  35% torque × 10^6 cycles (below endurance for most combos)

Under constant amplitude: damage rule variance = 0.9% (trivial scaling).
Under variable amplitude: damage rule variance = X% (meaningful divergence).
"""

import json, os, itertools, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import gear_params
from fatigue_models import apply_mean_stress_correction, apply_size_surface_factor
from gear_params import RZ_LEVELS, LOAD_SPECTRUM, SN_CURVES, MATERIAL
from aft_variance_decomposition import em_censored_normal, build_design, extract_arrays, FACTORS, FACTOR_ORDER

BASE_TORQUE = 700.0
BLOCK_HIGH = {'ratio': 1.00, 'cycles': 1e4}
BLOCK_LOW  = {'ratio': 0.35, 'cycles': 1e6}

def compute_effective_stress(torque_ratio, sn_source, mean_stress_method, std, rz_um):
    """Compute sigma_eff for a given torque ratio."""
    T = BASE_TORQUE * torque_ratio
    old_T = gear_params.LOAD['torque_pinion']
    old_Ft = gear_params.LOAD['tangential_force']
    try:
        gear_params.LOAD['torque_pinion'] = T
        Ft = 2 * T * 1000 / gear_params.GEAR['pitch_diameter_pinion']
        gear_params.LOAD['tangential_force'] = Ft
        gear_params.LOAD['Ft_per_unit_facewidth'] = Ft / gear_params.GEAR['face_width']

        sigma_F0 = (Ft / (gear_params.GEAR['face_width'] * gear_params.GEAR['module_mn'])
                    * 2.80 * 1.55 * 0.691)  # ISO Method B
        R = LOAD_SPECTRUM['stress_ratio_R']
        sigma_ar = apply_mean_stress_correction(sigma_F0, R, method=mean_stress_method,
                                                 sn_source=sn_source)
        sigma_eff = apply_size_surface_factor(sigma_ar, standard=std, Rz_um=rz_um)
        return sigma_eff
    finally:
        gear_params.LOAD['torque_pinion'] = old_T
        gear_params.LOAD['tangential_force'] = old_Ft


def compute_nf_basquin(sigma_eff, sn_source):
    """Basquin: Nf = 0.5 * (sigma_eff/sigma_f')^(1/b). Returns inf if below endurance."""
    curve = SN_CURVES[sn_source]
    if sigma_eff <= curve['endurance_limit']:
        return float('inf')
    return 0.5 * (sigma_eff / curve['sigma_f_prime']) ** (1.0 / curve['b'])


def compute_nf_haibach(sigma_eff, sn_source):
    """
    Haibach extension: below endurance limit, use modified slope 2b-1.
    Above endurance: standard Basquin. Returns inf if sigma_eff < 0.5*endurance.
    """
    curve = SN_CURVES[sn_source]
    se = curve['endurance_limit']
    if sigma_eff > se:
        return 0.5 * (sigma_eff / curve['sigma_f_prime']) ** (1.0 / curve['b'])
    elif sigma_eff > 0.5 * se:
        # Haibach: sigma_a = sigma_f' * (2Nf)^(2b-1) in the transition region
        b_h = 2 * curve['b'] - 1  # shallower slope
        N_knee = 0.5 * (se / curve['sigma_f_prime']) ** (1.0 / curve['b'])
        if b_h >= 0:
            return float('inf')
        # Continuity at knee: same Nf at sigma=se
        # Below knee: sigma_eff = C * (2Nf)^(b_h)
        C = se / (2 * N_knee) ** b_h
        return 0.5 * (sigma_eff / C) ** (1.0 / b_h)
    else:
        return float('inf')  # far below endurance


def variable_amplitude_life(sn_source, mean_stress_method, std, rz_um, damage_method):
    """
    Compute total cycles to failure under two-block variable-amplitude loading.
    """
    sigma_high = compute_effective_stress(1.0, sn_source, mean_stress_method, std, rz_um)
    sigma_low  = compute_effective_stress(0.35, sn_source, mean_stress_method, std, rz_um)

    if damage_method == 'Miner_original':
        # Miner with Haibach extension: sub-endurance cycles still cause damage
        Nf_high = compute_nf_basquin(sigma_high, sn_source)
        Nf_low  = compute_nf_haibach(sigma_low, sn_source)
        Dc = 1.0
    else:
        # Modified Miner: sub-endurance cycles are ignored
        Nf_high = compute_nf_basquin(sigma_high, sn_source)
        curve = SN_CURVES[sn_source]
        if sigma_low < 0.5 * curve['endurance_limit']:
            Nf_low = float('inf')  # ignored
        else:
            Nf_low = compute_nf_basquin(sigma_low, sn_source)
        Dc = 0.7

    damage = 0.0
    if np.isfinite(Nf_high) and Nf_high > 0:
        damage += BLOCK_HIGH['cycles'] / Nf_high
    if np.isfinite(Nf_low) and Nf_low > 0:
        damage += BLOCK_LOW['cycles'] / Nf_low

    if damage <= 0:
        return float('inf'), None

    total_cycles_per_pass = BLOCK_HIGH['cycles'] + BLOCK_LOW['cycles']
    passes = Dc / damage
    return passes * total_cycles_per_pass, np.log10(passes * total_cycles_per_pass)


# ============================================================
SN = ['Waterloo_SMDIdbase_Iter068', 'Waterloo_SMDIdbase_Iter066']
MEAN = ['Goodman', 'Gerber', 'Morrow', 'SWT']
STD = ['ISO_6336', 'AGMA_2001', 'FKM']
RZ = ['ground_Rz4', 'machined_Rz15', 'as_forged_Rz50']

print("=== Two-Level Variable-Amplitude Demo ===")
print(f"Block 1: {BLOCK_HIGH['ratio']*100:.0f}% torque, {BLOCK_HIGH['cycles']:.0e} cycles")
print(f"Block 2: {BLOCK_LOW['ratio']*100:.0f}% torque, {BLOCK_LOW['cycles']:.0e} cycles")
print()

results = []
for sn, ms, std, rz in itertools.product(SN, MEAN, STD, RZ):
    rz_um = RZ_LEVELS[rz]['Rz_um']
    row = {'sn': sn, 'ms': ms, 'std': std, 'rz': rz}
    for dm in ['Miner_original', 'Miner_modified']:
        Nf, logNf = variable_amplitude_life(sn, ms, std, rz_um, dm)
        if logNf is not None:
            row['logNf_' + dm] = round(float(logNf), 4)
    results.append(row)

# Build AFT entries for Miner
entries_miner = []
for r in results:
    e = {'sn_source': r['sn'], 'mean_stress_method': r['ms'],
         'damage_method': 'Miner_original', 'size_surface_standard': r['std'],
         'rz_level': r['rz']}
    ln = r.get('logNf_Miner_original')
    if ln is not None:
        e['logNf'] = float(ln); e['censored'] = 0
    else:
        e['logNf'] = None; e['censored'] = 1
    entries_miner.append(e)

# Build AFT entries for Modified Miner
entries_mod = []
for r in results:
    e = {'sn_source': r['sn'], 'mean_stress_method': r['ms'],
         'damage_method': 'Miner_modified', 'size_surface_standard': r['std'],
         'rz_level': r['rz']}
    ln = r.get('logNf_Miner_modified')
    if ln is not None:
        e['logNf'] = float(ln); e['censored'] = 0
    else:
        e['logNf'] = None; e['censored'] = 1
    entries_mod.append(e)

# Combine: Miner + ModMiner as the "damage_method" factor
entries_combined = []
for r_miner, r_mod in zip(results, results):
    for label, r in [('Miner_original', r_miner), ('Miner_modified', r_mod)]:
        e = {'sn_source': r['sn'], 'mean_stress_method': r['ms'],
             'damage_method': label, 'size_surface_standard': r['std'],
             'rz_level': r['rz']}
        key = 'logNf_' + label
        ln = r.get(key)
        if ln is not None:
            e['logNf'] = float(ln); e['censored'] = 0
        else:
            e['logNf'] = None; e['censored'] = 1
        entries_combined.append(e)

for elist in [entries_combined]:
    MAX = max(e['logNf'] for e in elist if e['logNf'] is not None)
    for e in elist:
        if e['censored'] == 1:
            e['logNf_lower'] = MAX

# Run AFT on combined entries
Y, C, L = extract_arrays(entries_combined)
Xf = build_design(entries_combined, FACTOR_ORDER)
bf, sf, llf = em_censored_normal(Xf, Y, C, L)

drops = {}
for drop_f in FACTOR_ORDER:
    rem = [f for f in FACTOR_ORDER if f != drop_f]
    Xd = build_design(entries_combined, rem)
    bd, sd, ld = em_censored_normal(Xd, Y, C, L)
    drops[drop_f] = llf - ld

# Interaction
ls = FACTORS['size_surface_standard']; lr = FACTORS['rz_level']
Xc = []
for si in range(1, len(ls)):
    for rj in range(1, len(lr)):
        Xc.append(np.array([1.0 if e['size_surface_standard']==ls[si] and e['rz_level']==lr[rj] else 0.0 for e in entries_combined]))
Xi = np.column_stack(Xc) if Xc else np.zeros((len(entries_combined),0))
Xi -= Xi.mean(axis=0); Xwi = np.column_stack([Xf, Xi])
bi, si, lli = em_censored_normal(Xwi, Y, C, L)
di = max(0, lli - llf)

total = sum(drops.values()) + di
print("=== AFT Variance Decomposition (Variable-Amplitude) ===")
for f in FACTOR_ORDER:
    pct = 100 * drops[f] / total
    print(f"  {f:30s}: {pct:.1f}%")
print(f"  {'Std x Rz':30s}: {100*di/total:.1f}%")
print(f"  N={len(entries_combined)}, censored={sum(C)}, sigma={sf:.4f}")

damage_var = 100 * drops['damage_method'] / total
print(f"\n  Damage rule variance: {damage_var:.1f}% (vs 0.9% under constant amplitude)")
print(f"  Ratio VA/CA: {damage_var/0.9:.1f}x")

# Save
output = {
    'description': 'Two-level variable-amplitude damage rule comparison',
    'blocks': [BLOCK_HIGH, BLOCK_LOW],
    'damage_rule_variance_pct': round(damage_var, 1),
    'constant_amplitude_baseline_pct': 0.9,
    'ratio_VA_to_CA': round(damage_var/0.9, 1),
}
with open('output/variable_amplitude_demo.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nSaved output/variable_amplitude_demo.json")
