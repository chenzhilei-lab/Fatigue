"""
multi_torque_analysis.py — Run sweep at 3 torque levels, compare ANOVA
Output: output/multi_torque_anova.json
"""
import json, os, itertools, copy, numpy as np
from scipy.stats import f as f_dist

os.makedirs('output', exist_ok=True)

import gear_params
from fatigue_models import compute_fatigue_life
from gear_params import RZ_LEVELS

def run_sweep_at_torque(torque):
    """Temporarily set torque and run sweep."""
    old_torque = gear_params.LOAD['torque_pinion']
    old_Ft = gear_params.LOAD['tangential_force']
    try:
        gear_params.LOAD['torque_pinion'] = torque
        Ft = 2 * torque * 1000 / gear_params.GEAR['pitch_diameter_pinion']
        gear_params.LOAD['tangential_force'] = Ft
        gear_params.LOAD['Ft_per_unit_facewidth'] = Ft / gear_params.GEAR['face_width']

        SN = ['Waterloo_SMDIdbase_Iter068', 'Waterloo_SMDIdbase_Iter066']
        MEAN = ['Goodman', 'Gerber', 'Morrow', 'SWT']
        DAM = ['Miner_original', 'Miner_modified']
        STD = ['ISO_6336', 'AGMA_2001', 'FKM']
        RZ = list(RZ_LEVELS.keys())

        combos = list(itertools.product(SN, MEAN, DAM, STD, RZ))
        results = []
        for sn, ms, dm, ss, rz_key in combos:
            rz_um = RZ_LEVELS[rz_key]['Rz_um']
            Nf, diag = compute_fatigue_life(sn, 'Peterson', ms, dm, ss, rz_um=rz_um)
            results.append({
                'sn_source': sn, 'mean_stress_method': ms,
                'damage_method': dm, 'size_surface_standard': ss,
                'rz_level': rz_key,
                'Nf': float(Nf) if np.isfinite(Nf) else float('inf'),
                'log10_Nf': float(np.log10(Nf)) if (np.isfinite(Nf) and Nf > 0) else None,
                'sigma_nominal': float(diag['sigma_nominal']),
                'sigma_ar': float(diag['sigma_ar']),
                'sigma_effective': float(diag['sigma_effective']),
            })
        return results
    finally:
        gear_params.LOAD['torque_pinion'] = old_torque
        gear_params.LOAD['tangential_force'] = old_Ft
        Ft0 = 2 * old_torque * 1000 / gear_params.GEAR['pitch_diameter_pinion']
        gear_params.LOAD['Ft_per_unit_facewidth'] = Ft0 / gear_params.GEAR['face_width']

def anova_on_sweep(sweep_results):
    """Compute ANOVA on a sweep result list."""
    valid = [r for r in sweep_results if r['log10_Nf'] is not None]
    y = np.array([r['log10_Nf'] for r in valid])
    n_valid = len(y)
    n_total = len(sweep_results)
    if n_valid < 10:
        return None

    factors = ['sn_source','mean_stress_method','damage_method','size_surface_standard','rz_level']
    n_levels = [2, 4, 2, 3, 3]

    fv = {}
    for f in factors:
        cats = sorted(set(r[f] for r in valid))
        fv[f] = {c:i for i,c in enumerate(cats)}
    X = np.zeros((n_valid, len(factors)), dtype=int)
    for i, r in enumerate(valid):
        for j, f in enumerate(factors):
            X[i,j] = fv[f][r[f]]

    gm = y.mean()
    SSt = np.sum((y - gm)**2)
    SS = {}
    for j, f in enumerate(factors):
        gmeans = {}
        for lv in range(n_levels[j]):
            mk = X[:,j]==lv
            if mk.sum(): gmeans[lv] = y[mk].mean()
        ss = sum((X[:,j]==lv).sum()*(gmeans[lv]-gm)**2 for lv in gmeans)
        SS[f] = float(ss)

    SSr = SSt - sum(SS.values())
    df_res = n_valid - 1 - sum(n_levels) + len(factors)
    MSr = SSr / df_res if df_res > 0 else 1.0

    table = []
    for j, f in enumerate(factors):
        df_f = n_levels[j] - 1
        MSf = SS[f] / df_f
        Fv = MSf / MSr
        pv = 1.0 - f_dist.cdf(Fv, df_f, df_res)
        table.append({'factor':f, 'variance_frac_%': round(100*SS[f]/SSt,1),
                      'F': round(Fv,1), 'p': float(pv)})

    # Interaction
    j1, j2 = factors.index('size_surface_standard'), factors.index('rz_level')
    cell_means = {}
    for l1 in range(n_levels[j1]):
        for l2 in range(n_levels[j2]):
            mk = (X[:,j1]==l1) & (X[:,j2]==l2)
            if mk.sum() > 0:
                cell_means[(l1,l2)] = (y[mk].mean(), mk.sum())
    SS_cell = sum(n*(mv-gm)**2 for (mv,n) in cell_means.values())
    SS_int = max(SS_cell - SS['size_surface_standard'] - SS['rz_level'], 0)
    df_int = (n_levels[j1]-1)*(n_levels[j2]-1)
    MS_int = SS_int / df_int
    F_int = MS_int / MSr
    p_int = 1.0 - f_dist.cdf(F_int, df_int, df_res)
    table.append({'factor':'standard_x_rz','variance_frac_%': round(100*SS_int/SSt,1),
                  'F': round(F_int,1), 'p': float(p_int)})

    return {
        'torque': gear_params.LOAD['torque_pinion'],
        'sigma_F0': float(valid[0]['sigma_nominal']),
        'n_total': n_total, 'n_valid': n_valid, 'n_runout': n_total - n_valid,
        'spread_dex': round(float(max(y)-min(y)), 2),
        'SS_total': float(SSt), 'SS_residual': float(SSr),
        'residual_%': round(100*SSr/SSt, 1),
        'anova_table': table
    }

# Run
print("Multi-torque sweep (500, 600, 700 N-m)")
torque_results = {}
for T in [500, 600, 700]:
    print(f"  T={T} Nm...", end=' ')
    sweep = run_sweep_at_torque(T)
    anova = anova_on_sweep(sweep)
    if anova:
        torque_results[str(T)] = anova
        print(f"OK: {anova['n_valid']}/{anova['n_total']} finite, spread={anova['spread_dex']} dex")
    else:
        print(f"FAILED")

print("\n=== FACTOR RANKING BY TORQUE ===")
header = f"{'Factor':<28} {'500Nm':>8} {'600Nm':>8} {'700Nm':>8}"
print(header)
print("-"*56)
for row in torque_results['700']['anova_table']:
    fname = row['factor']
    vals = []
    for T in ['500','600','700']:
        match = [r for r in torque_results[T]['anova_table'] if r['factor']==fname]
        vals.append(f"{match[0]['variance_frac_%']:.1f}%" if match else "N/A")
    print(f"{fname:<28} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8}")

print(f"\n{'Residual %':<28} ", end='')
for T in ['500','600','700']:
    print(f"{torque_results[T]['residual_%']:.1f}%".rjust(8), end=' ')
print()

# Save
with open('output/multi_torque_anova.json', 'w') as f:
    json.dump(torque_results, f, indent=2)
print("\nSaved output/multi_torque_anova.json")
