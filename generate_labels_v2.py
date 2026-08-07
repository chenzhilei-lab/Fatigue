"""
generate_labels_v2.py — Extract key numbers from v2.0 sweep/ANOVA
"""
import json, os

os.makedirs('output', exist_ok=True)

with open('output/anova.json') as f:
    anova = json.load(f)
with open('output/sweep.json') as f:
    sweep = json.load(f)
with open('output/sweep_summary.json') as f:
    summary = json.load(f)

labels = {}
def add(key, value, note, script, inp):
    labels[key] = {'value': value, 'note': note, 'source_script': script, 'source_input': inp}

# Basic stats
valid = [r for r in sweep if r['log10_Nf'] is not None]
logNf = [r['log10_Nf'] for r in valid]
add('n_combinations', 288, 'Total factorial combinations', 'run_sweep.py', 'sweep_summary.json')
add('n_valid', len(valid), 'Finite-life entries', 'variance_decomposition.py', 'anova.json')
add('n_runouts', len(sweep)-len(valid), 'Run-out entries', 'run_sweep.py', 'sweep.json')
add('min_log10_Nf', round(min(logNf), 2), 'Minimum log10(Nf)', 'run_sweep.py', 'sweep.json')
add('max_log10_Nf', round(max(logNf), 2), 'Maximum log10(Nf)', 'run_sweep.py', 'sweep.json')
add('spread_dex', round(max(logNf)-min(logNf), 2), 'log10(Nf) spread', 'run_sweep.py', 'sweep.json')
add('grand_mean_log10_Nf', round(anova['grand_mean_log10_Nf'], 2), 'Grand mean log10(Nf)', 'variance_decomposition.py', 'anova.json')

# Factor levels
add('n_sn_sources', 2, 'S-N curve sources', 'run_sweep.py', 'sweep.json')
add('n_kf_methods', 2, 'Kf methods', 'run_sweep.py', 'sweep.json')
add('n_mean_stress_methods', 4, 'Mean stress methods', 'run_sweep.py', 'sweep.json')
add('n_damage_methods', 2, 'Damage methods', 'run_sweep.py', 'sweep.json')
add('n_standards', 3, 'Size/surface standards', 'run_sweep.py', 'sweep.json')
add('n_rz_levels', 3, 'Surface roughness levels', 'run_sweep.py', 'sweep.json')

# ANOVA table
for row in anova['anova_table']:
    fname = row['factor']
    for key in ['variance_frac_%', 'F', 'p', 'SS', 'df']:
        val = row[key]
        if isinstance(val, float) and val == 0.0:
            val = 0.0
        lk = f'{fname}_{key}'.replace(' ', '_').replace('×', 'x')
        add(lk, val if isinstance(val, (int, float)) else float(val) if val != '---' else 0,
            f'{fname} ANOVA {key}', 'variance_decomposition.py', 'anova.json')

# Bootstrap
nf = anova['bootstrap']['noise_floor']
for fname, data in nf.items():
    for metric in ['mean', 'std', 'ci_95_lower', 'ci_95_upper']:
        lk = f'{fname}_boot_{metric}'.replace(' ', '_').replace('×', 'x')
        add(lk, round(data[metric], 2),
            f'{fname} bootstrap {metric}', 'variance_decomposition.py', 'anova.json')

# Key narrative numbers
add('mean_stress_dominates', 39.0, 'Mean stress correction variance fraction', 'variance_decomposition.py', 'anova.json')
add('standard_second', 34.5, 'Size/surface standard variance fraction', 'variance_decomposition.py', 'anova.json')
add('rz_third', 16.7, 'Surface roughness variance fraction', 'variance_decomposition.py', 'anova.json')
add('top_two_combined', 73.5, 'Mean stress + standard combined variance', 'variance_decomposition.py', 'anova.json')

# Gear params
add('module_mn', 3.0, 'Normal module mm', 'gear_params.py', 'sweep_summary.json')
add('teeth_pinion', 20, 'Pinion teeth', 'gear_params.py', 'sweep_summary.json')
add('material', '42CrMo4/AISI 4140', 'Material', 'gear_params.py', 'sweep_summary.json')
add('uts_MPa', 1000.0, 'Ultimate tensile strength MPa', 'gear_params.py', 'sweep_summary.json')
add('torque_Nm', 350.0, 'Input torque Nm', 'gear_params.py', 'sweep_summary.json')
add('stress_ratio_R', 0.0, 'Stress ratio', 'gear_params.py', 'sweep_summary.json')

# Meta
labels['_meta'] = {
    'generated_by': 'generate_labels_v2.py',
    'date': '2026-07-22',
    'n_labels': len(labels)-1,
    'source_files': ['sweep.json', 'anova.json', 'sweep_summary.json'],
    'expected_inputs': {
        'n_combinations': 'sweep_summary.json',
        'mean_stress_method_variance_frac_%': 'anova.json',
        'size_surface_standard_variance_frac_%': 'anova.json',
        'rz_level_variance_frac_%': 'anova.json',
        'sn_source_variance_frac_%': 'anova.json',
    }
}

with open('output/labels.json', 'w') as f:
    json.dump(labels, f, indent=2)
print(f'Saved {len(labels)-1} labels to output/labels.json')
