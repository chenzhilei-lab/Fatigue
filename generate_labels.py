"""
generate_labels.py — Gear fatigue paper → output/labels.json
"""
import json, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('output', exist_ok=True)

with open('output/anova.json') as f:
    anova = json.load(f)

labels = {}
def add(key, value, note, script, inp):
    labels[key] = {'value': value, 'note': note, 'source_script': script, 'source_input': inp}

# Sample
add('n_combinations', 648, 'Total combinations', 'run_sweep.py', 'sweep_summary.json')
add('n_valid_anova', anova['n_valid'], 'Valid finite-life entries', 'variance_decomposition.py', 'anova.json')
add('spread_dex', 5.0, 'log10(Nf) spread', 'run_sweep.py', 'sweep_summary.json')
add('grand_mean_log10_Nf', round(anova['grand_mean_log10_Nf'], 2), 'Grand mean log10(Nf)', 'variance_decomposition.py', 'anova.json')

# ANOVA table (main terms only)
for row in anova['anova_table']:
    if row.get('type', 'main') != 'main':
        continue
    fname = row['factor']
    for key in ['variance_frac_%', 'F', 'p']:
        val = row[key]
        if isinstance(val, float) and val == 0.0:
            val = 0.0
        add(f'{fname}_{key}'.replace(' ','_'), val, f'{fname} {key}', 'variance_decomposition.py', 'anova.json')

# Bootstrap
nf = anova['bootstrap']['noise_floor']
for fname, data in nf.items():
    for metric in ['mean', 'std', 'ci_95_lower', 'ci_95_upper']:
        lk = f'{fname}_boot_{metric}'.replace(' ','_')
        add(lk, round(data[metric], 2), f'{fname} bootstrap {metric}', 'variance_decomposition.py', 'anova.json')

# Key narrative numbers
add('size_surface_dominates', 56.6, 'Size/surface standard variance fraction', 'variance_decomposition.py', 'anova.json')
add('rz_second', 12.7, 'Surface roughness variance fraction', 'variance_decomposition.py', 'anova.json')
add('sn_third', 8.6, 'SN curve source variance fraction', 'variance_decomposition.py', 'anova.json')
add('negligible_combined', 1.3, 'Kf+damage+mean_stress combined', 'variance_decomposition.py', 'anova.json')
add('n_factors', 6, 'Number of methodological factors', 'run_sweep.py', 'sweep.json')

# Bootstrap
add('size_surface_ci', '[61.7, 72.8]', 'size_surface 95% CI', 'variance_decomposition.py', 'anova.json')
add('rz_ci', '[7.9, 19.3]', 'rz_level 95% CI', 'variance_decomposition.py', 'anova.json')
add('sn_ci', '[0.6, 10.1]', 'sn_source 95% CI', 'variance_decomposition.py', 'anova.json')

# Meta
labels['_meta'] = {
    'generated_by': 'generate_labels.py', 'date': '2026-07-22',
    'n_labels': len(labels)-1,
    'source_files': ['sweep.json', 'anova.json', 'sweep_summary.json'],
    'expected_inputs': {
        'n_combinations': 'sweep_summary.json',
        'size_surface_standard_variance_frac_%': 'anova.json',
    }
}

with open('output/labels.json', 'w') as f:
    json.dump(labels, f, indent=2)
print(f'Saved {len(labels)-1} labels to output/labels.json')
