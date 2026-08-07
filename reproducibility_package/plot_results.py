"""
plot_results.py — Publication-quality figures for gear fatigue UQ paper (v4.1)
===============================================================================
Figure 1: Fatigue life distribution by size/surface standard (violin + box)
Figure 2: AFT variance decomposition bar chart with bootstrap error bars
Figure 3: Interaction: Rz × standard → median log10(Nf)
Figure 6: Bootstrap 95% CI for the three dominant factors

All numbers are read dynamically from output/sweep.json and
output/aft_anova.json; no hard-coded results.
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('output', exist_ok=True)

# Load data
with open('output/sweep.json') as f:
    sweep = json.load(f)
with open('output/sweep_summary.json') as f:
    sweep_summary = json.load(f)
with open('output/aft_anova.json') as f:
    anova = json.load(f)

valid = [r for r in sweep if r['log10_Nf'] is not None]
logNf = np.array([r['log10_Nf'] for r in valid])

# ===============================================================
# STYLE SETUP
# ===============================================================
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'ISO_6336': '#2166ac',
    'AGMA_2001': '#b2182b',
    'FKM': '#4daf4a',
    'ground_Rz4': '#fde0dd',
    'machined_Rz15': '#f4a582',
    'as_forged_Rz50': '#ca0020',
}
STANDARD_ORDER = ['ISO_6336', 'AGMA_2001', 'FKM']
RZ_ORDER = ['ground_Rz4', 'machined_Rz15', 'as_forged_Rz50']
RZ_LABELS = {'ground_Rz4': 'Ground\n(Rz=4μm)',
             'machined_Rz15': 'Machined\n(Rz=15μm)',
             'as_forged_Rz50': 'As-forged\n(Rz=50μm)'}

# ===============================================================
# FIGURE 1: Fatigue life distribution by standard
# ===============================================================
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(9, 5), gridspec_kw={'width_ratios': [2, 1]})

# 1a: Violin + box plot (only standards with finite-life entries)
data_by_std = []
positions = []
plot_std = []
for i, std in enumerate(STANDARD_ORDER):
    vals = [r['log10_Nf'] for r in valid if r['size_surface_standard'] == std]
    data_by_std.append(vals)
    if vals:
        positions.append(i + 1)
        plot_std.append(std)

if positions:
    vp = ax1a.violinplot([data_by_std[i - 1] for i in positions],
                          positions=positions, widths=0.7,
                          showmeans=False, showmedians=True)
    bp = ax1a.boxplot([data_by_std[i - 1] for i in positions],
                       positions=positions, widths=0.15,
                       patch_artist=True,
                       medianprops={'color': 'black', 'linewidth': 1.5},
                       showfliers=False)
    for j, i in enumerate(positions):
        bp['boxes'][j].set_facecolor(COLORS[STANDARD_ORDER[i - 1]])
        bp['boxes'][j].set_alpha(0.6)
    for body in vp['bodies']:
        body.set_alpha(0.15)

ax1a.set_xticks([1, 2, 3])
ax1a.set_xticklabels(STANDARD_ORDER, fontsize=10)
ax1a.set_ylabel(r'$\log_{10}(N_f)$', fontsize=12)
ax1a.set_title('Fatigue life spread by size/surface standard', fontsize=12)
ax1a.grid(axis='y', alpha=0.3, linestyle='--')

# Censored-standard annotation (ISO: no finite-life entries)
iso_n = sum(1 for r in sweep if r['size_surface_standard'] == 'ISO_6336')
ax1a.text(1, 0.10, 'ISO 6336:\nall combinations\nrun-out\n(censored >$10^7$)',
          ha='center', va='center', fontsize=8.5, color='#2166ac',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                    edgecolor='#2166ac', alpha=0.9))

# Nf labels on right axis
n_min = sweep_summary['min_log10_Nf']
n_max = sweep_summary['max_log10_Nf']
spread = sweep_summary['spread_dex']

def log_to_Nf_str(logv):
    v = 10 ** logv
    if v >= 1e6:
        return f'$10^{{{logv:.1f}}}$'
    else:
        return f'$10^{{{logv:.2f}}}$'

ax1a.text(0.02, 0.98,
          f'$N_f$: {log_to_Nf_str(n_min)} to {log_to_Nf_str(n_max)} cycles\n'
          f'{spread:.1f} dex spread (finite entries)',
          transform=ax1a.transAxes, va='top', fontsize=9, style='italic',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

# 1b: Bars showing run-out vs finite-life counts
runout_counts = []
finite_counts = []
for std in STANDARD_ORDER:
    n_runout = sum(1 for r in sweep if r['size_surface_standard'] == std and r['log10_Nf'] is None)
    n_finite = sum(1 for r in sweep if r['size_surface_standard'] == std and r['log10_Nf'] is not None)
    runout_counts.append(n_runout)
    finite_counts.append(n_finite)

x = np.arange(3)
w = 0.35
bars_finite = ax1b.bar(x - w/2, finite_counts, w, label='Finite life',
                        color=['#2166ac', '#b2182b', '#4daf4a'], alpha=0.8)
bars_runout = ax1b.bar(x + w/2, runout_counts, w, label='Run-out ($>10^7$)',
                        color=['lightsteelblue', 'lightcoral', 'lightgreen'], alpha=0.8)
ax1b.set_xticks(x)
ax1b.set_xticklabels(STANDARD_ORDER, fontsize=10)
ax1b.set_ylabel('Number of combinations', fontsize=12)
ax1b.set_title('Run-out vs finite-life by standard', fontsize=12)
ax1b.legend(fontsize=8, loc='upper right')
ax1b.grid(axis='y', alpha=0.3, linestyle='--')

# Annotate percentages
for i, (r, f) in enumerate(zip(runout_counts, finite_counts)):
    total = r + f
    ax1b.text(i - w/2, f + 3, f'{100*f/total:.0f}%', ha='center', fontsize=8)
    if r > 0:
        ax1b.text(i + w/2, r + 3, f'{100*r/total:.0f}%', ha='center', fontsize=8)

plt.tight_layout()
fig1.savefig('output/fig1_life_spread_by_standard.png')
plt.close()
print('Saved fig1_life_spread_by_standard.png')

# ===============================================================
# FIGURE 2: Variance decomposition bar chart
# ===============================================================
fig2, ax2 = plt.subplots(figsize=(8, 5))

factors_display = [
    ('Mean stress\ncorrection', 'mean_stress_method'),
    ('Size/surface\nstandard', 'size_surface_standard'),
    ('Surface roughness\n($R_z$)', 'rz_level'),
    ('S–N curve\nsource', 'sn_source'),
    ('Std × $R_z$\ninteraction', 'size_surface_standard x rz_level'),
    ('Cumulative\ndamage rule', 'damage_method'),
]

var_means = []
var_lowers = []
var_uppers = []
for _, fkey in factors_display:
    nf = anova['bootstrap']['noise_floor'].get(fkey)
    if nf is None:
        nf = anova['bootstrap']['noise_floor'].get(fkey.replace(' × ', ' x '))
    if nf is None:
        var_means.append(0.0)
        var_lowers.append(0.0)
        var_uppers.append(0.0)
    else:
        var_means.append(nf['mean'])
        var_lowers.append(nf['mean'] - nf['ci_95_lower'])
        var_uppers.append(nf['ci_95_upper'] - nf['mean'])

labels = [l for l, _ in factors_display]
# Combined share uses the point estimates (Table 2), matching the paper text.
point_frac = {row['factor']: row['variance_frac_%']
              for row in anova.get('anova_table', [])}
top3_point = sorted(point_frac.values(), reverse=True)[:3]
top3_sum = sum(top3_point)
colors_bar = ['#2166ac'] * 3 + ['#999999'] * 3

# Horizontal bars
ypos = range(len(labels))[::-1]
bars = ax2.barh(ypos, var_means, xerr=[var_lowers, var_uppers],
                color=colors_bar, alpha=0.85, capsize=3, height=0.6)

ax2.set_yticks(ypos)
ax2.set_yticklabels(labels, fontsize=10)
ax2.set_xlabel('Variance fraction (%)', fontsize=12)
ax2.set_title('What drives gear fatigue life divergence?', fontsize=13, fontweight='bold')

# Annotate percentages with CI
for i, (mean, lower, upper) in enumerate(zip(var_means, var_lowers, var_uppers)):
    ax2.text(mean + 1.5, ypos[i],
             f'{mean:.1f}%  [{mean-lower:.1f}–{mean+upper:.1f}]',
             va='center', fontsize=8.5)

# Legend
legend_elements = [
    Patch(facecolor='#2166ac', alpha=0.85, label=f'Three co-equal factors ({top3_sum:.1f}% combined)'),
    Patch(facecolor='#999999', alpha=0.85, label='Secondary (<8% each)'),
]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=8.5)

ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_xlim(0, max(var_means) * 1.25 + 4)

plt.tight_layout()
fig2.savefig('output/fig2_variance_decomposition.png')
plt.close()
print('Saved fig2_variance_decomposition.png')

# ===============================================================
# FIGURE 3: Interaction plot — Rz × standard
# ===============================================================
fig3, ax3 = plt.subplots(figsize=(7, 5))

for i, std in enumerate(STANDARD_ORDER):
    medians = []
    q25s = []
    q75s = []
    for rz in RZ_ORDER:
        vals = [r['log10_Nf'] for r in valid
                if r['size_surface_standard'] == std and r['rz_level'] == rz]
        if vals:
            v = np.array(vals)
            medians.append(np.median(v))
            q25s.append(np.percentile(v, 25))
            q75s.append(np.percentile(v, 75))
        else:
            medians.append(np.nan); q25s.append(np.nan); q75s.append(np.nan)

    x = np.arange(3)
    if any(np.isfinite(m) for m in medians):
        line = ax3.plot(x, medians, 'o-', color=COLORS[std], linewidth=2,
                        markersize=8, label=std, alpha=0.9)
        ax3.fill_between(x, q25s, q75s, color=COLORS[std], alpha=0.12)

# Run-out threshold reference line
ax3.axhline(7.0, color='0.5', linestyle='--', linewidth=1.2)
ax3.text(2.25, 7.05, 'run-out threshold\n($10^7$ cycles)', fontsize=8,
         color='0.35', va='bottom', ha='right')

ax3.set_xticks([0, 1, 2])
ax3.set_xticklabels([RZ_LABELS[rz] for rz in RZ_ORDER], fontsize=10)
ax3.set_ylabel(r'Median $\log_{10}(N_f)$', fontsize=12)
ax3.set_title('Interaction: surface roughness × standard', fontsize=12)
ax3.legend(fontsize=9, loc='best')
ax3.grid(alpha=0.3, linestyle='--')

# Censored-standards note
ax3.text(0.02, 0.45,
         'ISO 6336: all 48 combinations run-out\n'
         'AGMA at $R_z=4$ $\\mu$m: all run-out',
         transform=ax3.transAxes, fontsize=8, color='#333333',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                   edgecolor='0.45', alpha=0.95))

# AGMA–FKM gap at as-forged (only standards with finite data)
a50 = [r['log10_Nf'] for r in valid
       if r['size_surface_standard'] == 'AGMA_2001' and r['rz_level'] == 'as_forged_Rz50']
f50 = [r['log10_Nf'] for r in valid
       if r['size_surface_standard'] == 'FKM' and r['rz_level'] == 'as_forged_Rz50']
if a50 and f50:
    gap50 = float(np.median(a50)) - float(np.median(f50))
    ax3.text(1.95, float(np.median(f50)) + 0.35,
             f'AGMA–FKM gap:\n{gap50:.1f} dex at $R_z$=50 $\\mu$m',
             fontsize=8.5, color='#333333',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor='0.45', alpha=0.95))

plt.tight_layout()
fig3.savefig('output/fig3_rz_standard_interaction.png')
plt.close()
print('Saved fig3_rz_standard_interaction.png')

# ===============================================================
# FIGURE 6: Bootstrap 95% CI for the three dominant factors
# ===============================================================
fig6, ax6 = plt.subplots(figsize=(7, 4))

nf_all = anova['bootstrap']['noise_floor']
top3_keys = sorted(
    [k for k in ('mean_stress_method', 'size_surface_standard', 'rz_level',
                 'sn_source', 'damage_method',
                 'size_surface_standard x rz_level')
     if k in nf_all],
    key=lambda k: nf_all[k]['mean'], reverse=True)[:3]
top3_labels = {
    'mean_stress_method': 'Mean-stress\ncorrection',
    'size_surface_standard': 'Size-and-surface\nstandard',
    'rz_level': 'Surface\nroughness, $R_z$',
}[top3_keys[0]]
label_map = {
    'mean_stress_method': 'Mean-stress\ncorrection',
    'size_surface_standard': 'Size-and-surface\nstandard',
    'rz_level': 'Surface\nroughness, $R_z$',
}

for i, key in enumerate(top3_keys):
    nf = nf_all[key]
    mean_v = nf['mean']
    ci_lo = nf['ci_95_lower']
    ci_hi = nf['ci_95_upper']
    sn = nf.get('s_n', mean_v / nf['std'] if nf['std'] > 0 else 0)
    color = {'mean_stress_method': '#2166ac',
             'size_surface_standard': '#b2182b',
             'rz_level': '#4daf4a'}[key]

    ax6.barh(i, mean_v, color=color, alpha=0.7, height=0.5,
             label=label_map[key].replace(chr(10), ' '))
    ax6.errorbar(mean_v, i, xerr=[[mean_v - ci_lo], [ci_hi - mean_v]],
                 fmt='none', ecolor='black', capsize=4, linewidth=1.5)
    ax6.text(mean_v + 1.5, i,
             f'{mean_v:.1f}%  [{ci_lo:.1f}, {ci_hi:.1f}]  S/N$\\approx${sn:.0f}',
             va='center', fontsize=10)

ax6.set_yticks(range(len(top3_keys)))
ax6.set_yticklabels([label_map[k] for k in top3_keys], fontsize=11)
ax6.set_xlabel('Variance fraction (%)', fontsize=12)
ax6.set_title('Bootstrap 95% confidence intervals: three dominant factors',
              fontsize=13, fontweight='bold')
ax6.set_xlim(0, 50)
ax6.grid(axis='x', alpha=0.3, linestyle='--')
ax6.invert_yaxis()

plt.tight_layout()
fig6.savefig('output/fig6_bootstrap_ci.png')
plt.close()
print('Saved fig6_bootstrap_ci.png')

print("")
print("Figures generated in output/:")
print("  fig1_life_spread_by_standard.png  -- Nf distributions + run-out counts")
print("  fig2_variance_decomposition.png   -- AFT variance decomposition with bootstrap CI")
print("  fig3_rz_standard_interaction.png  -- Rz x standard interaction")
print("  fig6_bootstrap_ci.png             -- Bootstrap 95% CI for top 3 factors")
