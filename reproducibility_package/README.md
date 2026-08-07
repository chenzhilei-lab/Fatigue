# Gear Fatigue UQ — Reproducibility Package (v5.1)

**Paper:** Method-Induced Uncertainty in Gear Bending Fatigue: Factorial Decomposition of Size-and-Surface Factor Sensitivity Under a Common ISO Stress Baseline  
**Author:** Zhilei Chen (Guangdong Peizheng College)  
**Target:** *Journal of Mechanical Science and Technology* (KSME/Springer)

## Quick Start

```bash
bash reproduce.sh
```

Runs 12 steps: sweep → AFT decomposition → sensitivity analyses → Monte Carlo
validation → figures → labels/audit → canonical/MANIFEST check. ~20 minutes (mostly AFT bootstrap and the
500-sim Monte Carlo validation).

## Requirements

- Python 3.10+
- numpy, scipy, matplotlib

## File Tree

```
├── reproduce.sh              # One-click reproduction (v5.1)
├── README.md
├── generate_labels_v3.py     # Rebuild output/labels.json + number-source ledger
├── phase0_canonical.py    # Canonical numbers + MANIFEST (deterministic anchor)
├── run_sweep.py              # 144-combination full-factorial sweep
├── aft_variance_decomposition.py # AFT censored-likelihood (EM, 2000 draws)
├── ka_sensitivity.py         # K_A load-factor sensitivity
├── ysa_sensitivity.py        # Y_Sa asymmetry sensitivity
├── native_stress_sensitivity.py # Native stress-chain sensitivity
├── native_bias_propagation.py   # Native bias propagation
├── threshold_sensitivity.py  # Run-out threshold sensitivity
├── multi_torque_aft.py       # Torque dependence: 500/600/700 Nm (AFT)
├── multi_aft_comparison.py   # Distribution family comparison (log-normal/log-logistic)
├── aft_residual_diagnostics.py # Cox-Snell/Martingale residuals + Shapiro-Wilk
├── benchmark_wang2023.py     # 42CrMo4 bending-fatigue literature benchmark
├── run_weibull_comparison.py # Weibull vs log-normal AIC comparison
├── run_sobol_comparison.py   # Sobol' first-order vs AFT LR fractions
├── multi_sn_sensitivity.py   # Multi S-N curve sensitivity (5 heat treatments)
├── sobol_censoring_gradient.py # Sobol'/ANOVA bias vs censoring gradient
├── simulation_validation.py  # Monte Carlo validation of the decomposition
├── two_level_variable_amplitude.py # Two-level block-spectrum illustration (Fig. 7)
├── plot_results.py           # Publication figures (AFT-based)
├── paper_check.py            # Universal paper auditor
├── fatigue_models.py         # ISO 6336 / AGMA / FKM implementations
├── gear_params.py            # Fixed gear geometry + material
├── 数字来源-齿轮疲劳v4.8.csv/.xlsx  # Number-source ledger (every key number + source)
├── paper/
│   ├── gear_fatigue_v4.0.tex # LaTeX source (Basquin-consistent 610/610 baseline)
│   ├── gear_fatigue_v4.0.pdf # Compiled PDF
│   ├── gear_fatigue_v4.3.tex # v4.3: common-σ rerun + 0805 reviewer fixes
│   ├── gear_fatigue_v4.3.pdf
│   ├── gear_fatigue_v4.4.tex # v4.4: + independent 42CrMo4 S–N anchor (§4.3, Table 3)
│   ├── gear_fatigue_v4.4.pdf
│   ├── gear_fatigue_v4.5.tex # v4.5: + two-level spectrum Fig. 7, 0805-medium fixes
│   ├── gear_fatigue_v4.5.pdf
│   ├── gear_fatigue_v4.6.tex # v4.6: v4.5 + later reviewer/priority fixes
│   ├── gear_fatigue_v4.6.pdf
│   ├── gear_fatigue_v4.7.tex # v4.7: converged 0805 final state (B/C priority list)
│   ├── gear_fatigue_v4.7.pdf
│   ├── gear_fatigue_v4.8.tex # v4.8: fix duplicate fig8, @source captions, Y_Sa math, "a a"
│   ├── gear_fatigue_v4.8.pdf
│   ├── gear_fatigue_v4.9.tex # v4.9: comment-only + % @source tags for paper_check Layer 1
│   ├── gear_fatigue_v4.9.pdf
│   ├── gear_fatigue_v5.0.tex # v5.0: retarget JMST + Declaration of Generative AI Use
│   ├── gear_fatigue_v5.0.pdf
│   ├── gear_fatigue_v5.1.tex # v5.1: final JMST submission — Claude polish, overfull fixed
│   └── gear_fatigue_v5.1.pdf
│   ├── check_compile.sh      # 编译四查: hard errors / undefined / overfull / table
│   ├── ref_papers/
│   │   ├── ulrich2025_42CrMo4_s10010-025-00818-x.pdf  # open-access anchor data
│   │   ├── bonaiti2024_ijf.pdf / pagliari2025_forschung.pdf / vietze2024_machines.pdf
│   │   ├── wang2023_jxqd_42CrMo.pdf / fatemi1998_ijf.pdf / glodez2002_ijf.pdf
│   │   ├── dowling2009_ffems.pdf
│   │   ├── Iter_066.pdf + .docx / Iter_068.pdf + .docx  # Waterloo SMDIdbase records
│   │   └── 材料力学性能测试数据表.xlsx
│   └── 审稿问题清单_v4.0.txt  # Audit checklist (A–E sections + v4.2 notes)
└── output/
    ├── sweep.json            # Raw 144 results
    ├── sweep_summary.json    # Summary stats
    ├── aft_anova.json        # AFT variance decomposition + bootstrap (2000 draws)
    ├── aft_diagnostics.json  # Cox-Snell/Martingale/Shapiro-Wilk/Levene
    ├── ka_sensitivity.json   # K_A sensitivity
    ├── ysa_sensitivity.json  # Y_Sa sensitivity
    ├── native_stress_sensitivity.json
    ├── native_bias_propagation.json
    ├── threshold_sensitivity.json
    ├── multi_torque_aft.json # Torque-dependent results
    ├── multi_aft_comparison.json
    ├── sobol_vs_aft.json     # Sobol' first-order comparison
    ├── two_level_variable_amplitude.json # Two-level block-spectrum AFT result
    ├── multi_sn_sensitivity.json # S-N share 2.5% → 34.7% (5 heat treatments)
    ├── sobol_censoring_gradient.json # gap 0 → 40 pp across censoring
    ├── simulation_validation.json # 500-sim MC recovery + coverage + Sobol' R²
    ├── weibull_aft.json      # Log-normal vs Weibull AIC comparison
    ├── benchmark_wang2023.json # Wang 2023 test-gear benchmark
    ├── labels.json           # paper_check label registry (141 labels)
    ├── canonical_numbers.json # canonical numbers registry (141 numbers)
    ├── MANIFEST.txt          # deterministic MD5 inventory of output/
    ├── fig1_life_spread_by_standard.png
    ├── fig2_variance_decomposition.png
    ├── fig3_rz_standard_interaction.png
    ├── fig6_bootstrap_ci.png
    ├── fig7_two_level_comparison.png
    ├── fig8_multi_sn_sensitivity.png
    ├── fig9_sobol_censoring_gradient.png
    ├── fig_diag_qq.png / fig_ka_sensitivity.png / fig_sim_validation.png
    └── fig_ysa_sensitivity.png / fig_ysa_sensitivity_curve.png
```

## Methodology

| Switch | Options | Count |
|--------|---------|:-----:|
| S-N curve source | Waterloo #66, #68 | 2 |
| Mean stress correction | Goodman, Gerber, Morrow, SWT | 4 |
| Cumulative damage rule | Miner, Modified Miner | 2 |
| Size & surface standard | ISO 6336, AGMA 2001, FKM | 3 |
| Surface roughness Rz | 4, 15, 50 μm | 3 |

Fixed: module 3 mm, 20 teeth, 42CrMo4, 700 N·m, R=0

## Key Results (AFT Censored-Likelihood, EM Estimation, common-σ)

The drop-one-factor log-likelihood decomposition evaluates every reduced model
with the **full-model σ** (common-σ). This makes the likelihood-ratio shares
mathematically equivalent to Sobol' first-order indices on the deterministic
life model (verified numerically, R² = 0.9933), instead of the refit-σ shortcut
used in v4.0 (which saturated small effects and compressed the ranking).

| Factor | ΔlogLik | Var.% | Bootstrap Mean | CI95 | S/N |
|--------|:-------:|:-----:|:--------------:|------|:---:|
| Size/surface standard | 985.7 | 55.3% | 55.6% | [48.7, 63.0] | ~15 |
| Surface roughness, Rz | 403.3 | 22.6% | 22.3% | [16.9, 27.8] | ~8 |
| Mean stress correction | 324.2 | 18.2% | 18.2% | [10.8, 25.0] | ~5 |
| S-N curve source | 43.9 | 2.5% | 2.4% | [1.3, 3.7] | ~4 |
| Standard × Rz | 20.5 | 1.2% | 1.1% | [0.6, 1.8] | ~4 |
| Cumulative damage rule | 5.9 | 0.3% | 0.4% | [0.1, 0.8] | ~2 |

Three factors account for 96.1% of total log-variance. σ_aft = 0.175.  
Nf range (finite-life entries): 1.3×10³ – 8.3×10⁵ cycles (2.79 dex);
84 of 144 entries are run-outs (≥10⁶ cycles). Bootstrap: 2000 draws,
seed 42.

### Monte Carlo validation

500 synthetic datasets (balanced 144-design, log-normal AFT, 30% censoring)
recover the true factor shares with |bias| ≤ 3.4 pp and top-3 rank recovery of
98.2% (Spearman ρ = 0.973). Bootstrap 95% intervals achieve 92–98.7% coverage.
Censoring-envelope test: at the paper's observed 58% censoring the dominant
factor share is biased low by ~7 pp, so 55.3% is a conservative lower bound.

## Formula Verification

All implementations verified against original standards: ISO 6336-3:2019,
AGMA 2001-D04, FKM 7th ed. (standard PDFs in the project-level `Standard/`
directory). S-N parameters from publicly accessible Waterloo SMDIdbase
(Iter_066/068 PDFs in `paper/ref_papers/`).

## Audit

```bash
python generate_labels_v3.py        # rebuild output/labels.json + number-source ledger
python phase0_canonical.py --check   # verify canonical numbers + MANIFEST (deterministic)
python paper_check.py paper/gear_fatigue_v5.1.tex
# → [PASS] Layer 0 lineage OK; Layer 1: 34 MATCH / 0 MISMATCH; Layer 2: 270 traced / 0 orphan
bash paper/check_compile.sh gear_fatigue_v5.1
# → compile hygiene: 0 hard errors / 0 undefined / overfull < 20 pt
```

Number-source ledger (every key number + source): `数字来源-齿轮疲劳v4.8.csv`
and `.xlsx` in the package root (regenerate with `python generate_labels_v3.py`).
