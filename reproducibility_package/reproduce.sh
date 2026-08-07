#!/bin/bash
# ================================================================
# Gear Fatigue UQ — One-Click Reproducibility (v5.0)
# Usage: bash reproduce.sh
# Requirements: Python 3.10+, numpy, scipy, matplotlib
# Reproduces all JSON outputs used by paper/gear_fatigue_v5.0.tex,
# then verifies the canonical numbers + MANIFEST consistency anchor.
# ================================================================
set -e
cd "$(dirname "$0")"

echo "=== [1/12] Full-factorial sweep (144 combinations) ==="
python run_sweep.py

echo ""
echo "=== [2/12] AFT censored-likelihood decomposition + bootstrap ==="
python aft_variance_decomposition.py

echo ""
echo "=== [3/12] K_A load-factor sensitivity ==="
python ka_sensitivity.py

echo ""
echo "=== [4/12] Y_Sa asymmetry sensitivity ==="
python ysa_sensitivity.py

echo ""
echo "=== [5/12] Native stress-chain sensitivity ==="
python native_stress_sensitivity.py

echo ""
echo "=== [6/12] Native bias propagation ==="
python native_bias_propagation.py

echo ""
echo "=== [7/12] Threshold sensitivity ==="
python threshold_sensitivity.py

echo ""
echo "=== [8/12] Residual diagnostics + multi-torque + distribution comparison ==="
python aft_residual_diagnostics.py
python multi_torque_aft.py
python multi_aft_comparison.py
python run_weibull_comparison.py
python run_sobol_comparison.py
python multi_sn_sensitivity.py
python sobol_censoring_gradient.py

echo ""
echo "=== [9/12] Monte Carlo validation of common-sigma decomposition ==="
python simulation_validation.py

echo ""
echo "=== [10/12] Figures ==="
python plot_results.py

echo ""
echo "=== [11/12] Labels + paper audit ==="
python generate_labels_v3.py
python paper_check.py paper/gear_fatigue_v5.0.tex

echo ""
echo "=== [12/12] Canonical numbers + MANIFEST consistency check ==="
python phase0_canonical.py --check

echo ""
echo "=========================================="
echo " DONE. All outputs in output/"
echo "=========================================="