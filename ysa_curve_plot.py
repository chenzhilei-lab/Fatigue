"""
ysa_curve_plot.py — Y_Sa sensitivity curve (B4 / 0805意见02 二.1)
=================================================================
Plots the size-and-surface standard variance share vs. the Y_Sa value
applied to AGMA/FKM entries (ISO keeps the reference Y_Sa=1.55), using
the already-computed full AFT decompositions in output/ysa_sensitivity.json
(perturbation factors 0.75x-1.33x, i.e., Y_Sa 1.16-2.06).

Output: output/fig_ysa_sensitivity_curve.png
"""

import json
import os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplcache"))

with open("output/ysa_sensitivity.json") as f:
    d = json.load(f)

baseline_ysa = d["baseline_ysa"]  # 1.55
factors = [float(k) for k in d["sensitivity"].keys()]
factors.sort()
ysa_vals = [baseline_ysa * fac for fac in factors]
sens = d["sensitivity"]
def share_for(fac):
    for k, v in sens.items():
        if abs(float(k) - fac) < 1e-9:
            return v["variance_fractions"]["size_surface_standard"]
    raise KeyError(fac)
std_shares = [share_for(fac) for fac in factors]
base_share = d["summary"]["baseline_standard_variance_%"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.plot(ysa_vals, std_shares, "o-", color="#2166ac", lw=2, ms=6)
ax.axhline(base_share, color="0.45", linestyle="--", lw=1.2)
ax.text(1.62, base_share + 1.2, f"reference design\nY$_{{Sa}}$=1.55: {base_share:.1f}%",
        fontsize=8.5, color="0.3")

# Highlight the ±12.3 pp band claimed in the text (Y_Sa 1.32-1.78)
lo = baseline_ysa * 0.85
hi = baseline_ysa * 1.15
ax.axvspan(lo, hi, color="0.93", alpha=0.8)
ax.text((lo+hi)/2, 3, f"Y$_{{Sa}}$={lo:.2f}-{hi:.2f}\n(0.85x-1.15x)",
        ha="center", fontsize=8, color="0.35")

ax.set_xlabel("Y$_{Sa}$ applied to AGMA/FKM entries (ISO fixed at 1.55)")
ax.set_ylabel("Size-and-surface standard\nvariance share (%)")
ax.set_title("Y$_{Sa}$ baseline sensitivity of the standard-factor share")
ax.set_ylim(0, 70)
ax.grid(alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
plt.tight_layout()
fig.savefig("output/fig_ysa_sensitivity_curve.png", dpi=300)
plt.close()

print("Y_Sa sweep: share ranges "
      f"{min(std_shares):.1f}%-{max(std_shares):.1f}% "
      f"across Y_Sa {min(ysa_vals):.2f}-{max(ysa_vals):.2f}")
print("Saved output/fig_ysa_sensitivity_curve.png")
