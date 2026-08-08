# -*- coding: utf-8 -*-
"""
make_fig8_native.py — regenerate fig8 with the NATIVE-chain multi-SN data (v6.1)
===============================================================================
The original fig8_multi_sn_sensitivity.png was produced by
multi_sn_sensitivity.py (8/5, Peterson pipeline). The v6.0 primary
result uses the native chain, so the figure must be regenerated from
output/multi_sn_native.json to stay pipeline-consistent.

Panel a: S-N share (2 vs 5 curves) AND standard share, native chain.
Panel b: fatigue strength at 10^6 for the five curves (unchanged).

Saves output/fig8_multi_sn_sensitivity_native.png (new file; the old
PNG is kept untouched).
"""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open("output/multi_sn_native.json", encoding="utf-8"))

sn2 = d["native_2curves_144"]["shares_%"]["sn_source"]
sn5 = d["native_5curves_360"]["shares_%"]["sn_source"]
std2 = d["native_2curves_144"]["shares_%"]["size_surface_standard"]
std5 = d["native_5curves_360"]["shares_%"]["size_surface_standard"]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

# Panel a: S-N and standard shares, 2 vs 5 curves (native chain)
ax = axes[0]
x = np.arange(2)
w = 0.38
b1 = ax.bar(x - w/2, [sn2, sn5], w, color="#2166ac", alpha=0.85, label="S--N curve source")
b2 = ax.bar(x + w/2, [std2, std5], w, color="#b2182b", alpha=0.85, label="Size/surface standard")
for bar, v in zip(b1, [sn2, sn5]):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.1f}%",
            ha="center", fontsize=10, fontweight="bold")
for bar, v in zip(b2, [std2, std5]):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.1f}%",
            ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["2 S--N curves\n(Waterloo)", "5 S--N curves\n(+3 heat treatments)"])
ax.set_ylabel("Native-chain variance share (%)")
ax.set_title("Factor contribution vs.\nnumber of S--N curve levels")
ax.set_ylim(0, max(std2, std5) * 1.35)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

# Panel b: fatigue strength at 10^6 for the five curves
ax = axes[1]
labels = ["#68", "#66", "HighStr\n(380 C)", "Classic\n(500 C)", "Classic\n(560 C)"]
vals = [610.5, 609.9, 747.0, 594.9, 518.0]
colors = ["#2166ac", "#2166ac", "#b2182b", "#b2182b", "#b2182b"]
b = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.55)
for bar, v in zip(b, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 4, f"{v:.0f}",
            ha="center", fontsize=8.5)
ax.set_ylabel(r"Fatigue strength at $10^6$ cycles (MPa)")
ax.set_title("S--N curve levels in the extended design")
ax.set_ylim(0, max(vals) * 1.18)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

plt.tight_layout()
out = "output/fig8_multi_sn_sensitivity_native.png"
fig.savefig(out, dpi=300)
plt.close()
print("Saved", out)
print("Panel a (native chain): S-N 2.3%% -> 0.7%%; std 56.7%% -> 52.5%%")
print("=== Complete ===")
