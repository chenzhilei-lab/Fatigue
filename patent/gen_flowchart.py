"""Generate patent flowchart for gear fatigue UQ method."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
cn_font = FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
import matplotlib.patches as mpatches

fig, ax = plt.subplots(1, 1, figsize=(8, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

box_style = dict(boxstyle='round,pad=0.5', facecolor='#E8F0FE', edgecolor='#1a56db', linewidth=1.5)
arrow_props = dict(arrowstyle='->', color='#333', lw=1.5)

boxes = [
    (5, 13, 'S1\n获取齿轮几何参数、材料\n参数和载荷条件'),
    (5, 11.5, 'S2\n确定 N 个工程设计选择\n作为方差来源因子'),
    (5, 9.8, 'S3\n全因子遍历：对每个因子水平\n组合计算齿根疲劳寿命'),
    (5, 8.0, 'S3-1\n齿根名义弯曲应力 σ_F0\n(统一 Y_Sa 系数)'),
    (5, 6.5, 'S3-2\n平均应力修正\n(Goodman/Gerber/Morrow/SWT)'),
    (5, 5.0, 'S3-3\n尺寸与表面修正\n(ISO/AGMA/FKM + Rz等级)'),
    (5, 3.5, 'S3-4\nS-N 曲线预测 Nf_raw'),
    (5, 2.0, 'S3-5\n累积损伤法则输出最终 Nf'),
    (5, 0.3, 'S4\n输入 AFT 生存模型\n(右删失处理 run-out 记录)'),
]

# Draw boxes
for x, y, text in boxes:
    ax.text(x, y, text, ha='center', va='center', fontsize=8,
            bbox=box_style, fontproperties=cn_font)

# Side box for S5 + S6
side_style = dict(boxstyle='round,pad=0.5', facecolor='#FCE8D5', edgecolor='#e67e22', linewidth=1.5)
ax.text(5, -1.2, 'S5\n删除单一因子比较 ΔlogLik\n分解方差贡献百分比\n+ Bootstrap 2000 次验证', 
        ha='center', va='center', fontsize=8, bbox=side_style, fontproperties=cn_font)

out_style = dict(boxstyle='round,pad=0.5', facecolor='#D5F5E3', edgecolor='#27ae60', linewidth=2)
ax.text(5, -2.5, 'S6\n输出各因子方差贡献排序\n标识最大不确定性来源', 
        ha='center', va='center', fontsize=9, bbox=out_style, fontproperties=cn_font,
        fontweight='bold')

# Draw arrows
ax.annotate('', xy=(5, 12.5), xytext=(5, 10.5),
            arrowprops=arrow_props)
ax.annotate('', xy=(5, 9.3), xytext=(5, 8.5),
            arrowprops=arrow_props)
ax.annotate('', xy=(5, 7.5), xytext=(5, 7.0),
            arrowprops=arrow_props)
ax.annotate('', xy=(5, 6.0), xytext=(5, 5.5),
            arrowprops=arrow_props)
ax.annotate('', xy=(5, 4.5), xytext=(5, 4.0),
            arrowprops=arrow_props)
ax.annotate('', xy=(5, 1.5), xytext=(5, 1.0),
            arrowprops=arrow_props)
ax.annotate('', xy=(5, -0.2), xytext=(5, -0.8),
            arrowprops=arrow_props)
ax.annotate('', xy=(5, -1.7), xytext=(5, -2.0),
            arrowprops=arrow_props)

# Loop arrow for S3 sub-steps
ax.annotate('', xy=(8.0, 9.5), xytext=(8.0, 2.5),
            arrowprops=dict(arrowstyle='->', color='#666', lw=1, linestyle='dashed'))
ax.text(8.8, 6.0, '144\n组合', ha='center', va='center', fontsize=7, color='#666')

# Title
ax.text(5, 14.2, '齿轮弯曲疲劳寿命不确定性量化方法 — 流程图',
        ha='center', va='center', fontsize=12, fontweight='bold',
        fontproperties=cn_font)

# Patent info
ax.text(5, 14.8, '发明专利附图', ha='center', va='center', fontsize=8,
        color='#999', fontproperties=cn_font)

plt.tight_layout(pad=0.5)
plt.savefig('/mnt/d/Papers/0723-Gear_Fatigue_UQ/patent/flowchart.pdf', dpi=200, bbox_inches='tight')
plt.savefig('/mnt/d/Papers/0723-Gear_Fatigue_UQ/patent/flowchart.png', dpi=200, bbox_inches='tight')
print('Done: flowchart.pdf + flowchart.png')
