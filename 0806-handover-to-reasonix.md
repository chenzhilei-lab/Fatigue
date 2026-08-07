# 交接简报 — 齿轮疲劳论文 JMST 投稿准备（给 reasonix）

任务来源：用户决定直接投 JMST（Journal of Mechanical Science and Technology, KSME/Springer）。
本文档是唯一任务依据；执行前先通读，再按 A/B/C 顺序执行。

## 0. 工作目录与关键路径
- 项目根目录：D:\Papers\0723-Gear_Fatigue_UQ
- 稿件目录：D:\Papers\0723-Gear_Fatigue_UQ\paper
- 复现包：D:\Papers\0723-Gear_Fatigue_UQ\reproducibility_package
- 输出数据：D:\Papers\0723-Gear_Fatigue_UQ\output
- 工作标准（铁律，每次投稿前必读）：D:\Papers\工作标准.txt
- 进度记录（含历史决策）：D:\Papers\0723-Gear_Fatigue_UQ\0806-工作进度.txt

## 1. 当前状态（2026-08-06，已全部验证）
- 最新稿件：paper/gear_fatigue_v4.9.tex + gear_fatigue_v4.9.pdf（37 页）
  - 3 遍 pdflatex：0 error / 0 undefined / 0 重复标签 / Overfull>20pt=0
  - paper_check.py paper/gear_fatigue_v4.9.tex → [PASS]（Layer0 OK；Layer1 34 MATCH/0 MISMATCH；Layer2 270 traced/0 orphan）
  - check_compile.sh gear_fatigue_v4.9 → PASS
- 核心数字（与 output/*.json 一致，勿改）：144 组合、84 run-out(58%)、60 finite、
  55.3/22.6/18.2/2.5/1.2/0.3%、bootstrap 2000(seed 42)、Shapiro-Wilk W=0.962 p=0.056、
  ΔAIC=+77.1、Nf 1.3e3–8.3e5（2.79 dex）、MC 500 次、11 图 6 表
- 数字来源表：数字来源-齿轮疲劳v4.8.csv/.xlsx（根目录 + 复现包内）
- output/labels.json（141 标签）+ generate_labels_v3.py
- 复现包已与根目录 MD5 全量一致（脚本 22、output 31、paper v4.0-v4.9 tex/pdf、ref_papers 13）
- 参考文献本地库：paper/ref_papers（8 篇期刊 PDF + Waterloo Iter66/68 + ulrich + xlsx）；
  3 本标准在项目级 Standard/

## 2. 待办任务（按顺序，做完一项记录一项）

### A. Cover Letter 重写（JMST 版）
- 旧版：paper/cover_letter_v4.0.tex（数字过时：92 run-out、39.1/35.3/23.5%、p=0.599、ΔAIC=29.7、7 图 3 表；含 Dowling 2009 引用——正文没有此文）
- 要求：另存新文件 paper/cover_letter_v5.0_jmst.tex（不覆盖旧版），编译 PDF。
  内容要点：
  1) 收件方：Editor-in-Chief, Journal of Mechanical Science and Technology (JMST), Korean Society of Mechanical Engineers
  2) 作者信息（投稿方）：Zhilei Chen, Guangdong Peizheng College, Data and Computer Science, 53 Peizheng Rd., Chini Town, Huadu, Guangzhou, Guangdong 510830, China; E-mail: 2604513@peizheng.edu.cn
  3) 论文题目与正文一致：Method-Induced Uncertainty in Gear Bending Fatigue: Factorial Decomposition of Size-and-Surface Factor Sensitivity Under a Common ISO Stress Baseline
  4) 数字全部用第 1 节的现值（144/84/60、55.3/22.6/18.2/2.5/1.2/0.3、2000 bootstrap seed 42、500 MC、Shapiro p=0.056、ΔAIC=+77.1、11 图 6 表）；删除 Dowling 2009 引用
  5) 必须含 AI 声明一句（原文照抄）："Generative AI tools assisted with LaTeX formatting, Python code implementation, and English language editing."
  6) 不得编造审稿人、不得编造与正文不一致的任何数字；初投稿（非修订回复）
- 编译：cd paper && pdflatex -interaction=nonstopmode cover_letter_v5.0_jmst.tex（两遍）

### B. 稿件 v5.0（= v4.9 + 两处小改）
- 另存 paper/gear_fatigue_v5.0.tex（v4.9 不动）
- 改动 1：头注释 "Target: International Journal of Fatigue (IJF, Elsevier)" → "Target: Journal of Mechanical Science and Technology (JMST, KSME/Springer)"；并在变更日志加 v5.0 行（说明：目标期刊改 JMST + AI 声明）
- 改动 2：在 Data and Code Availability 段之后、thebibliography 之前加一小节：
  \section*{Declaration of Generative AI Use}
  Generative AI tools assisted with LaTeX formatting, Python code implementation, and English language editing.
- 除上述外不得改动正文任何数字/文字（机器味词、长句等只允许评估报告，不允许直接改）
- 编译验证：3 遍 pdflatex 0 error / 0 undefined；bash check_compile.sh gear_fatigue_v5.0 → PASS；
  paper_check.py paper/gear_fatigue_v5.0.tex → PASS

### C. 复现包同步 + 压缩包重建
- 把 gear_fatigue_v5.0.tex/.pdf 复制进 reproducibility_package/paper/
- 复现包 README.md 两处更新：
  1) "**Target:** *International Journal of Fatigue* (Elsevier)" → "**Target:** *Journal of Mechanical Science and Technology* (KSME/Springer)"
  2) paper/ 文件树补 v5.0 两行
- 更新后对复现包与根目录做 MD5 一致性复验（脚本/output/paper tex+pdf/ref_papers），差异必须为 0
- 重建压缩包：Compress-Archive 复现包全部内容 → 项目根目录 gear_fatigue_uq_reproducibility_v5.0_jmst.zip（不覆盖旧的 gear_fatigue_uq_reproducibility.zip，保留旧档）

### D.（只报告不改）机器味/长句复核
- 扫描 v5.0：However ×4、Notably ×2、"In this section" 若干、>40 词长句若干
- 结论写入进度文件即可，不修改正文；如发现必须改的低级错误（拼写/重复词/渲染错误），
  另存 v5.1 修复并跑全验证

## 3. 完成标准（全部满足才算收工）
- cover_letter_v5.0_jmst.tex/.pdf 存在且编译 0 error，数字与正文一致，含 AI 声明
- gear_fatigue_v5.0.tex/.pdf 存在，3 遍编译 0 error / 0 undefined，check_compile PASS，paper_check PASS
- 复现包 MD5 全一致；zip 生成成功（记录大小）
- 把上述所有动作与证据（命令输出片段）追加进 0806-工作进度.txt 的"0806 续 — reasonix 交接执行"段落

## 4. 铁律（违反任何一条视为失败）
- 另存新版，绝不覆盖旧版（v4.9、cover_letter_v4.0 等都保留）
- 不编造任何数字、事实、审稿人、参考文献；所有数字必须能溯源到 output/*.json 或本地 PDF
- 声称"已完成/已验证"前必须贴命令原始输出（grep/diff/pdflatex 片段），禁止只画汇总表
- 正文只允许 B 节规定的两处改动；其余任何文字/数字修改需先报告等用户确认
- 一次只做一件事，做完验证再进入下一项
- 不确定时停下，把问题写进进度文件，不要猜测

## 5. 环境备忘
- Windows PowerShell；pdflatex/MiKTeX 在 PATH；Git Bash 在 D:\Program Files\Git\bin\bash.exe
- python = C:\Python314\python（脚本用 python xxx.py 跑）
- 本机沙箱辅助进程异常：reasonix 自身按用户权限运行，写文件一般无需审批；
  若遇到权限/网络错误，报告即可
