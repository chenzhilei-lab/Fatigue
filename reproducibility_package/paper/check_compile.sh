#!/usr/bin/env bash
# =====================================================================
# check_compile.sh — 编译四查 (hard errors / undefined / overfull / table)
# =====================================================================
# 用途: 每版论文编译后的自动体检，防止 "Extra alignment tab" 这类硬错误
#       在 log 里躺着却没人看 (2026-08-05 v4.4 锚点表格事故的教训)。
#
# 用法:
#   bash check_compile.sh <jobname>          # 例如 gear_fatigue_v4.5
#   bash check_compile.sh <jobname>.tex      # 也接受带 .tex 的写法
#
# 检查项:
#   1. PDF 是否生成 + 页数
#   2. 硬错误 (log 中 ^! 开头行; 含 Extra alignment tab / Misplaced \noalign)
#   3. undefined citation / reference
#   4. 表格结构 (Extra alignment tab / Misplaced \noalign 单列)
#   5. Overfull hbox (>20pt 判 FAIL, 5-20pt 警告)
#   6. 交叉引用是否需要第三遍编译 (Rerun warning)
#
# 退出码: 0 = 全部硬检查通过; 1 = 存在硬检查失败; 2 = 用法/环境错误
# =====================================================================
set -u

JOB="${1:-}"
if [[ -z "$JOB" ]]; then
  echo "用法: bash check_compile.sh <jobname|jobname.tex>" >&2
  exit 2
fi
JOB="${JOB%.tex}"

if [[ ! -f "$JOB.tex" ]]; then
  echo "错误: 找不到 $JOB.tex" >&2
  exit 2
fi
if ! command -v pdflatex >/dev/null 2>&1; then
  echo "错误: 找不到 pdflatex" >&2
  exit 2
fi

FAIL=0
WARN=0
pass()  { echo "  [PASS] $1"; }
warn()  { echo "  [WARN] $1"; WARN=1; }
fail()  { echo "  [FAIL] $1"; FAIL=1; }

# ---------------------------------------------------------------
echo "== 编译 $JOB.tex (第一遍) =="
pdflatex -interaction=nonstopmode -jobname="$JOB" "$JOB.tex" >/dev/null 2>&1
echo "== 编译 $JOB.tex (第二遍) =="
pdflatex -interaction=nonstopmode -jobname="$JOB" "$JOB.tex" >/dev/null 2>&1

LOG="$JOB.log"
if [[ ! -f "$LOG" ]]; then
  echo "[FAIL] 编译未产生 $LOG，检查 pdflatex 环境。" >&2
  exit 1
fi

# ---------------------------------------------------------------
echo "== 检查 1/6: PDF 生成 =="
if [[ -f "$JOB.pdf" ]]; then
  if command -v pdfinfo >/dev/null 2>&1; then
    pages=$(pdfinfo "$JOB.pdf" 2>/dev/null | awk '/^Pages:/{print $2}')
    pass "PDF 已生成 ($pages 页)"
  else
    pass "PDF 已生成"
  fi
else
  fail "未生成 $JOB.pdf"
fi

# ---------------------------------------------------------------
echo "== 检查 2/6: 硬错误 (log 中 ^! 行) =="
ERR=$(grep -n '^!' "$LOG" | head -25)
if [[ -z "$ERR" ]]; then
  pass "无硬错误"
else
  fail "发现 $(grep -c '^!' "$LOG") 个硬错误 (前 25 行):"
  echo "$ERR"
fi

# ---------------------------------------------------------------
echo "== 检查 3/6: undefined citation / reference =="
UNDEF=$(grep -n "Citation .*undefined\|Reference .*undefined" "$LOG" | head -10)
if [[ -z "$UNDEF" ]]; then
  pass "无 undefined 引用"
else
  fail "存在 undefined 引用:"
  echo "$UNDEF"
fi

# ---------------------------------------------------------------
echo "== 检查 4/6: 表格结构 (Extra alignment tab / Misplaced noalign) =="
TAB=$(grep -nE 'Extra alignment tab|Misplaced \\noalign' "$LOG" | head -10)
if [[ -z "$TAB" ]]; then
  pass "表格结构正常"
else
  fail "表格结构错误 (拉出 v4.4 锚点表格事故的那类 bug):"
  echo "$TAB"
fi

# ---------------------------------------------------------------
echo "== 检查 5/6: Overfull hbox =="
OVF=$(grep -n 'Overfull \\hbox' "$LOG" | head -25)
if [[ -z "$OVF" ]]; then
  pass "无 Overfull hbox"
else
  n=$(grep -c 'Overfull \\hbox' "$LOG")
  big=$(grep 'Overfull \\hbox' "$LOG" |
        sed -n 's/.*Overfull \\hbox (\([0-9.]*\)pt too wide).*/\1/p' |
        awk '$1 > 20' | wc -l)
  mid=$(grep 'Overfull \\hbox' "$LOG" |
        sed -n 's/.*Overfull \\hbox (\([0-9.]*\)pt too wide).*/\1/p' |
        awk '$1 > 5 && $1 <= 20' | wc -l)
  if [[ "$big" -gt 0 ]]; then
    fail "$n 处 Overfull，其中 >20pt 有 $big 处 (排版破损风险):"
  else
    warn "$n 处 Overfull (其中 5-20pt 有 $mid 处，>20pt 无)"
  fi
  echo "$OVF"
fi

# ---------------------------------------------------------------
echo "== 检查 6/6: 交叉引用是否需要第三遍 =="
RERUN=$(grep -n "Rerun to get cross-references right\|Label(s) may have changed" "$LOG" | head -5)
if [[ -z "$RERUN" ]]; then
  pass "两遍编译已收敛"
else
  warn "存在 'Rerun' 提示，补第三遍编译:"
  echo "$RERUN"
  echo "  -> pdflatex -interaction=nonstopmode -jobname=$JOB $JOB.tex"
fi

# ---------------------------------------------------------------
echo "============================================================"
if [[ "$FAIL" -eq 0 ]]; then
  echo "结果: PASS (硬检查全过; 警告 $WARN)"
  exit 0
else
  echo "结果: FAIL (有硬检查未过，详见上方)" >&2
  exit 1
fi
