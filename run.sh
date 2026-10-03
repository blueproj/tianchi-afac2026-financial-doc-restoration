#!/bin/bash
# ============================================================
# AFAC2026 赛题二「复杂金融文档还原挑战」B 榜一键复现脚本
# COM团队
# 用法：bash run.sh
# 产物：submission_B.csv（B 榜提交格式，100 行）
# ============================================================
set -e
cd "$(dirname "$0")"

# 自动激活 conda 环境 afac2026（若已激活则跳过；若未安装 conda 则用当前 Python）
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "afac2026" ]; then
    if command -v conda &>/dev/null && conda env list 2>/dev/null | grep -q '^afac2026 '; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate afac2026
    fi
fi

# B 榜评测数据（已随包附带；如需替换数据集，修改以下两个目录即可）
TABLE_DIR="data/finix_huge_table_rest_B/images"
LONG_DIR="data/finix_huge_long_rest_B/images"

python scripts/run_pipeline_v2.py \
    --table-dir "$TABLE_DIR" \
    --long-dir "$LONG_DIR" \
    --no-em-fill --no-resume \
    --save-dir output/pipeline_B \
    --output submission_B.csv

# 生成中间产物的汇总导航页（纯本地 HTML，不影响提交结果）
python scripts/gen_index_html.py --dir output/pipeline_B

echo ""
echo "===== 复现完成 ====="
echo "提交文件: submission_B.csv"
echo "结果汇总页: output/pipeline_B/index.html（浏览器打开，可逐样本查看渲染表格与 chunk 级中间过程）"
echo "中间产物: output/pipeline_B/（每样本含 final.md / report.html）"
echo "参考对照: reference_submission_B_v8.csv（B 榜实际提交，得分 93.876479）"
