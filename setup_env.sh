#!/bin/bash
# ============================================================
# 运行环境配置脚本（主办方复现用）
# 要求：已安装 conda（或直接用系统 Python 3.11 + pip 安装依赖）
# ============================================================
set -e

# 方式一：conda（推荐）
conda create -n afac2026 python=3.11 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate afac2026
pip install -r requirements.txt

echo ""
echo "环境就绪。运行复现："
echo "  conda activate afac2026 && bash run.sh"

# 方式二（备选）：无 conda 时，任意 Python>=3.10 环境直接：
#   pip install -r requirements.txt && bash run.sh
