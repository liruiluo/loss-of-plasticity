#!/usr/bin/env bash

set -e

# One-click script to run full Ant-v3 experiments for multiple algorithms and seeds.
# 默认配置：
#   算法: std, ns, l2, cbp, redo （对应 cfg/ant 下的 5 个 yml）
#   种子: 0..19 共 20 个
#   设备: cuda（可通过环境变量 DEVICE 覆盖）
#
# 使用方式（在仓库根目录）:
#   bash lop/rl/run_ant_all.sh
# 或者自定义设备 / 种子:
#   DEVICE=cpu SEEDS="0 1 2 3" bash lop/rl/run_ant_all.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# 选择 Python：优先使用本仓库的 .venv
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

ALGOS=("std" "ns" "l2" "cbp" "redo")
SEEDS=${SEEDS:-"0 1 2"}
DEVICE=${DEVICE:-"cuda"}

for algo in "${ALGOS[@]}"; do
  CFG_PATH="lop/rl/cfg/ant/${algo}.yml"
  if [[ ! -f "${CFG_PATH}" ]]; then
    echo "Config not found: ${CFG_PATH}" >&2
    exit 1
  fi
  for seed in ${SEEDS}; do
    echo "Running Ant-v3 algo=${algo}, seed=${seed}, device=${DEVICE}"
    "${PYTHON_BIN}" -m lop.rl.run_ppo -c "${CFG_PATH}" -s "${seed}" -d "${DEVICE}"
  done
done
