#!/usr/bin/env bash

set -e

# Run Ant-v3 experiments comparing standard PPO vs PPO+CBP only.
# 算法: std, cbp
# 种子: 默认 0..19（可通过 SEEDS 覆盖）
# 设备: 默认 cuda（可通过 DEVICE 覆盖）
#
# 用法（仓库根目录）:
#   bash lop/rl/run_ant_cbp.sh
# 或:
#   DEVICE=cpu SEEDS="0 1 2 3" bash lop/rl/run_ant_cbp.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

ALGOS=("std" "cbp")
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"}
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

