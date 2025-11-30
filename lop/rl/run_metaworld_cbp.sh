#!/usr/bin/env bash

set -e

# Run MetaWorld single-task PPO+CBP experiments for 10 tasks.
# 使用（仓库根目录）:
#   bash lop/rl/run_metaworld_cbp.sh
# 可选环境变量:
#   SEEDS="0 1 2"   DEVICE="cpu"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CFG_DIR="lop/rl/cfg/metaworld_cbp"
TASKS=(
  hammer
  push-wall
  faucet-close
  push-back
  stick-pull
  handle-press-side
  push
  shelf-place
  window-close
  peg-unplug-side
)

SEEDS=${SEEDS:-"0 1 2 3 4"}
DEVICE=${DEVICE:-"cuda"}

for task in "${TASKS[@]}"; do
  cfg="${CFG_DIR}/${task}.yml"
  if [[ ! -f "${cfg}" ]]; then
    echo "Config not found: ${cfg}" >&2
    exit 1
  fi
  for seed in ${SEEDS}; do
    echo "Running CBP task=${task}, seed=${seed}, device=${DEVICE}"
    uv run python -m lop.rl.run_ppo -c "${cfg}" -s "${seed}" -d "${DEVICE}"
  done
done

