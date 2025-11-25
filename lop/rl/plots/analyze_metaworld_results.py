import argparse
import os
import pickle
from typing import List, Tuple

import numpy as np


DEFAULT_TASKS = [
    "hammer",
    "push-wall",
    "faucet-close",
    "push-back",
    "stick-pull",
    "handle-press-side",
    "push",
    "shelf-place",
    "window-close",
    "peg-unplug-side",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze MetaWorld PPO results.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="data/metaworld",
        help="Base directory where per-task result folders are stored.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated list of task names (folder names under base-dir).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated list of seeds to include.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1_000_000,
        help="Environment step budget (e.g., 1_000_000) for computing metrics.",
    )
    return parser.parse_args()


def load_run_metrics(path: str, step_limit: int) -> Tuple[float, float]:
    """
    Loads a single run file and computes success rate and average return
    up to step_limit.

    success_rate = mean(successes for episodes whose termination_step <= step_limit)
    avg_return   = mean(episode_returns for episodes whose termination_step <= step_limit)
    """
    if not os.path.isfile(path):
        return None, None

    with open(path, "rb") as f:
        data = pickle.load(f)

    term_steps = np.asarray(data.get("termination_steps", []), dtype=np.int64)
    successes = data.get("successes", None)
    rets = data.get("rets", None)

    if successes is None or rets is None:
        return None, None

    successes = np.asarray(successes, dtype=np.float32)
    rets = np.asarray(rets, dtype=np.float32)

    if term_steps.shape[0] == 0 or successes.shape[0] == 0 or rets.shape[0] == 0:
        return None, None

    n = min(term_steps.shape[0], successes.shape[0], rets.shape[0])
    term_steps = term_steps[:n]
    successes = successes[:n]
    rets = rets[:n]

    mask = term_steps <= step_limit
    if not np.any(mask):
        return None, None

    return float(successes[mask].mean()), float(rets[mask].mean())


def main():
    args = parse_args()
    base_dir = args.base_dir
    tasks: List[str] = [t.strip() for t in args.tasks.split(",") if t.strip()]
    seeds: List[int] = [int(s) for s in args.seeds.split(",") if s.strip()]
    step_limit = args.steps

    per_task_succ_means = []
    per_task_succ_stds = []
    per_task_ret_means = []
    per_task_ret_stds = []

    print(f"Analyzing MetaWorld results in '{base_dir}' up to {step_limit} steps.")
    print(f"Tasks: {tasks}")
    print(f"Seeds: {seeds}")
    print()

    for task in tasks:
        task_dir = os.path.join(base_dir, task)
        succ_rates = []
        avg_rets = []
        for seed in seeds:
            run_path = os.path.join(task_dir, f"{seed}.log")
            succ, ret = load_run_metrics(run_path, step_limit=step_limit)
            if succ is not None and ret is not None:
                succ_rates.append(succ)
                avg_rets.append(ret)

        if len(succ_rates) == 0:
            print(f"{task:20s} -> no valid runs found")
            continue

        succ_rates = np.asarray(succ_rates, dtype=np.float32)
        avg_rets = np.asarray(avg_rets, dtype=np.float32)

        succ_mean = float(succ_rates.mean())
        succ_std = float(succ_rates.std())
        ret_mean = float(avg_rets.mean())
        ret_std = float(avg_rets.std())

        per_task_succ_means.append(succ_mean)
        per_task_succ_stds.append(succ_std)
        per_task_ret_means.append(ret_mean)
        per_task_ret_stds.append(ret_std)

        print(
            f"{task:20s} "
            f"succ_mean={succ_mean:.3f} succ_std={succ_std:.3f} "
            f"ret_mean={ret_mean:.1f} ret_std={ret_std:.1f} (n={len(succ_rates)})"
        )

    if len(per_task_succ_means) == 0:
        print("\nNo tasks with valid data. Nothing to summarize.")
        return

    per_task_succ_means = np.asarray(per_task_succ_means, dtype=np.float32)
    per_task_succ_stds = np.asarray(per_task_succ_stds, dtype=np.float32)
    per_task_ret_means = np.asarray(per_task_ret_means, dtype=np.float32)
    per_task_ret_stds = np.asarray(per_task_ret_stds, dtype=np.float32)

    overall_succ_mean = float(per_task_succ_means.mean())
    overall_succ_std_mean = float(per_task_succ_stds.mean())
    overall_ret_mean = float(per_task_ret_means.mean())
    overall_ret_std_mean = float(per_task_ret_stds.mean())

    print("\nSummary over tasks:")
    print(f"Mean of per-task success means    = {overall_succ_mean:.3f}")
    print(f"Mean of per-task success stds     = {overall_succ_std_mean:.3f}")
    print(f"Mean of per-task return means     = {overall_ret_mean:.1f}")
    print(f"Mean of per-task return stds      = {overall_ret_std_mean:.1f}")


if __name__ == "__main__":
    main()

