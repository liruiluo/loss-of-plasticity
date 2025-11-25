import argparse
import os
import pickle
from typing import List

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
        help="Environment step budget (e.g., 1_000_000) for computing success rate.",
    )
    return parser.parse_args()


def load_run_success_rate(path: str, step_limit: int) -> float:
    """
    Loads a single run file and computes success rate up to step_limit.
    success_rate = mean(successes for episodes whose termination_step <= step_limit)
    """
    if not os.path.isfile(path):
        return None

    with open(path, "rb") as f:
        data = pickle.load(f)

    term_steps = np.asarray(data.get("termination_steps", []), dtype=np.int64)
    successes = data.get("successes", None)
    if successes is None:
        return None
    successes = np.asarray(successes, dtype=np.float32)

    if term_steps.shape[0] == 0 or successes.shape[0] == 0:
        return None
    n = min(term_steps.shape[0], successes.shape[0])
    term_steps = term_steps[:n]
    successes = successes[:n]

    mask = term_steps <= step_limit
    if not np.any(mask):
        return None

    return float(successes[mask].mean())


def main():
    args = parse_args()
    base_dir = args.base_dir
    tasks: List[str] = [t.strip() for t in args.tasks.split(",") if t.strip()]
    seeds: List[int] = [int(s) for s in args.seeds.split(",") if s.strip()]
    step_limit = args.steps

    per_task_means = []
    per_task_stds = []

    print(f"Analyzing MetaWorld results in '{base_dir}' up to {step_limit} steps.")
    print(f"Tasks: {tasks}")
    print(f"Seeds: {seeds}")
    print()

    for task in tasks:
        task_dir = os.path.join(base_dir, task)
        rates = []
        for seed in seeds:
            run_path = os.path.join(task_dir, f"{seed}.log")
            rate = load_run_success_rate(run_path, step_limit=step_limit)
            if rate is not None:
                rates.append(rate)

        if len(rates) == 0:
            print(f"{task:20s} -> no valid runs found")
            continue

        rates = np.asarray(rates, dtype=np.float32)
        mean_rate = float(rates.mean())
        std_rate = float(rates.std())
        per_task_means.append(mean_rate)
        per_task_stds.append(std_rate)

        print(f"{task:20s} mean={mean_rate:.3f}  std={std_rate:.3f}  (n={len(rates)})")

    if len(per_task_means) == 0:
        print("\nNo tasks with valid data. Nothing to summarize.")
        return

    per_task_means = np.asarray(per_task_means, dtype=np.float32)
    per_task_stds = np.asarray(per_task_stds, dtype=np.float32)

    overall_mean_of_means = float(per_task_means.mean())
    overall_mean_of_stds = float(per_task_stds.mean())

    print("\nSummary over tasks:")
    print(f"Mean of per-task means      = {overall_mean_of_means:.3f}")
    print(f"Mean of per-task stds       = {overall_mean_of_stds:.3f}")


if __name__ == "__main__":
    main()

