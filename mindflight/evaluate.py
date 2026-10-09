"""Score trained policies on fixed test starts: `python -m mindflight.evaluate policies/hover.pt [more.pt ...]`.

Each policy flies one episode on each of 1,000 starts (the same starts every time, so scores can be compared),
using its mean action (no exploration noise), on the task's hardest setting. Prints one table row per policy.
"""

import argparse

import torch

from mindflight.envs import TASKS, make_env
from mindflight.rl.train import evaluate, load_policy


def score(
    path: str, n: int = 1000, device: str | None = None, seed: int = 12345, task: str | None = None, **overrides
) -> dict[str, float]:
    """Evaluate one checkpoint -> averages of its per-episode numbers (success, crashed, return, ...).

    task: test on a different task than the one it was trained for (it must have the same observation size),
    e.g. the waypoint policy on the windy sturdy task.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_policy(path, device)
    kwargs = {} if task else dict(checkpoint.get("env_kwargs", {}))
    task = task or checkpoint["task"]
    if getattr(TASKS[task], "has_curriculum", False):
        kwargs["difficulty"] = 1.0
    kwargs.update(overrides)
    env = make_env(task, n, device=device, seed=seed, **kwargs)
    return {"task": task, **evaluate(model, env, seed=seed)}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m mindflight.evaluate", description=__doc__)
    parser.add_argument("policies", nargs="+", help="checkpoint files")
    parser.add_argument("--n", type=int, default=1000, help="test episodes per policy (default 1000)")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--task", default=None, help="test every policy on this task instead of its own")
    parser.add_argument("--env", nargs="*", default=[], metavar="KEY=VALUE", help="task options, e.g. gust_scale=2")
    args = parser.parse_args()
    overrides = {}
    for item in args.env:
        key, value = item.split("=", 1)
        overrides[key] = value.lower() == "true" if value.lower() in ("true", "false") else float(value)

    print(f"{'policy':<34} {'task':<10} {'success':>8} {'crashed':>8} {'return':>8} {'length':>7} {'lap':>7}")
    for path in args.policies:
        s = score(path, n=args.n, seed=args.seed, task=args.task, **overrides)
        lap = f"{s['lap_time_s']:6.2f}s" if "lap_time_s" in s else "      -"
        print(
            f"{path:<34} {s['task']:<10} {100 * s.get('success', 0):7.1f}% {100 * s['crashed']:7.1f}% "
            f"{s['return']:8.1f} {s['length_s']:6.1f}s {lap}"
        )


if __name__ == "__main__":
    main()
