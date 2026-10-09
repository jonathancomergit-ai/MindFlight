"""Give an existing policy more observation inputs, so it can start a task whose observation is longer.

    python scripts/widen_policy.py policies/waypoint_attitude.pt --task follow --out runs/follow/init.pt
    python -m mindflight.train --task follow --init runs/follow/init.pt --reset-std 0.15

`--init` needs the same observation size. A new task that keeps the old observation as its *first* columns and adds
more after them (follow keeps waypoint's 22 and adds 12) can still start from the old brain:

- the first layer of both networks gets new input columns with **zero** weights, so at first the new inputs change
  nothing and the widened policy flies exactly like the old one;
- the observation scaling (`obs_norm`) keeps the old columns' mean and variance; the new columns get theirs from a
  short rollout of the new task;
- the scaling's sample count is cut to 1 M, so the statistics can follow the new task within a few iterations.
"""

import argparse
from pathlib import Path

import torch

from mindflight.envs import make_env
from mindflight.rl.train import load_policy

NEW_COUNT = 1e6


@torch.no_grad()
def widen(path: str, task: str, env_kwargs: dict, device: str, n: int = 4096, steps: int = 100) -> dict:
    """Load checkpoint `path` and return a checkpoint dict for `task` with the first layers widened."""
    model, checkpoint = load_policy(path, device)
    kwargs = dict(checkpoint.get("env_kwargs", {}))
    kwargs.update(env_kwargs)
    env = make_env(task, n, device=device, seed=0, **kwargs)
    old_size = model.obs_norm.mean.shape[0]
    new_size = env.obs_size
    if new_size < old_size:
        raise ValueError(f"{task} has {new_size} observations, fewer than the policy's {old_size}")

    # Statistics of the new columns, from the old policy flying the new task (it only reads the old columns).
    obs = env.reset()
    seen = []
    for _ in range(steps):
        seen.append(obs)
        obs, _, _, _ = env.step(model.act(obs[:, :old_size], deterministic=True))
    seen = torch.cat(seen)

    state = {k: v.clone() for k, v in model.state_dict().items()}
    for net in ("actor", "critic"):
        weight = state[f"{net}.0.weight"]
        state[f"{net}.0.weight"] = torch.cat((weight, weight.new_zeros(weight.shape[0], new_size - old_size)), dim=1)
    mean, var = seen.mean(0), seen.var(0, unbiased=False).clamp_min(1e-4)
    mean[:old_size], var[:old_size] = state["obs_norm.mean"], state["obs_norm.var"]
    state["obs_norm.mean"], state["obs_norm.var"] = mean, var
    state["obs_norm.count"] = torch.tensor(min(NEW_COUNT, float(state["obs_norm.count"])))
    return {
        "task": task,
        "env_kwargs": kwargs,
        "iteration": 0,
        "config": checkpoint["config"],
        "model": state,
        "widened_from": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("policy")
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--env", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    env_kwargs = dict(item.split("=", 1) for item in args.env)
    checkpoint = widen(args.policy, args.task, env_kwargs, args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)
    print(f"{args.policy} -> {args.out} ({args.task}, obs {checkpoint['model']['obs_norm.mean'].shape[0]})")


if __name__ == "__main__":
    main()
