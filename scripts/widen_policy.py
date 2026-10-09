"""Give a trained brain more inputs, so a new task with a longer observation can start from it.

    python scripts/widen_policy.py policies/waypoint_attitude.pt --task pack --out runs/pack/widened.pt
    python -m mindflight.train --task pack --init runs/pack/widened.pt --reset-std 0.15

Why: the pack task's first 22 observation columns mean exactly what the waypoint task's 22 mean (velocity, rotation,
spin, last action, target − position). It just adds more after them (the pack-mates). A network's first layer
computes  h = W · x + b. If we give W extra columns of **zeros** for the new inputs, the widened brain flies
exactly like the old one (the new inputs are multiplied by 0) while gradients can still grow those columns from
zero. So learning starts from "flies to its own slot" instead of from nothing.

The new columns' observation scaling (mean and spread) is measured from a short flight of the new task, so the new
inputs arrive at the network already about mean 0, spread 1.
"""

import argparse
from pathlib import Path

import torch

from mindflight.envs import make_env
from mindflight.rl.ppo import ActorCritic, PPOConfig
from mindflight.rl.train import load_policy, save_checkpoint


def widen(model: ActorCritic, new_obs_size: int) -> ActorCritic:
    """A copy of `model` that takes `new_obs_size` inputs: the old ones first, the new ones with zero weights."""
    old_obs_size = model.obs_norm.mean.shape[0]
    if new_obs_size < old_obs_size:
        raise ValueError(f"can only widen: {old_obs_size} -> {new_obs_size}")
    return remap(model, list(range(old_obs_size)) + [None] * (new_obs_size - old_obs_size))


def remap(model: ActorCritic, columns: list[int | None]) -> ActorCritic:
    """A copy of `model` whose input i is the old input columns[i] (None = a new input, zero weights).

    Old inputs that are left out are dropped. That is the same as always feeding them their average: after
    normalization the average is 0, and 0 × weight adds nothing. So `remap` can also shrink an observation."""
    new_obs_size = len(columns)
    hidden = tuple(layer.out_features for layer in model.actor if isinstance(layer, torch.nn.Linear))[:-1]
    action_size = model.log_std.shape[0]
    wide = ActorCritic(new_obs_size, action_size, hidden, normalize_obs=model.obs_norm.enabled)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    old_mean, old_var = state["obs_norm.mean"], state["obs_norm.var"]
    mean, var = torch.zeros(new_obs_size), torch.ones(new_obs_size)
    for net in ("actor", "critic"):
        old_weight = state[f"{net}.0.weight"]  # the first Linear layer: (hidden, old_obs)
        new_weight = torch.zeros(old_weight.shape[0], new_obs_size)
        for i, old in enumerate(columns):
            if old is not None:
                new_weight[:, i] = old_weight[:, old]
                mean[i], var[i] = old_mean[old], old_var[old]
        state[f"{net}.0.weight"] = new_weight
    state["obs_norm.mean"], state["obs_norm.var"] = mean, var
    wide.load_state_dict(state)
    return wide


@torch.no_grad()
def measure_new_columns(model: ActorCritic, task: str, env_kwargs: dict, old_obs_size: int, steps: int = 100) -> None:
    """Fly `steps` policy steps of the new task with the widened brain and set the new columns' mean and spread."""
    env = make_env(task, 1200, device="cpu", seed=1, **env_kwargs)
    obs = env.reset(stagger=True)
    seen = []
    for _ in range(steps):
        seen.append(obs)
        obs, _, _, _ = env.step(model.act(obs, deterministic=True))
    seen = torch.cat(seen)
    model.obs_norm.mean[old_obs_size:] = seen[:, old_obs_size:].mean(0)
    model.obs_norm.var[old_obs_size:] = seen[:, old_obs_size:].var(0).clamp_min(1e-4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("policy", help="the trained checkpoint to start from")
    parser.add_argument("--task", required=True, help="the new task (its first columns must mean the same)")
    parser.add_argument("--out", required=True, help="where to write the widened checkpoint")
    parser.add_argument("--env", nargs="*", default=[], metavar="KEY=VALUE", help="task options (strings or numbers)")
    parser.add_argument(
        "--columns", default=None, help='old columns to keep, in order, e.g. "0:40,52,53:62" (default: all, then new)'
    )
    args = parser.parse_args()

    model, checkpoint = load_policy(args.policy, "cpu")
    env_kwargs = dict(checkpoint.get("env_kwargs", {}))
    for item in args.env:
        key, value = item.split("=", 1)
        env_kwargs[key] = value
    old_obs_size = model.obs_norm.mean.shape[0]
    new_obs_size = make_env(args.task, 12, device="cpu", **env_kwargs).obs_size
    if args.columns:
        columns: list[int | None] = []
        for part in args.columns.split(","):
            low, _, high = part.partition(":")
            columns += list(range(int(low), int(high))) if high else [int(low)]
        columns += [None] * (new_obs_size - len(columns))
        wide = remap(model, columns)
    else:
        wide = widen(model, new_obs_size)
    if args.columns is None:  # (with --columns, new columns keep mean 0, spread 1)
        measure_new_columns(wide, args.task, env_kwargs, old_obs_size)

    class Holder:  # save_checkpoint wants something with .model and .config
        pass

    holder = Holder()
    holder.model, holder.config = wide, PPOConfig(**checkpoint["config"])
    save_checkpoint(Path(args.out), holder, args.task, env_kwargs, 0, {"widened_from": args.policy})
    print(f"{args.policy} ({old_obs_size} inputs) -> {args.out} ({new_obs_size} inputs, task {args.task})")


if __name__ == "__main__":
    main()
