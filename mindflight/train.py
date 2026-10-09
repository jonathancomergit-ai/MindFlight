"""Train a policy: `python -m mindflight.train --task hover`.

Prints one line per iteration and writes runs/<task>/<date-time>/ with metrics.jsonl, latest.pt and best.pt.
"""

import argparse
from pathlib import Path

from mindflight.envs import TASKS
from mindflight.rl.ppo import PPOConfig
from mindflight.rl.train import train


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m mindflight.train", description=__doc__)
    parser.add_argument("--task", default="hover", choices=sorted(TASKS))
    parser.add_argument("--n", type=int, default=4096, help="drones training at once (default 4096)")
    parser.add_argument("--iters", type=int, default=300, help="training iterations (default 300)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--init", default=None, help="start from this checkpoint instead of from scratch")
    parser.add_argument("--run-dir", default=None, help="where to write the run (default runs/<task>/<date-time>)")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--view", action="store_true", help="watch it learn live in the viewer")
    parser.add_argument("--port", type=int, default=8000, help="viewer port with --view (default 8000)")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="override PPOConfig fields, e.g. --set init_std=0.3"
    )
    parser.add_argument("--env", nargs="*", default=[], metavar="KEY=VALUE", help="task options, e.g. --env randomize=false")
    args = parser.parse_args()

    env_kwargs = {}
    for item in args.env:
        key, value = item.split("=", 1)
        lowered = value.lower()
        if lowered in ("true", "false"):
            env_kwargs[key] = lowered == "true"
        else:
            try:
                env_kwargs[key] = float(value) if "." in value else int(value)
            except ValueError:
                env_kwargs[key] = value

    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        default = getattr(PPOConfig(), key)  # raises if the key is not a PPOConfig field
        if isinstance(default, tuple):
            overrides[key] = tuple(int(v) for v in value.split(","))
        elif isinstance(default, bool):
            overrides[key] = value.lower() in ("1", "true", "yes", "on")
        else:
            overrides[key] = type(default)(value)

    view = {}

    def start_view(ppo) -> None:
        from mindflight.viewer.live import TrainingView

        view["view"] = TrainingView(args.task, env_kwargs, ppo.model, port=args.port)

    def update_view(env, metrics, model) -> None:
        view["view"](env, metrics, model)

    run_dir = train(
        args.task,
        n_envs=args.n,
        iterations=args.iters,
        seed=args.seed,
        device=args.device,
        config=PPOConfig(**overrides),
        env_kwargs=env_kwargs,
        run_dir=Path(args.run_dir) if args.run_dir else None,
        init_checkpoint=args.init,
        eval_every=args.eval_every,
        on_start=start_view if args.view else None,
        on_iteration=update_view if args.view else None,
    )
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
