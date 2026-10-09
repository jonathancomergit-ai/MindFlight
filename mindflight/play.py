"""Watch a trained policy fly: `python -m mindflight.play --policy policies/hover.pt`, then open the printed URL.

The task comes from the checkpoint. Tasks where every drone has its own world (hover, waypoint, obstacles) are laid
out on a grid; on the gate course all drones fly the same course side by side, and a new course comes every round.
Press R in the page to start over.
"""

import argparse

import uvicorn

from mindflight.rl.train import load_policy
from mindflight.viewer.__main__ import HOST
from mindflight.viewer.policy_source import SHOW, PolicySource, env_for_showing
from mindflight.viewer.server import create_app

class EpisodeTally:
    """Running counts of finished episodes, shown as info lines in the page."""

    def __init__(self, source: PolicySource, header: list[str]):
        self.source = source
        self.header = header
        self.episodes = 0
        self.crashes = 0
        self.success_sum = 0.0
        self.update_lines()

    def __call__(self, info: dict) -> None:
        episodes = info["episodes"]
        self.episodes += len(episodes["return"])
        self.crashes += int(episodes["crashed"].sum().item())
        if "success" in episodes:
            self.success_sum += episodes["success"].sum().item()
        self.update_lines()

    def update_lines(self) -> None:
        lines = list(self.header)
        if self.episodes:
            lines.append(
                f"episodes {self.episodes} · success {100 * self.success_sum / self.episodes:.0f}% · "
                f"crashes {self.crashes}"
            )
        self.source.info_lines = lines


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m mindflight.play", description=__doc__)
    parser.add_argument("--policy", required=True, help="checkpoint file, e.g. policies/hover.pt")
    parser.add_argument("--n", type=int, default=None, help="number of drones (default depends on the task)")
    parser.add_argument("--port", type=int, default=8000, help="port on 127.0.0.1 (default 8000)")
    parser.add_argument("--device", default="cpu", help="cpu (default; fastest for a few dozen drones) or cuda")
    parser.add_argument("--drone", default=None, help="fly this drone file instead of the one it trained on")
    parser.add_argument("--speed", type=float, default=1.0, help="sim seconds per real second (0.25 = slow motion)")
    args = parser.parse_args()

    model, checkpoint = load_policy(args.policy, args.device)
    task = checkpoint["task"]
    layout, default_n, spacing_m, round_s = SHOW.get(task, ("grid", 16, 4.0, None))
    n = args.n or default_n
    env_kwargs = dict(checkpoint.get("env_kwargs", {}))
    if args.drone:
        env_kwargs["drone"] = args.drone
    env = env_for_showing(task, env_kwargs, n, args.device)
    source = PolicySource(env, model, layout=layout, spacing_m=spacing_m, speed=args.speed, round_s=round_s)

    evaluation = checkpoint.get("eval", {})
    header = [f"task {task} · policy {args.policy} (iteration {checkpoint.get('iteration', '?')})"]
    if "eval_success" in evaluation:
        header.append(
            f"tested on 1,000 starts: success {100 * evaluation['eval_success']:.1f}% · "
            f"crashes {100 * evaluation.get('eval_crashed', 0):.1f}%"
        )
    source.on_episode_end = EpisodeTally(source, header)

    app = create_app(source, on_reset=source.reset)
    print(f"MindFlight play: http://{HOST}:{args.port}/  ({task}, {n} drones, Ctrl+C to stop)")
    uvicorn.run(app, host=HOST, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
