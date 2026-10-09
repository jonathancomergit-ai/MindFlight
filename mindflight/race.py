"""Level 6: race trained policies against each other, live. `python -m mindflight.race`, then open the printed URL.

Every entrant (a trained policy) flies 2 drones on the same random gate course. Each drone gets one try per round:
finish all 6 gates, or crash (out). A round ends when every drone is done, or after 20 s. The fastest lap wins the
round, then a new course is made. The leaderboard in the corner keeps score: round wins, finishes, crashes, best
and average lap.

    python -m mindflight.race                                   # the shipped policies
    python -m mindflight.race --entrant fast=policies/race.pt --entrant steady=policies/gates.pt ...
"""

import argparse
from pathlib import Path

import torch
import uvicorn

from mindflight.envs.race import RaceEnv
from mindflight.rl.train import load_policy
from mindflight.viewer.__main__ import HOST
from mindflight.viewer.policy_source import PolicySource
from mindflight.viewer.server import create_app

DEFAULT_ENTRANTS = [
    ("racer", "policies/race.pt", "gates policy, then trained to hurry"),
    ("gates", "policies/gates.pt", "gates policy (Level 3)"),
    ("rookie", "policies/gates_rookie.pt", "gates policy that only saw easy courses"),
]
COLORS = ["#6fdc8c", "#ff8389", "#4ea8ff", "#c08cff"]  # the viewer's group colours (main.js GROUP_COLORS), in order
ROUND_S = 20.0
PAUSE_S = 2.0


class Entrant:
    """One policy in the race, and its running score."""

    def __init__(self, name: str, path: str, note: str = ""):
        self.name, self.path, self.note = name, path, note
        self.wins = 0
        self.finishes = 0
        self.crashes = 0
        self.laps: list[float] = []

    def row(self, color: str, round_result: str) -> list:
        best = f"{min(self.laps):.2f} s" if self.laps else "–"
        mean = f"{sum(self.laps) / len(self.laps):.2f} s" if self.laps else "–"
        return [f"{color}|{self.name}", self.wins, self.finishes, self.crashes, best, mean, round_result]


class RaceSource(PolicySource):
    """A PolicySource that runs rounds: one try per drone, frozen in place when it finishes or crashes."""

    def __init__(self, entrants: list[Entrant], models: list, drones_each: int = 2, device: str = "cpu"):
        n = drones_each * len(entrants)
        env = RaceEnv(n, device=device, shared_course=True, seed=int(torch.randint(0, 2**31 - 1, ()).item()))
        policy_of_drone = torch.arange(n) // drones_each
        super().__init__(env, models, policy_of_drone=policy_of_drone, layout="shared")
        self.entrants = entrants
        self.drones_each = drones_each
        self.round = 0
        self.start_round()

    def start_round(self) -> None:
        self.round += 1
        self.done = torch.zeros(self.env.n_envs, dtype=torch.bool)
        self.result_s = torch.full((self.env.n_envs,), float("inf"))  # lap time, inf = didn't finish
        self.gates_reached = torch.zeros(self.env.n_envs, dtype=torch.long)
        self.frozen_pose = torch.zeros(self.env.n_envs, 7)
        self.round_time_s = 0.0
        self.pause_s = 0.0
        self.env.new_shared_course()
        self.obs = self.env.reset()
        self.update_board()

    def advance(self, n_steps: int) -> None:
        for _ in range(n_steps):
            if self.pause_s > 0:  # showing the result before the next round
                self.pause_s -= 0.02
                if self.pause_s <= 0:
                    self.start_round()
                continue
            pose_before = self.env.poses().to("cpu").clone()
            next_gate_before = self.env.next_gate.to("cpu").clone()
            self.obs, _, done, info = self.env.step(self.actions(self.obs))
            self.round_time_s += 0.02
            # info["episodes"] lists the drones that just ended, in drone order.
            for j, i in enumerate(done.nonzero(as_tuple=True)[0].tolist()):
                if self.done[i]:
                    continue  # already had its try this round (the env reset it; we ignore the rest)
                self.frozen_pose[i] = pose_before[i]
                if info["episodes"]["success"][j] > 0:
                    self.result_s[i] = self.round_time_s
                    self.gates_reached[i] = 6
                else:
                    self.gates_reached[i] = next_gate_before[i]
                    self.frozen_pose[i, 2] = 0.0  # crashed or out of time: shown lying on the floor
                self.done[i] = True
            if self.done.all() or self.round_time_s >= ROUND_S:
                self.finish_round()

    def restart(self) -> None:
        """The page's R key: a new round on a new course (scores are kept)."""
        with self._lock:
            self.start_round()

    def finish_round(self) -> None:
        for e, entrant in enumerate(self.entrants):
            mine = slice(e * self.drones_each, (e + 1) * self.drones_each)
            laps = self.result_s[mine]
            entrant.finishes += int(torch.isfinite(laps).sum())
            entrant.crashes += int((self.done[mine] & ~torch.isfinite(laps)).sum())
            entrant.laps += [float(t) for t in laps if torch.isfinite(t)]
        best = int(self.result_s.argmin())
        if torch.isfinite(self.result_s[best]):
            self.entrants[best // self.drones_each].wins += 1
        self.update_board()
        self.done[:] = True
        self.pause_s = PAUSE_S

    def update_board(self) -> None:
        rows = []
        for e, entrant in enumerate(self.entrants):
            mine = slice(e * self.drones_each, (e + 1) * self.drones_each)
            laps = self.result_s[mine]
            if torch.isfinite(laps).any():
                now = f"lap {float(laps.min()):.2f} s"
            elif self.done[mine].all():
                now = f"out (gate {int(self.gates_reached[mine].max()) + 1})"
            else:
                now = "flying"
            rows.append(entrant.row(COLORS[e % len(COLORS)], now))
        rows.sort(key=lambda r: (-r[1], float(r[4].split()[0]) if r[4] != "–" else 1e9))
        self.board = {
            "title": f"Race · round {self.round}",
            "columns": ["policy", "wins", "finishes", "crashes", "best lap", "avg lap", "this round"],
            "rows": rows,
        }

    def frame(self, time_s: float):
        poses = super().frame(time_s)
        frozen = self.done.numpy()
        poses[frozen] = self.frozen_pose[self.done].numpy()
        return poses


def parse_entrants(args_entrants: list[str] | None) -> list[Entrant]:
    if not args_entrants:
        entrants = [Entrant(name, path, note) for name, path, note in DEFAULT_ENTRANTS if Path(path).exists()]
    else:
        entrants = []
        for item in args_entrants:
            name, _, path = item.partition("=")
            entrants.append(Entrant(name, path))
    if len(entrants) < 2:
        raise SystemExit("need at least 2 policies to race (see --entrant)")
    return entrants


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m mindflight.race", description=__doc__)
    parser.add_argument("--entrant", action="append", metavar="NAME=PATH", help="a policy to race (repeat)")
    parser.add_argument("--drones-each", type=int, default=2)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    entrants = parse_entrants(args.entrant)
    models = [load_policy(e.path, "cpu")[0] for e in entrants]
    source = RaceSource(entrants, models, drones_each=args.drones_each)
    source.info_lines = ["Race: each policy flies 2 drones on the same course. One try per round. Fastest lap wins."]
    groups = [
        {
            "name": e.name,
            "motors": e.path,
            "expect": e.note,
            "first": i * args.drones_each,
            "count": args.drones_each,
        }
        for i, e in enumerate(entrants)
    ]
    app = create_app(source, extra_config={"groups": groups}, on_reset=source.restart)
    print(f"MindFlight race: http://{HOST}:{args.port}/  ({len(entrants)} policies, Ctrl+C to stop)")
    for e in entrants:
        print(f"  {e.name:<8} {e.path}")
    uvicorn.run(app, host=HOST, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
