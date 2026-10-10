"""Fly a trained policy on a task, in real time, for the viewer.

`PolicySource` is a `FrameSource` (see `server.py`). Each frame it looks at the wall clock and runs as many policy
steps (50 per second, each with its 10 physics steps) as real time needs, capped so a stall can't freeze it, then
hands the viewer the poses. It also sends the viewer the task's objects (targets, gates, pillars) as a "scene"
message, plus optional info lines, charts and a leaderboard.

Layouts:
- "grid": every drone has its own world (the usual training setup), so they are spread out on a grid with
  `spacing_m` between them, and each drone's targets and gates are moved along with it. If the env has a
  `world_id` (N,) (packs: several drones per world), the grid has one spot per *world* and all its drones share it.
- "shared": every drone flies in the same world (gates: they all fly the same course, like a race). Nothing is
  moved, and objects that every drone shares are drawn once.

The policy can be swapped while it flies (`set_policy`): the training view does that after every iteration, so
you watch the current brain fly.
"""

import math
import threading
import time
from collections.abc import Callable

import numpy as np
import torch

from mindflight.envs import TASKS, make_env
from mindflight.envs.base import POLICY_DT_S, DroneEnv
from mindflight.rl.ppo import ActorCritic


# How each task is shown: layout, drones, spacing between worlds (m), seconds per round (None = never restart).
SHOW = {
    "hover": ("grid", 64, 3.0, None),
    "waypoint": ("grid", 36, 4.5, None),
    "sturdy": ("grid", 36, 4.5, None),
    "gates": ("shared", 8, 0.0, 20.0),
    "obstacles": ("grid", 4, 10.0, None),
    "race": ("shared", 8, 0.0, 20.0),
    "follow": ("grid", 4, 16.0, None),
    "intercept": ("grid", 4, 12.0, None),
    "pack": ("grid", 24, 9.0, None),  # 4 packs of 6; spacing is between packs (worlds), not drones
}


def env_for_showing(task: str, env_kwargs: dict, n: int, device: str):
    """The task env as it is shown: hardest difficulty, and one shared course for the gate tasks."""
    kwargs = dict(env_kwargs)
    if getattr(TASKS[task], "has_curriculum", False):
        kwargs["difficulty"] = 1.0
    if SHOW.get(task, ("grid",))[0] == "shared":
        kwargs["shared_course"] = True
    return make_env(task, n, device=device, seed=int(torch.randint(0, 2**31 - 1, ()).item()), **kwargs)


def grid_offsets(n: int, spacing_m: float) -> torch.Tensor:
    """(n, 3) offsets that lay n worlds out on a square grid centred on the origin, spacing_m apart."""
    side = math.ceil(math.sqrt(n))
    index = torch.arange(n)
    row, col = index // side, index % side
    centre = (side - 1) / 2
    offsets = torch.zeros(n, 3)
    offsets[:, 0] = (row - centre) * spacing_m
    offsets[:, 1] = (col - centre) * spacing_m
    return offsets


class PolicySource:
    """Runs `env` with `model` in real time. `policy_for_drone` lets different drones fly different policies (race).

    env: a task env, usually small (64 drones) and on the CPU. model: an ActorCritic, or a list of them plus
    `policy_of_drone` (N,) saying which model flies which drone.
    """

    def __init__(
        self,
        env: DroneEnv,
        model: ActorCritic | list[ActorCritic],
        policy_of_drone: torch.Tensor | None = None,
        layout: str = "grid",
        spacing_m: float = 3.0,
        speed: float = 1.0,
        max_catch_up_s: float = 0.1,
        on_episode_end: Callable[[dict], None] | None = None,
        round_s: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.env = env
        self.models = model if isinstance(model, list) else [model]
        self.policy_of_drone = (
            torch.zeros(env.n_envs, dtype=torch.long) if policy_of_drone is None else policy_of_drone
        ).to(env.device)
        self.layout = layout
        self.offsets = (grid_offsets(env.n_envs, spacing_m) if layout == "grid" else torch.zeros(env.n_envs, 3)).to(
            env.device
        )
        world_id = getattr(env, "world_id", None)  # envs where several drones share a world (packs)
        if layout == "grid" and world_id is not None:  # one grid spot per world: its drones stay together
            world_id = world_id.cpu()
            self.offsets = grid_offsets(int(world_id.max()) + 1, spacing_m)[world_id].to(env.device)
        self.speed = speed
        self.max_catch_up_s = max_catch_up_s
        self.on_episode_end = on_episode_end
        self.round_s = round_s  # start everything over (and a new shared course) this often, in sim seconds
        self._clock = clock
        self._lock = threading.Lock()
        self.n_drones = env.n_envs

        # Things the page shows besides the drones (see server.py -> text messages).
        self.info_lines: list[str] = []
        self.chart_series: list[dict] = []
        self.metric_points: list[dict] = []
        self.board: dict | None = None
        self.reset()

    def set_policy(self, state_dict: dict, index: int = 0) -> None:
        """Swap in new weights for policy `index` (thread-safe: training calls this from its own thread)."""
        with self._lock:
            self.models[index].load_state_dict(state_dict)

    def reset(self) -> None:
        with self._lock:
            self._reset_unlocked()

    def _reset_unlocked(self) -> None:
        if hasattr(self.env, "new_shared_course") and getattr(self.env, "shared_course", False):
            self.env.new_shared_course()
        self.obs = self.env.reset()
        self.round_time_s = 0.0
        self._unspent_s = 0.0
        self._last_clock_s = self._clock()

    @torch.no_grad()
    def actions(self, obs: torch.Tensor) -> torch.Tensor:
        """Each drone's action from its own policy (the mean action: no exploration noise when showing off)."""
        if len(self.models) == 1:
            return self.models[0].act(obs, deterministic=True)
        action = torch.zeros(obs.shape[0], self.env.action_size, device=obs.device)
        for index, model in enumerate(self.models):
            mine = self.policy_of_drone == index
            if mine.any():
                action[mine] = model.act(obs[mine], deterministic=True)
        return action

    def advance(self, n_steps: int) -> None:
        """Run n policy steps right now (each = 10 physics steps)."""
        for _ in range(n_steps):
            self.obs, _, _, info = self.env.step(self.actions(self.obs))
            if self.on_episode_end is not None and info["episodes"]:
                self.on_episode_end(info)
            self.round_time_s += POLICY_DT_S
            if self.round_s is not None and self.round_time_s >= self.round_s:
                self._reset_unlocked()

    def frame(self, time_s: float) -> np.ndarray:
        """Catch up with the wall clock (capped), then return the (N, 7) float32 poses, moved to their grid spots."""
        with self._lock:
            now_s = self._clock()
            owed_s = min(max((now_s - self._last_clock_s) * self.speed, 0.0), self.max_catch_up_s)
            self._last_clock_s = now_s
            self._unspent_s += owed_s
            n_steps = int(self._unspent_s / POLICY_DT_S)
            self._unspent_s -= n_steps * POLICY_DT_S
            self.advance(n_steps)
            poses = self.env.poses().clone()
            poses[:, 0:3] += self.offsets
            return poses.to("cpu", torch.float32).numpy()

    # ---------- text messages for the page ----------

    def scene(self) -> dict:
        """The task's objects as a "scene" message, moved to each drone's grid spot."""
        # One copy to the CPU first: on the GPU, every .tolist() below would be its own slow round trip.
        objects = {k: v.cpu() if torch.is_tensor(v) else v for k, v in self.env.display_objects().items()}
        scene: dict = {"type": "scene"}
        offsets = self.offsets.cpu()
        shared = self.layout == "shared"

        if "targets" in objects:
            targets = objects["targets"] + offsets
            if shared:
                targets = torch.unique(targets.round(decimals=3), dim=0)
            scene["targets"] = targets.round(decimals=3).tolist()

        if "gate_centre" in objects:
            centres = objects["gate_centre"] + offsets[:, None, :]  # (N, G, 3)
            headings = objects["gate_heading"]  # (N, G)
            opening = objects["gate_opening"]  # (N,)
            next_gate = objects["next_gate"]  # (N,)
            racing = self.racing_mask().cpu()
            if shared and racing.any():
                next_gate = next_gate[racing]  # gate colours follow only the drones still in the running
            n_gates = centres.shape[1]
            drones = [0] if shared else range(centres.shape[0])
            gates = []
            for i in drones:
                for k in range(n_gates):
                    if shared:
                        done = bool((next_gate > k).all())
                        is_next = bool((next_gate == k).any())
                    else:
                        done = bool(next_gate[i] > k)
                        is_next = bool(next_gate[i] == k)
                    gates.append(
                        {
                            "c": [round(v, 3) for v in centres[i, k].tolist()],
                            "yaw": round(float(headings[i, k]), 4),
                            "size": round(float(opening[i]), 3),
                            "state": "done" if done else "next" if is_next else "later",
                        }
                    )
            scene["gates"] = gates

        if "pillars" in objects:
            pillars = objects["pillars"]  # (N, P, 3): x, y, radius (radius 0 = not there)
            drones = [0] if shared else range(pillars.shape[0])
            scene["pillars"] = [
                [round(float(p[0] + offsets[i, 0]), 3), round(float(p[1] + offsets[i, 1]), 3), round(float(p[2]), 3)]
                for i in drones
                for p in pillars[i]
                if p[2] > 0
            ]

        # General shapes any task can use (people, balls, links between drones), per drone in its own world:
        #   "spheres"   (N, K, 4): x, y, z, radius            radius ≤ 0 = skip
        #   "cylinders" (N, K, 5): x, y, z_bottom, height, radius (upright)
        #   "lines"     (N, K, 6): x1, y1, z1, x2, y2, z2     zero length = skip
        # with optional "<kind>_colors": K hex strings like "#ff9f43", one per K slot.
        for kind, width in (("spheres", 4), ("cylinders", 5), ("lines", 6)):
            if kind not in objects:
                continue
            shapes = objects[kind]
            colors = objects.get(f"{kind}_colors") or ["#c8d0da"] * shapes.shape[1]
            drones = [0] if shared else range(shapes.shape[0])
            out = []
            for i in drones:
                for k in range(shapes.shape[1]):
                    row = shapes[i, k].tolist()
                    if kind == "lines":
                        row[0:3] = [v + o for v, o in zip(row[0:3], offsets[i].tolist())]
                        row[3:6] = [v + o for v, o in zip(row[3:6], offsets[i].tolist())]
                        if row[0:3] == row[3:6]:
                            continue
                    else:
                        if row[-1] <= 0:
                            continue
                        row[0:3] = [v + o for v, o in zip(row[0:3], offsets[i].tolist())]
                    out.append([round(v, 3) for v in row[:width]] + [colors[k]])
            scene[kind] = out
        return scene

    def racing_mask(self) -> torch.Tensor:
        """(N,) bool: drones whose progress the gate colours should follow. All of them here; the race overrides it."""
        return torch.ones(self.env.n_envs, dtype=torch.bool, device=self.env.device)

    def state_messages(self) -> list[dict]:
        with self._lock:
            messages = [self.scene()]
        if self.info_lines:
            messages.append({"type": "info", "lines": self.info_lines})
        if self.chart_series:
            messages.append({"type": "charts", "series": self.chart_series})
        if self.board is not None:
            messages.append({"type": "board", **self.board})
        return messages
