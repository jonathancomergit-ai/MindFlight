"""Level 2 task: fly to a point. Every 2.5 s the drone gets a new random target somewhere in a 3 × 3 × 1.5 m box.

It is the hover task with a moving target, so the reward is the same shape (see `hover.py`), with gentler distance
numbers because targets can be over 3 m apart:

    reward = 1 − 0.5 · min(distance, 2) − (tilt, spin and twitch terms as in hover)

A target counts as **reached** when the drone has been within 10 cm of it for the last 0.5 s before the target
changes ("got there and settled", not just flew through). An episode is 10 s, so 4 targets. The episode's
"success" is the fraction of its 4 targets reached; a crash fails the rest.
"""

import torch

from mindflight.envs.hover import SUCCESS_RADIUS_M, HoverEnv
from mindflight.physics.dynamics import POS

TARGET_HOLD_S = 2.5
SETTLE_S = 0.5
BOX_XY_M = 1.5  # targets and starts: x and y in [−1.5, 1.5] m
BOX_Z_M = (0.5, 2.0)  # height range (m)
ARENA_XY_M = 3.0  # leaving |x| or |y| > 3 m, or going above 3.5 m, is a crash
ARENA_TOP_M = 3.5


class WaypointEnv(HoverEnv):
    """Level 2: reach a new random target every 2.5 s. Task observation: target − position (m, world)."""

    name = "waypoint"
    distance_weight = 0.5
    distance_cap_m = 2.0

    def __init__(self, n_envs: int, episode_s: float = 10.0, device: str | None = None, seed: int = 0):
        super().__init__(n_envs, episode_s, device, seed)
        self.hold_steps = round(TARGET_HOLD_S / 0.02)
        self.settle_steps = round(SETTLE_S / 0.02)
        self.targets_per_episode = max(1, self.max_steps // self.hold_steps)
        self.close_steps = torch.zeros(n_envs, dtype=torch.long, device=self.device)  # in a row within 10 cm
        self.targets_reached = torch.zeros(n_envs, device=self.device)

    def random_targets(self, n: int) -> torch.Tensor:
        target = torch.empty(n, 3, device=self.device)
        target[:, 0] = self.uniform(n, -BOX_XY_M, BOX_XY_M)
        target[:, 1] = self.uniform(n, -BOX_XY_M, BOX_XY_M)
        target[:, 2] = self.uniform(n, *BOX_Z_M)
        return target

    def _reset_task(self, ids: torch.Tensor) -> None:
        super()._reset_task(ids)  # messy start, as in hover
        self.state[ids, POS] = self.random_targets(len(ids))  # ...somewhere in the box
        self.target[ids] = self.random_targets(len(ids))
        self.close_steps[ids] = 0
        self.targets_reached[ids] = 0.0

    def _post_step(self) -> None:
        """Count steps spent close to the target; when its 2.5 s are up, score it and pick the next one."""
        close = self.distance() < SUCCESS_RADIUS_M
        self.close_steps = torch.where(close, self.close_steps + 1, torch.zeros_like(self.close_steps))
        switch = (self.step_count % self.hold_steps == 0) & (self.step_count > 0)
        reached = switch & (self.close_steps >= self.settle_steps)
        self.targets_reached += reached.float()
        ids = switch.nonzero(as_tuple=True)[0]
        if len(ids) > 0:
            self.target[ids] = self.random_targets(len(ids))
            self.close_steps[ids] = 0

    def _out_of_bounds(self) -> torch.Tensor:
        pos = self.state[:, POS]
        return (pos[:, 0].abs() > ARENA_XY_M) | (pos[:, 1].abs() > ARENA_XY_M) | (pos[:, 2] > ARENA_TOP_M)

    def _episode_info(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "success": self.targets_reached[ids] / self.targets_per_episode,
            "final_distance_m": self.distance()[ids],
        }
