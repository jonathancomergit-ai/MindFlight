"""Level 2 task: fly to a point. Every 2.5 s the drone gets a new random target somewhere in a 3 × 3 × 1.5 m box.

It is the hover task with a moving target, so the reward is the same shape (see `hover.py`), with gentler distance
numbers because targets can be over 3 m apart:

    reward = 2 − 0.5 · min(distance, 4) + 0.5 · max(0, 1 − distance / 0.3) − (tilt, spin, twitch as in hover)

           + 0.5 while within 10 cm

The last two terms are a "precision cone" and a flat bonus inside the success radius. Without them, 10 cm off
cost only 0.05 points a step, and the drone was happy to hover 7 cm away (we watched success slide from 91 % to
55 % while the reward stayed flat: the reward and the test were asking for different things).

The distance term reaches out to 4 m at 0.5 points per metre. An earlier version stopped counting at 2 m, and then
the first metre of a long trip earned nothing for hurrying: targets 3+ m away were reached only 32 % of the time,
against 99 % under 2 m. Halving the slope instead (0.25 per metre) made the drone lazy everywhere: 1.5–2.5 m
targets fell from 89 % to 67 %. The alive bonus is 2 so the reward stays ≥ 0 even 4 m out.

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
    alive_bonus = 2.0  # big enough that 0.5 × 4 m never pushes the reward below 0 (or crashing would pay)
    distance_weight = 0.5
    distance_cap_m = 4.0  # targets can be up to ~4.5 m apart: count (almost) all of it
    precision_bonus = 0.5  # the last 30 cm count extra, so it settles right on the target
    inside_bonus = 0.5  # and being inside the 10 cm success radius pays a flat extra

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
        # A drone whose time is up keeps its last target, so the episode's final distance is measured to it.
        ids = (switch & (self.step_count < self.max_steps)).nonzero(as_tuple=True)[0]
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
