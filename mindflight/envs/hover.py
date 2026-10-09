"""Level 1 task: hover. Hold still at a point 1 m above the pad, starting from a random messy start.

Each episode a drone appears somewhere within 1 m sideways of the target and between 0.3 m and 2 m high, tipped up
to 30°, drifting at up to 0.5 m/s and spinning at up to 1 rad/s, with the motors at hover speed. It has 10 s.

Reward, per policy step (50 per second):

    reward = 1                          staying alive is worth 1 point a step
           − 1.0 · min(distance, 1)     minus 1 point per metre from the target (capped, so it never goes below 0)
           − 0.2 · (1 − R_zz)           minus a little for tipping over (R_zz = cos(tilt): 1 when level)
           − 0.02 · |ω|²                minus a little for spinning (ω in rad/s)
           − 0.1 · |a − a_prev|²        minus a little for twitchy motor changes
    crash: −5 once, and the episode ends

A crash is: below 5 cm (the floor), upside down (R_zz < 0), more than 3 m from the target, or spinning past 40 rad/s.

Success (the PLAN.md "done when"): the drone never crashed and stayed within 10 cm of the target for the whole second
half of the episode (the last 5 s).
"""

import math

import torch

from mindflight.envs.base import DroneEnv
from mindflight.physics.dynamics import OMEGA, POS, QUAT, VEL
from mindflight.physics.rotations import quat_to_matrix

TARGET_HEIGHT_M = 1.0
SUCCESS_RADIUS_M = 0.10
CRASH_PENALTY = 5.0


class HoverEnv(DroneEnv):
    """Level 1: hover at (0, 0, 1) m in each drone's own world. Task observation: target − position (m, world)."""

    name = "hover"
    task_obs_size = 3
    alive_bonus = 1.0  # reward for every step still flying
    distance_weight = 1.0  # reward lost per metre from the target...
    distance_cap_m = 1.0  # ...counted up to this far, so the reward stays ≥ 0 and crashing never pays
    precision_bonus = 0.0  # extra reward for being very close: bonus · max(0, 1 − distance / 0.3 m)
    inside_bonus = 0.0  # extra reward for every step inside the 10 cm success radius

    def __init__(self, n_envs: int, episode_s: float = 10.0, device: str | None = None, seed: int = 0):
        super().__init__(n_envs, episode_s, device, seed)
        self.target = torch.zeros(n_envs, 3, device=self.device)
        self.target[:, 2] = TARGET_HEIGHT_M
        # Furthest the drone got from the target during the second half of its episode (m).
        self.worst_late_distance = torch.zeros(n_envs, device=self.device)

    def _reset_task(self, ids: torch.Tensor) -> None:
        n = len(ids)
        start = torch.empty(n, 3, device=self.device)
        start[:, 0] = self.uniform(n, -1.0, 1.0)
        start[:, 1] = self.uniform(n, -1.0, 1.0)
        start[:, 2] = self.uniform(n, 0.3, 2.0)
        self.state[ids, POS] = start
        self.state[ids, QUAT] = self.random_tilt(n, math.radians(30))
        self.state[ids, VEL] = self.uniform(n, -0.5, 0.5, size=3)
        self.state[ids, OMEGA] = self.uniform(n, -1.0, 1.0, size=3)
        self.worst_late_distance[ids] = 0.0

    def distance(self) -> torch.Tensor:
        """(N,) metres from each drone to its target."""
        return (self.target - self.state[:, POS]).norm(dim=1)

    def display_objects(self) -> dict[str, torch.Tensor]:
        return {"targets": self.target}

    def _task_obs(self) -> torch.Tensor:
        return self.target - self.state[:, POS]

    def _reward(self) -> torch.Tensor:
        distance = self.distance()
        upright = quat_to_matrix(self.state[:, QUAT])[:, 2, 2]  # R_zz = cos(tilt)
        spin = (self.state[:, OMEGA] ** 2).sum(dim=1)
        twitch = ((self.action - self.prev_action) ** 2).sum(dim=1)

        late = self.step_count > self.max_steps // 2
        self.worst_late_distance = torch.where(
            late, torch.maximum(self.worst_late_distance, distance), self.worst_late_distance
        )

        precision = self.precision_bonus * (1.0 - distance / 0.3).clamp_min(0.0)
        precision = precision + self.inside_bonus * (distance < SUCCESS_RADIUS_M).float()
        reward = self.alive_bonus + precision - self.distance_weight * distance.clamp(max=self.distance_cap_m) - 0.2 * (1.0 - upright) - 0.02 * spin - 0.1 * twitch
        return torch.where(self._crashed(), reward - CRASH_PENALTY, reward)

    def _crashed(self) -> torch.Tensor:
        upright = quat_to_matrix(self.state[:, QUAT])[:, 2, 2]
        return (
            (self.state[:, 2] < 0.05)
            | (upright < 0.0)
            | self._out_of_bounds()
            | (self.state[:, OMEGA].norm(dim=1) > 40.0)
        )

    def _out_of_bounds(self) -> torch.Tensor:
        """(N,) bool: flown too far away. Hover: more than 3 m from the target."""
        return self.distance() > 3.0

    def _episode_info(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        survived = self.step_count[ids] >= self.max_steps
        success = survived & ~self._crashed()[ids] & (self.worst_late_distance[ids] < SUCCESS_RADIUS_M)
        return {"success": success.float(), "final_distance_m": self.distance()[ids]}
