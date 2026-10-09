"""Level 4 task: cross a field of pillars to a goal on the far side, seeing only with a fake lidar.

The arena (seen from above, x forward):

    start (x = −3.5 m)   |  pillar field, x from −2.4 to +2.4 m  |   goal (x = +3.5 m)
    walls at y = ±3 m, ceiling 3 m

The pillars are vertical cylinders, floor to ceiling, radius 8–20 cm, on a jittered grid (4 columns × 5 rows, each
moved up to 15 cm at random). That grid keeps a gap of at least about 30 cm somewhere in every row, so a path always
exists. The drone does **not** get the pillar positions. It only gets 48 lidar distances (see `sensors.py`), like a
real robot. Touching a pillar (its centre closer than radius + 6 cm, the Crazyflie's size) is a crash.

Observation (task part, 51 numbers):
    0:3    goal − position (m, world)
    3:51   lidar: 48 rays around the drone's heading, each as closeness = 1 − distance / 3 m (0 = nothing in range,
           1 = touching). Closeness rather than distance so that "nothing there" is a calm 0.

Reward, per policy step:
    10 · (metres closer to the goal than last step, counted up to 2 m/s) − small spin and twitch terms
    (Without the 2 m/s cap it learned to blast across at ~6 m/s and hit a pillar 40 % of the time: going faster
    always paid more. With the cap, extra speed earns nothing, so looking where it's going is worth it.)
    reaching the goal (within 20 cm): +20 and the episode ends. Crash: −30 and the episode ends.

Curriculum: each grid spot holds a pillar with probability 0.3 + 0.6 · difficulty, so the field fills up from 6 to
18 pillars on average as the success rate climbs past 80 %.
"""

import math

import torch

from mindflight.envs.base import DroneEnv
from mindflight.envs.sensors import ray_circle_distance, ray_directions, ray_wall_distance
from mindflight.physics.dynamics import OMEGA, POS, QUAT, VEL
from mindflight.physics.rotations import quat_to_matrix

# 48 rays, 7.5° apart: at 1.2 m the gap between neighbouring rays is 16 cm, the thinnest pillar's width. We started
# with 16 rays (22.5°, a 39 cm gap at 1 m) and most crashes were into pillars that slipped between the rays.
N_RAYS = 48
LIDAR_RANGE_M = 3.0
WALL_Y_M = 3.0
CEILING_M = 3.0
START_X_M = -3.5
GOAL_X_M = 3.5
GOAL_RADIUS_M = 0.2
DRONE_RADIUS_M = 0.06  # params.collision_radius_m
COLUMNS_X_M = (-1.8, -0.6, 0.6, 1.8)
ROWS_Y_M = (-2.0, -1.0, 0.0, 1.0, 2.0)
JITTER_M = 0.15
RADIUS_RANGE_M = (0.08, 0.20)
MAX_PAID_SPEED_MPS = 2.0  # progress faster than this earns nothing extra
CRASH_PENALTY = 30.0  # big enough that crashing late still costs more than the progress it banked
GOAL_BONUS = 20.0


class ObstaclesEnv(DroneEnv):
    """Level 4: through the pillars to the goal, by lidar."""

    name = "obstacles"
    task_obs_size = 3 + N_RAYS
    has_curriculum = True

    def __init__(
        self,
        n_envs: int,
        episode_s: float = 15.0,
        device: str | None = None,
        seed: int = 0,
        difficulty: float | None = None,
        start_difficulty: float = 0.0,
        **options,
    ):
        super().__init__(n_envs, episode_s, device, seed, **options)
        self.fixed_difficulty = difficulty is not None
        # start_difficulty: where the curriculum starts (for carrying on from an earlier run's policy).
        self.difficulty = float(start_difficulty) if difficulty is None else float(difficulty)
        self.smoothed_success = 0.0

        n_pillars = len(COLUMNS_X_M) * len(ROWS_Y_M)
        grid_x, grid_y = torch.meshgrid(torch.tensor(COLUMNS_X_M), torch.tensor(ROWS_Y_M), indexing="ij")
        self.grid = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1).to(self.device)  # (P, 2)
        self.pillar_xy = torch.zeros(n_envs, n_pillars, 2, device=self.device)
        self.pillar_radius = torch.zeros(n_envs, n_pillars, device=self.device)  # 0 = empty spot
        self.goal = torch.zeros(n_envs, 3, device=self.device)
        self.prev_pos = torch.zeros(n_envs, 3, device=self.device)
        self.lidar_m = torch.full((n_envs, N_RAYS), LIDAR_RANGE_M, device=self.device)

    def _reset_task(self, ids: torch.Tensor) -> None:
        n = len(ids)
        start = torch.empty(n, 3, device=self.device)
        start[:, 0] = START_X_M + self.uniform(n, -0.2, 0.2)
        start[:, 1] = self.uniform(n, -1.0, 1.0)
        start[:, 2] = self.uniform(n, 0.8, 1.2)
        self.state[ids, POS] = start
        self.state[ids, QUAT] = self.random_tilt(n, math.radians(15))
        self.state[ids, VEL] = self.uniform(n, -0.3, 0.3, size=3)
        self.state[ids, OMEGA] = self.uniform(n, -0.5, 0.5, size=3)
        self.goal[ids, 0] = GOAL_X_M
        self.goal[ids, 1] = self.uniform(n, -1.0, 1.0)
        self.goal[ids, 2] = self.uniform(n, 0.8, 1.5)
        self.prev_pos[ids] = start

        n_spots = self.grid.shape[0]
        self.pillar_xy[ids] = self.grid + self.uniform(n * n_spots, -JITTER_M, JITTER_M, size=2).view(n, n_spots, 2)
        radius = self.uniform(n * n_spots, *RADIUS_RANGE_M).view(n, n_spots)
        present = self.uniform(n * n_spots, 0.0, 1.0).view(n, n_spots) < 0.3 + 0.6 * self.difficulty
        self.pillar_radius[ids] = torch.where(present, radius, torch.zeros_like(radius))
        self.lidar_m[ids] = self.scan()[ids]

    # ---------- sensing ----------

    def heading(self) -> torch.Tensor:
        """(N,) which way the nose points, seen from above (rad): the body x axis, flattened onto the floor."""
        nose = quat_to_matrix(self.state[:, QUAT])[:, :, 0]  # first column of R = body x in world
        return torch.atan2(nose[:, 1], nose[:, 0])

    def scan(self) -> torch.Tensor:
        """(N, 16) lidar distances (m) to the nearest pillar or wall along each ray, at most 3 m."""
        origin = self.state[:, 0:2]
        directions = ray_directions(self.heading(), N_RAYS)
        pillars = ray_circle_distance(origin, directions, self.pillar_xy, self.pillar_radius, LIDAR_RANGE_M)
        walls = ray_wall_distance(origin, directions, WALL_Y_M, LIDAR_RANGE_M)
        return torch.minimum(pillars, walls)

    def _task_obs(self) -> torch.Tensor:
        closeness = 1.0 - self.lidar_m / LIDAR_RANGE_M
        return torch.cat((self.goal - self.state[:, POS], closeness), dim=1)

    def display_objects(self) -> dict[str, torch.Tensor]:
        return {"targets": self.goal, "pillars": torch.cat((self.pillar_xy, self.pillar_radius[..., None]), dim=2)}

    # ---------- the step ----------

    def goal_distance(self) -> torch.Tensor:
        return (self.goal - self.state[:, POS]).norm(dim=1)

    def touching_pillar(self) -> torch.Tensor:
        """(N,) bool: the drone's centre is within radius + 6 cm of a pillar's axis."""
        gap = (self.pillar_xy - self.state[:, None, 0:2]).norm(dim=-1) - self.pillar_radius - DRONE_RADIUS_M
        return ((gap < 0) & (self.pillar_radius > 0)).any(dim=1)

    def _reward(self) -> torch.Tensor:
        self.lidar_m = self.scan()
        progress = (self.goal - self.prev_pos).norm(dim=1) - self.goal_distance()
        progress = progress.clamp(max=MAX_PAID_SPEED_MPS * 0.02)  # metres per policy step
        spin = (self.state[:, OMEGA] ** 2).sum(dim=1)
        twitch = ((self.action - self.prev_action) ** 2).sum(dim=1)
        reward = 10.0 * progress - 0.01 * spin - 0.05 * twitch
        reward = torch.where(self._finished(), reward + GOAL_BONUS, reward)
        return torch.where(self._crashed(), reward - CRASH_PENALTY, reward)

    def _crashed(self) -> torch.Tensor:
        pos = self.state[:, POS]
        upright = quat_to_matrix(self.state[:, QUAT])[:, 2, 2]
        return (
            self.touching_pillar()
            | (pos[:, 2] < 0.05)
            | (pos[:, 2] > CEILING_M)
            | (pos[:, 1].abs() > WALL_Y_M - DRONE_RADIUS_M)
            | (pos[:, 0] < START_X_M - 1.5)
            | (pos[:, 0] > GOAL_X_M + 1.5)
            | (upright < 0.0)
            | (self.state[:, OMEGA].norm(dim=1) > 40.0)
        )

    def _finished(self) -> torch.Tensor:
        """Within 20 cm of the goal (and not touching a pillar while getting there)."""
        return (self.goal_distance() < GOAL_RADIUS_M) & ~self.touching_pillar()

    def _post_step(self) -> None:
        self.prev_pos = self.state[:, POS].clone()

    def _episode_info(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "success": self._finished()[ids].float(),
            "pillars": (self.pillar_radius[ids] > 0).float().sum(dim=1),
            "closest_lidar_m": self.lidar_m[ids].amin(dim=1),
        }

    def update_curriculum(self, episode_means: dict[str, float]) -> dict[str, float]:
        """Same rule as the gates: smoothed success above 80 % -> difficulty + 0.05 (up to 1)."""
        if "episode_success" in episode_means:
            self.smoothed_success = 0.8 * self.smoothed_success + 0.2 * episode_means["episode_success"]
        if not self.fixed_difficulty and self.smoothed_success > 0.8 and self.difficulty < 1.0:
            self.difficulty = min(1.0, round(self.difficulty + 0.05, 2))
            self.smoothed_success = 0.0
        return {"difficulty": self.difficulty}
