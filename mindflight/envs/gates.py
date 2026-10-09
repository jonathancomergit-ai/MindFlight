"""Level 3 task: fly through a course of gates, in order, as quickly as you can without touching them.

The course: 6 square gates standing upright. Each gate faces the way the course is going, so you fly through it
along its **normal** n = (cos ψ, sin ψ, 0), where ψ is the gate's heading. Each drone gets its own random course
every episode, built like a walk: turn a random amount, go forward 1.5–2.5 m (up or down a little), put a gate there.

Passing a gate (worked out each step, from where the drone was and where it is now):
    side = (p − c) · n                 which side of the gate's plane the drone is on (− before, + after)
    a crossing is side going from − to +; the point where it crossed is found by straight-line interpolation,
    then its sideways and up-down offsets from the gate centre say *where* in the gate's plane it went through:
      inside the opening (with 6 cm to spare for the drone's own size)   -> passed, on to the next gate
      on the frame (up to 10 cm outside the opening)                      -> crash
      wide of the gate altogether                                         -> nothing, you missed it; go back

Reward, per policy step:
    reward = 10 · (how many metres closer to the next gate's centre than last step)
           + 10 for each gate passed, + 10 more for the last one
           − small spin and twitch terms (no tilt penalty: a racing drone has to lean to speed up)
    crash: −5, and the episode ends

**Curriculum learning.** Starting straight on the hardest course is like learning to read with a novel: nearly
every try fails, so there is almost nothing to learn from. Instead `difficulty` starts at 0 (big 1 m gates in a
nearly straight line) and is raised a step each time the success rate (smoothed) passes 80 %. At difficulty 1 the
openings are 0.5 m, each gate can turn up to 60° from the last and climb or drop 0.5 m.
"""

import math

import torch

from mindflight.envs.base import DroneEnv
from mindflight.physics.dynamics import OMEGA, POS, QUAT, VEL

N_GATES = 6
DRONE_MARGIN_M = 0.06  # the Crazyflie's collision radius (params.collision_radius_m)
FRAME_WIDTH_M = 0.10
CRASH_PENALTY = 5.0
PASS_BONUS = 10.0
START_HEIGHT_M = 1.0


def opening_m(difficulty: float) -> float:
    """Gate opening (square side, m): 1.0 m at difficulty 0 down to 0.5 m at difficulty 1."""
    return 1.0 - 0.5 * difficulty


class GatesEnv(DroneEnv):
    """Level 3: a 6-gate course per drone. Task observation (11 numbers):

        0:3   next gate centre − position (m, world)
        3:5   next gate normal (cos ψ, sin ψ)
        5:8   the gate after it, centre − position (the last gate repeats itself)
        8:10  its normal
        10    opening size (m), so the policy knows how precise it has to be
    """

    name = "gates"
    task_obs_size = 11
    has_curriculum = True

    def __init__(
        self,
        n_envs: int,
        episode_s: float = 20.0,
        device: str | None = None,
        seed: int = 0,
        difficulty: float | None = None,
    ):
        super().__init__(n_envs, episode_s, device, seed)
        # difficulty None = curriculum (start at 0 and climb); a number = fixed (used for evaluation).
        self.fixed_difficulty = difficulty is not None
        self.difficulty = 0.0 if difficulty is None else float(difficulty)
        self.smoothed_success = 0.0

        self.gate_centre = torch.zeros(n_envs, N_GATES, 3, device=self.device)
        self.gate_heading = torch.zeros(n_envs, N_GATES, device=self.device)
        self.gate_opening = torch.zeros(n_envs, device=self.device)  # per drone: set when its course is made
        self.next_gate = torch.zeros(n_envs, dtype=torch.long, device=self.device)
        self.prev_pos = torch.zeros(n_envs, 3, device=self.device)
        self.passed = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
        self.hit_frame = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
        self.finish_time_s = torch.zeros(n_envs, device=self.device)

    # ---------- the course ----------

    def make_courses(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """n random courses at the current difficulty -> (centres (n, 6, 3), headings (n, 6))."""
        d = self.difficulty
        max_turn = math.radians(60) * d
        max_climb = 0.5 * d
        centres = torch.zeros(n, N_GATES, 3, device=self.device)
        headings = torch.zeros(n, N_GATES, device=self.device)
        point = torch.zeros(n, 3, device=self.device)
        point[:, 2] = START_HEIGHT_M
        heading = self.uniform(n, -math.pi, math.pi)
        for k in range(N_GATES):
            if k > 0:
                heading = heading + self.uniform(n, -max_turn, max_turn)
            distance = self.uniform(n, 1.5, 2.5)
            point = point.clone()
            point[:, 0] += distance * torch.cos(heading)
            point[:, 1] += distance * torch.sin(heading)
            point[:, 2] = (point[:, 2] + self.uniform(n, -max_climb, max_climb)).clamp(0.6, 2.0)
            centres[:, k] = point
            headings[:, k] = heading
        return centres, headings

    def _reset_task(self, ids: torch.Tensor) -> None:
        n = len(ids)
        start = torch.zeros(n, 3, device=self.device)
        start[:, :2] = self.uniform(n, -0.2, 0.2, size=2)
        start[:, 2] = START_HEIGHT_M + self.uniform(n, -0.2, 0.2)
        self.state[ids, POS] = start
        self.state[ids, QUAT] = self.random_tilt(n, math.radians(15))
        self.state[ids, VEL] = self.uniform(n, -0.3, 0.3, size=3)
        self.state[ids, OMEGA] = self.uniform(n, -0.5, 0.5, size=3)
        centres, headings = self.make_courses(n)
        self.gate_centre[ids] = centres
        self.gate_heading[ids] = headings
        self.gate_opening[ids] = opening_m(self.difficulty)
        self.next_gate[ids] = 0
        self.prev_pos[ids] = start
        self.passed[ids] = False
        self.hit_frame[ids] = False
        self.finish_time_s[ids] = 0.0

    # ---------- geometry helpers ----------

    def gate(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Centre (N, 3) and unit normal (N, 3) of gate `index` (N,) for each drone; indices past the end clamp."""
        index = index.clamp(max=N_GATES - 1)
        rows = torch.arange(self.n_envs, device=self.device)
        centre = self.gate_centre[rows, index]
        heading = self.gate_heading[rows, index]
        normal = torch.stack((torch.cos(heading), torch.sin(heading), torch.zeros_like(heading)), dim=1)
        return centre, normal

    def _task_obs(self) -> torch.Tensor:
        pos = self.state[:, POS]
        centre, normal = self.gate(self.next_gate)
        centre2, normal2 = self.gate(self.next_gate + 1)
        return torch.cat(
            (centre - pos, normal[:, :2], centre2 - pos, normal2[:, :2], self.gate_opening.unsqueeze(1)), dim=1
        )

    # ---------- the step ----------

    def _reward(self) -> torch.Tensor:
        pos = self.state[:, POS]
        centre, normal = self.gate(self.next_gate)
        progress = (centre - self.prev_pos).norm(dim=1) - (centre - pos).norm(dim=1)

        # Did the drone cross this gate's plane during the step, and where?
        side_before = ((self.prev_pos - centre) * normal).sum(dim=1)
        side_after = ((pos - centre) * normal).sum(dim=1)
        forward = (side_before < 0) & (side_after >= 0)
        backward = (side_before >= 0) & (side_after < 0)
        fraction = (side_before / (side_before - side_after)).nan_to_num(0.0).clamp(0.0, 1.0).unsqueeze(1)
        crossing_point = self.prev_pos + fraction * (pos - self.prev_pos)
        tangent = torch.stack((-normal[:, 1], normal[:, 0], torch.zeros_like(normal[:, 0])), dim=1)
        sideways = ((crossing_point - centre) * tangent).sum(dim=1).abs()
        updown = (crossing_point[:, 2] - centre[:, 2]).abs()
        half = 0.5 * self.gate_opening
        inside = (sideways < half - DRONE_MARGIN_M) & (updown < half - DRONE_MARGIN_M)
        on_frame = (sideways < half + FRAME_WIDTH_M) & (updown < half + FRAME_WIDTH_M) & ~inside
        self.passed = forward & inside & (self.next_gate < N_GATES)
        self.hit_frame = (forward | backward) & on_frame

        spin = (self.state[:, OMEGA] ** 2).sum(dim=1)
        twitch = ((self.action - self.prev_action) ** 2).sum(dim=1)
        last_gate = self.next_gate == N_GATES - 1
        reward = (
            10.0 * progress
            + PASS_BONUS * self.passed.float()
            + PASS_BONUS * (self.passed & last_gate).float()
            - 0.01 * spin
            - 0.05 * twitch
        )
        self.next_gate = self.next_gate + self.passed.long()
        self.finish_time_s = torch.where(
            self.passed & last_gate, self.step_count.float() * 0.02, self.finish_time_s
        )
        return torch.where(self._crashed(), reward - CRASH_PENALTY, reward)

    def _crashed(self) -> torch.Tensor:
        pos = self.state[:, POS]
        centre, _ = self.gate(self.next_gate)
        upright = 1 - 2 * (self.state[:, 4] ** 2 + self.state[:, 5] ** 2)  # R_zz from the quaternion's x and y
        return (
            self.hit_frame
            | (pos[:, 2] < 0.05)
            | (pos[:, 2] > 4.0)
            | (upright < 0.0)
            | ((centre - pos).norm(dim=1) > 6.0)
            | (self.state[:, OMEGA].norm(dim=1) > 40.0)
        )

    def _finished(self) -> torch.Tensor:
        return self.next_gate >= N_GATES

    def _post_step(self) -> None:
        self.prev_pos = self.state[:, POS].clone()

    def _episode_info(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        finished = self.next_gate[ids] >= N_GATES
        return {
            "success": finished.float(),
            "gates_passed": self.next_gate[ids].float(),
        }

    def update_curriculum(self, episode_means: dict[str, float]) -> dict[str, float]:
        """Smooth the success rate; when it is above 80 %, make the courses 0.05 harder (up to 1)."""
        if "episode_success" in episode_means:
            self.smoothed_success = 0.8 * self.smoothed_success + 0.2 * episode_means["episode_success"]
        if not self.fixed_difficulty and self.smoothed_success > 0.8 and self.difficulty < 1.0:
            self.difficulty = min(1.0, round(self.difficulty + 0.05, 2))
            self.smoothed_success = 0.0  # earn the next step on the new courses
        return {"difficulty": self.difficulty}
