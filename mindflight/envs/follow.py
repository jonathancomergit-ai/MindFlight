"""Phase 2 · P6 task: follow a walking person, circle them, or fly to the spot they point at. One policy, three modes.

Each drone has its own world with one simulated **person** in it. The person walks a random, smooth path inside a
10 × 10 m arena: they turn, speed up, slow down, sometimes stop, sometimes jog. For collisions and drawing the person
is an upright cylinder, 0.25 m radius and 1.75 m tall, standing on the floor.

The three modes (the policy is told which one with a one-hot, and the mode can change mid-episode):

| Mode | Target point (m, world) |
|---|---|
| follow | 2 m behind the person (behind = against their heading ψ), 1.5 m up:  p − 2 · (cos ψ, sin ψ), z = 1.5 |
| orbit | on a 2 m circle round the person, 1.5 m up, going round at ω = 0.3–0.7 rad/s (either way):  p + 2 · (cos θ, sin θ), θ += ω · dt |
| point | the person stops and "points": a spot 2–4 m away in some direction, 1.5 m up. A new spot every 4–6 s. Until the first spot comes, hold the follow point. |

**Sensing the person.** The policy never sees the true person. It sees the latest measurement from a person sensor
(`person_sensor=`, see `person_sensor.py`): "camera" (default: a camera on the drone's nose, 30 Hz, 60–100 ms late,
σ = 2 % of range + 3 cm, only within its 60–80° view), "gps_rtk" (10 Hz, 100 ms, 2–3 cm), "gps" (a phone: 1–5 Hz,
2–3 m wandering error) or "perfect". The target the policy is shown is worked out from that measurement. The reward
and the scores use the *true* target, so the policy has to cope (for example lead a jogging person a little, and
with the camera, turn to keep the person in view).

Task observation (16 numbers, all from the measurement):

    0:3    target − drone position (m, world)        (the same first 3 numbers as waypoint, so a waypoint brain can start here)
    3:6    target velocity (m/s, world)               (how the target is moving: person velocity + turning / orbiting)
    6:9    person body centre (z = 0.9 m) − drone position (m)
    9:12   person velocity − drone velocity (m/s)
    12:15  mode one-hot: follow, orbit, point
    15     age of the measurement (s, up to 5): its delay plus the time since it came. With the camera it keeps
           growing while the person is out of view.

**Safety.** The drone must keep at least 1 m from the person's body (the cylinder's surface):
- closer than 0.7 m = crash (−30, episode over),
- between 0.7 m and 1.0 m: a soft penalty that grows from 0 to 1 point a step,
- below 0.5 m height within 2 m (sideways) of the person = crash.

Reward per policy step, like waypoint (see `waypoint.py`), with d = distance to the true target (m) and g = distance
to the person's body (m):

    reward = 3 − 0.5 · min(d, 4) + 0.5 · max(0, 1 − d / 0.3) + 0.5 · [d < 0.1]
             − 0.2 · (1 − R_zz) − 0.02 · |ω|² − 0.1 · |a − a_prev|²          (tilt, spin, twitch, as in hover)
             − 1.0 · clamp((1.0 − g) / 0.3, 0, 1)                             (too close: 0 at 1 m, 1 at 0.7 m)
             − 0.3 · closing speed · clamp((1.5 − g) / 0.5, 0, 1)             (flying at the person, within 1.5 m)
    then clamped at ≥ 0;  + 0.3 while the person is in the camera's view (camera only)
    crash: −30 once, and the episode ends

Crashing must never pay (docs/level1-6 lessons), so the step reward is never below 0 before a crash. The alive
bonus 3 covers the worst normal case (4 m off and at the 0.7 m line: 3 − 2 − 1 = 0); the clamp covers the rest.

Why the closing-speed term: starting from the waypoint brain without it, 55 % of episodes crashed into the person and stayed there for 300
iterations. Measured: the drone flew straight at a pointed spot at ~3.4 m/s, through the person, 0.9 s after the
spot came. The 0.7–1.0 m soft zone is crossed in under 0.1 s at that speed, too short to learn from. The closing
term says "don't fly *at* the person" from 1.5 m out, while there is still time to curve round. (It started at
2 m, but the follow and orbit points are 1.75 m from the body, so it charged the drone for doing its job whenever the
person slowed down, and the brain learned to hang back 1.5 m behind the follow point.)

Scores (per episode; the doc's "done when" uses them over 1,000 fixed-seed episodes):
- follow: fraction of follow time within 0.3 m of the true follow point, not counting the first 2 s in the mode,
- orbit: fraction of orbit time with radius error |sideways distance to the person − 2 m| < 0.3 m, after 2 s,
- point: fraction of pointed spots reached (within 15 cm, held for 0.5 s) before the next spot comes.
"""

import math

import torch

from mindflight.envs.base import POLICY_DT_S
from mindflight.envs.hover import SUCCESS_RADIUS_M, HoverEnv
from mindflight.envs.person_sensor import PersonTracker, wrap_angle
from mindflight.physics.dynamics import OMEGA, POS, QUAT, VEL
from mindflight.physics.rotations import quat_from_axis_angle, quat_multiply, quat_to_matrix

# The person (ASSUMED: an average adult as a simple cylinder).
PERSON_RADIUS_M = 0.25
PERSON_HEIGHT_M = 1.75
PERSON_CENTRE_Z_M = 0.9  # the "position" the camera reports: roughly the hips
ARENA_M = 5.0  # the person stays in |x|, |y| ≤ 5 m
WALL_ZONE_M = 2.0  # closer than this to the arena edge: turn back toward the middle, no jogging
WALK_SPEED_MPS = (0.5, 1.5)
JOG_SPEED_MPS = (2.5, 3.0)
STOP_CHANCE = 0.15  # each new gait: stop 15 %, jog 15 %, walk 70 %
JOG_CHANCE = 0.15
GAIT_S = (2.0, 5.0)  # a gait (stop / walk / jog) lasts this long
SPEED_UP_MPS2 = 1.5
SLOW_DOWN_MPS2 = 2.5
TURN_TAU_S = 2.0  # turn rate: a random walk pulled back to 0 (Ornstein–Uhlenbeck), time constant...
TURN_SIGMA_RAD_S = 0.3  # ...and typical size. (0.5 rad/s with τ = 1 s made people weave all the time.)
TURN_MAX_RAD_S = 1.2
SIDEWAYS_ACCEL_MAX_MPS2 = 3.0  # turn rate ≤ this / speed: a jogger can't turn on the spot
TURN_ACCEL_MAX_RAD_S2 = 2.0  # how fast the turn rate itself can change

# The modes.
MODES = ("follow", "orbit", "point")
FOLLOW, ORBIT, POINT = 0, 1, 2
FLY_HEIGHT_M = 1.5
FOLLOW_BEHIND_M = 2.0
ORBIT_RADIUS_M = 2.0
ORBIT_RATE_RAD_S = (0.3, 0.7)
POINT_DISTANCE_M = (2.0, 4.0)
POINT_EVERY_S = (4.0, 6.0)
SPOT_LIMIT_M = 6.0  # pointed spots stay in |x|, |y| ≤ 6 m
MODE_SWITCH_S = (4.0, 10.0)  # with mode switching on, a new mode comes this often

# How the drone senses the person: see person_sensor.py.
AGE_CAP_S = 5.0  # the "how old is my measurement" input is capped here
VIEW_BONUS = 0.3  # camera: reward per step while the person is in view
START_YAW_RAD = math.radians(30)  # flights start with the nose within ±30° of the person

# Safety and the arena for the drone.
CRASH_DISTANCE_M = 0.7
SAFE_DISTANCE_M = 1.0
LOW_HEIGHT_M = 0.5
LOW_NEAR_M = 2.0
DRONE_LIMIT_XY_M = 8.5
DRONE_TOP_M = 4.0
CRASH_PENALTY = 30.0
PROXIMITY_WEIGHT = 1.0
CLOSING_ZONE_M = 1.5  # within 1.5 m of the body, flying *toward* the person costs a little (full inside 1 m)...
CLOSING_WEIGHT = 0.3  # ...0.3 points a step per m/s of closing speed...
CLOSING_MAX_MPS = 3.0  # ...counted up to 3 m/s

# Curriculum (see update_curriculum).
CURRICULUM_CRASH = 0.10
CURRICULUM_RAMP = 800

# Scoring.
SETTLE_S = 2.0
FOLLOW_TOLERANCE_M = 0.3
ORBIT_TOLERANCE_M = 0.3
POINT_TOLERANCE_M = 0.15
POINT_HOLD_S = 0.5


def person_distance(drone_pos: torch.Tensor, person_xy: torch.Tensor) -> torch.Tensor:
    """(N,) distance (m) from each drone centre to the surface of its person's body cylinder (0 if inside).

    sideways gap  g = max(|drone_xy − person_xy| − r, 0)
    up/down gap   h = max(z − height, 0) + max(−z, 0)
    distance      = √(g² + h²)
    """
    sideways = (drone_pos[:, :2] - person_xy).norm(dim=1)
    gap = (sideways - PERSON_RADIUS_M).clamp_min(0.0)
    above = (drone_pos[:, 2] - PERSON_HEIGHT_M).clamp_min(0.0) + (-drone_pos[:, 2]).clamp_min(0.0)
    return torch.sqrt(gap**2 + above**2)


class FollowEnv(HoverEnv):
    """P6: follow, orbit or fly to a pointed spot, around a simulated walking person. Attitude actions by default.

    mode: "all" (a random mode each episode, switching mid-episode if switch_modes) or one of "follow", "orbit",
          "point" (that mode only, for testing one at a time).
    person_sensor: how the drone senses the person: "camera" (default), "gps_rtk", "gps" or "perfect"
          (person_sensor.py).
    """

    name = "follow"
    task_obs_size = 16
    has_curriculum = True
    alive_bonus = 3.0
    distance_weight = 0.5
    distance_cap_m = 4.0
    precision_bonus = 0.5
    inside_bonus = 0.5

    def __init__(
        self,
        n_envs: int,
        episode_s: float = 20.0,
        device: str | None = None,
        seed: int = 0,
        mode: str = "all",
        switch_modes: bool = True,
        person_sensor: str = "camera",
        difficulty: float | None = None,
        start_difficulty: float = 0.0,
        action_mode: str = "attitude",
        **options,
    ):
        """difficulty: fix the curriculum here (1 = the full task, used for testing); None = let it climb from
        start_difficulty as training succeeds (see `update_curriculum`)."""
        super().__init__(n_envs, episode_s, device, seed, action_mode=action_mode, **options)
        self.fixed_difficulty = difficulty is not None
        self.difficulty = float(start_difficulty) if difficulty is None else float(difficulty)
        self.smoothed_crash = 1.0
        self.curriculum_calls = 0
        if mode != "all" and mode not in MODES:
            raise ValueError(f"mode must be 'all' or one of {MODES}, not {mode!r}")
        self.mode_choice = mode
        self.switch_modes = switch_modes and mode == "all"
        n, dev = n_envs, self.device
        self.tracker = PersonTracker(person_sensor, n, dev, self.generator, POLICY_DT_S)
        self.person_sensor = person_sensor

        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(n, *shape, device=dev, dtype=dtype)

        # The person (true state).
        self.person_xy = zeros(2)
        self.person_heading = zeros()  # ψ, rad, 0 = +x
        self.person_speed = zeros()  # m/s, along the heading
        self.person_turn_rate = zeros()  # rad/s, what the person actually turned this step
        self.turn_drive = zeros()  # rad/s, the random-walk part of the turn rate
        self.speed_goal = zeros()
        self.gait_steps_left = zeros(dtype=torch.long)

        # Modes.
        self.mode = zeros(dtype=torch.long)
        self.mode_steps = zeros(dtype=torch.long)  # steps since this mode began
        self.switch_steps_left = zeros(dtype=torch.long)
        self.orbit_angle = zeros()  # θ, rad
        self.orbit_rate = zeros()  # ω, rad/s
        self.spot = zeros(3)
        self.has_spot = zeros(dtype=torch.bool)
        self.spot_steps = zeros(dtype=torch.long)  # steps since this spot was pointed at
        self.spot_window_steps = zeros(dtype=torch.long)  # how long until the next spot
        self.spot_close_steps = zeros(dtype=torch.long)  # in a row within 15 cm
        self.spot_reached = zeros(dtype=torch.bool)

        # Targets: true (for reward and scores) and as seen through the camera (for the observation).
        self.target_vel = zeros(3)
        self.seen_target = zeros(3)
        self.seen_target_vel = zeros(3)

        # Scores, per episode (in steps / counts).
        self.follow_scored = zeros()
        self.follow_good = zeros()
        self.orbit_scored = zeros()
        self.orbit_good = zeros()
        self.spots_scored = zeros()
        self.spots_reached = zeros()
        self.close_steps = zeros()  # steps closer than 1 m to the person
        self.min_person_m = zeros()

        self.settle_steps = round(SETTLE_S / POLICY_DT_S)
        self.hold_steps = round(POINT_HOLD_S / POLICY_DT_S)
        self.min_window_steps = round(POINT_EVERY_S[0] / POLICY_DT_S)

    # What the drone senses of the person (person_sensor.py): the latest measurement.
    @property
    def seen_xy(self) -> torch.Tensor:
        return self.tracker.seen_xy

    @property
    def seen_vel(self) -> torch.Tensor:
        return self.tracker.seen_vel

    @property
    def seen_heading(self) -> torch.Tensor:
        return self.tracker.seen_heading

    @property
    def seen_turn_rate(self) -> torch.Tensor:
        return self.tracker.seen_turn_rate

    # ---------- random pieces ----------

    def random_modes(self, n: int) -> torch.Tensor:
        if self.mode_choice != "all":
            return torch.full((n,), MODES.index(self.mode_choice), dtype=torch.long, device=self.device)
        return torch.randint(0, len(MODES), (n,), generator=self.generator, device=self.device)

    def random_steps(self, n: int, seconds: tuple[float, float]) -> torch.Tensor:
        return (self.uniform(n, *seconds) / POLICY_DT_S).round().long()

    def random_gait_speeds(self, n: int) -> torch.Tensor:
        """A new speed goal (m/s): stop (15 %), jog 2.5–3 m/s (15 %) or walk 0.5–1.5 m/s (70 %), at difficulty 1.
        Easier: no jogging at difficulty 0 (jog chance 15 % × difficulty), and walking tops out lower
        (0.75 + 0.75 × difficulty m/s)."""
        d = self.difficulty
        choice = self.uniform(n, 0.0, 1.0)
        walk = self.uniform(n, WALK_SPEED_MPS[0], 0.75 + (WALK_SPEED_MPS[1] - 0.75) * d)
        jog = self.uniform(n, *JOG_SPEED_MPS)
        speed = torch.where(choice < STOP_CHANCE + JOG_CHANCE * d, jog, walk)
        return torch.where(choice < STOP_CHANCE, torch.zeros_like(speed), speed)

    def random_orbit_rates(self, n: int) -> torch.Tensor:
        """ω = ±(0.3–0.7) rad/s at difficulty 1; the top end comes down to 0.3 at difficulty 0."""
        sign = torch.where(self.uniform(n, 0.0, 1.0) < 0.5, -1.0, 1.0)
        low, high = ORBIT_RATE_RAD_S
        return sign * self.uniform(n, low, low + (high - low) * self.difficulty)

    def update_curriculum(self, episode_means: dict[str, float]) -> dict[str, float]:
        """Smoothed crash rate (while training, with exploring actions) below 10 % -> difficulty + 0.05; and in any
        case at least iterations / 800, so the full task always arrives by iteration 800.

        Crashes, not success, drive it: what the difficulty adds (jogging, faster orbits, spots on the far side of
        the person) is what makes the drone hit the person. Started at the full task, the waypoint brain crashed
        into the person in 55–65 % of flights and never learned its way out (docs/phase2-follow.md). The floor
        is there because exploring actions keep the training crash rate well above the test one."""
        self.curriculum_calls += 1
        if "episode_crashed" in episode_means:
            self.smoothed_crash = 0.8 * self.smoothed_crash + 0.2 * episode_means["episode_crashed"]
        if not self.fixed_difficulty and self.difficulty < 1.0:
            if self.smoothed_crash < CURRICULUM_CRASH:
                self.difficulty = min(1.0, round(self.difficulty + 0.05, 2))
                self.smoothed_crash = CURRICULUM_CRASH + 0.02  # wait a few iterations before the next step
            self.difficulty = max(self.difficulty, min(1.0, round(self.curriculum_calls / CURRICULUM_RAMP, 3)))
        return {"difficulty": self.difficulty}

    # ---------- reset ----------

    def _reset_task(self, ids: torch.Tensor) -> None:
        super()._reset_task(ids)  # messy start (tilt, drift, spin), as in hover; the position is set below
        n = len(ids)
        self.person_xy[ids] = self.uniform(n, -(ARENA_M - 1.0), ARENA_M - 1.0, size=2)
        self.person_heading[ids] = self.uniform(n, -math.pi, math.pi)
        self.speed_goal[ids] = self.random_gait_speeds(n)
        self.gait_steps_left[ids] = self.random_steps(n, GAIT_S)
        self.turn_drive[ids] = 0.0
        self.person_turn_rate[ids] = 0.0
        mode = self.random_modes(n)
        self.mode[ids] = mode
        # Pointing means standing still, so a "point" episode starts with the person standing.
        self.person_speed[ids] = torch.where(mode == POINT, torch.zeros(n, device=self.device), self.speed_goal[ids])
        self.mode_steps[ids] = 0
        self.switch_steps_left[ids] = self.random_steps(n, MODE_SWITCH_S)
        self.orbit_angle[ids] = self.uniform(n, -math.pi, math.pi)
        self.orbit_rate[ids] = self.random_orbit_rates(n)
        self.has_spot[ids] = False
        self.spot_steps[ids] = 0
        self.spot_close_steps[ids] = 0
        self.spot_reached[ids] = False
        self.spot[ids] = 0.0

        for name in ("follow_scored", "follow_good", "orbit_scored", "orbit_good", "spots_scored", "spots_reached",
                     "close_steps"):
            getattr(self, name)[ids] = 0.0
        self.min_person_m[ids] = float("inf")

        self.update_targets()  # the true target (the seen one is redone below)
        # Start near the target (±0.3 m, so at least ~1.2 m from the person), never below 1 m, tipped up to 30°,
        # nose within ±30° of the person (a camera drone takes off looking at you).
        start = self.target[ids] + self.uniform(n, -0.3, 0.3, size=3)
        start[:, 2] = start[:, 2].clamp_min(1.0)
        self.state[ids, POS] = start
        # Already moving with the target, as if it had been flying a while. (Starting still, a drone placed on the
        # orbit in front of a jogger had 0.35 s to reach 3 m/s backwards: impossible, and a crash every time.)
        self.state[ids, VEL] = self.target_vel[ids] + self.uniform(n, -0.3, 0.3, size=3)
        rel = self.person_xy[ids] - start[:, :2]
        yaw = torch.atan2(rel[:, 1], rel[:, 0]) + self.uniform(n, -START_YAW_RAD, START_YAW_RAD)
        up = torch.zeros(n, 3, device=self.device)
        up[:, 2] = 1.0
        tilt_axis = torch.zeros(n, 3, device=self.device)
        tilt_axis[:, :2] = torch.randn(n, 2, generator=self.generator, device=self.device)
        tilt = quat_from_axis_angle(tilt_axis, self.uniform(n, 0.0, math.radians(30)))
        self.state[ids, QUAT] = quat_multiply(tilt, quat_from_axis_angle(up, yaw))
        self.tracker.reset(ids, self.person_record(ids), start)
        self.update_targets()

    def person_record(self, ids: torch.Tensor | None = None) -> torch.Tensor:
        """(n, 6) the person's true state as one history row: x, y, heading, vx, vy, turn rate."""
        sl = slice(None) if ids is None else ids
        heading, speed = self.person_heading[sl], self.person_speed[sl]
        return torch.stack(
            (self.person_xy[sl, 0], self.person_xy[sl, 1], heading, speed * torch.cos(heading),
             speed * torch.sin(heading), self.person_turn_rate[sl]),
            dim=1,
        )

    # ---------- the person ----------

    def move_person(self) -> None:
        """One policy step (dt = 0.02 s) of the walking model, for every person at once.

        speed:   moves toward the gait's speed goal, at most +1.5 m/s² / −2.5 m/s²
        turning: u ← u − u·dt/τ + σ·√(2·dt/τ)·noise   (a smooth random turn drive, τ = 2 s, σ = 0.3 rad/s)
                 near the arena edge and heading outward: turn toward the middle as hard as allowed
                 |turn rate| ≤ min(1.2 rad/s, 3 m/s² / speed); standing people turn slowly;
                 and the turn rate changes by at most 2 rad/s² (no snapping from left to right)
        position: p ← p + speed · (cos ψ, sin ψ) · dt, kept inside |x|, |y| ≤ 5 m
        """
        dt = POLICY_DT_S
        n = self.n_envs
        self.gait_steps_left -= 1
        new_gait = self.gait_steps_left <= 0
        self.speed_goal = torch.where(new_gait, self.random_gait_speeds(n), self.speed_goal)
        self.gait_steps_left = torch.where(new_gait, self.random_steps(n, GAIT_S), self.gait_steps_left)

        xy = self.person_xy
        edge_gap = ARENA_M - xy.abs().max(dim=1).values
        near_wall = edge_gap < WALL_ZONE_M
        goal = torch.where(near_wall, self.speed_goal.clamp(max=WALK_SPEED_MPS[1]), self.speed_goal)
        goal = torch.where(self.mode == POINT, torch.zeros_like(goal), goal)  # pointing = standing still
        change = (goal - self.person_speed).clamp(-SLOW_DOWN_MPS2 * dt, SPEED_UP_MPS2 * dt)
        self.person_speed = (self.person_speed + change).clamp(0.0, JOG_SPEED_MPS[1])

        noise = torch.randn(n, generator=self.generator, device=self.device)
        self.turn_drive = (
            self.turn_drive - self.turn_drive * dt / TURN_TAU_S + TURN_SIGMA_RAD_S * math.sqrt(2 * dt / TURN_TAU_S) * noise
        )
        limit = torch.minimum(
            torch.full_like(self.person_speed, TURN_MAX_RAD_S), SIDEWAYS_ACCEL_MAX_MPS2 / self.person_speed.clamp_min(1e-3)
        )
        heading = self.person_heading
        turn = self.turn_drive * (self.person_speed / 0.5).clamp(0.25, 1.0)
        to_middle = wrap_angle(torch.atan2(-xy[:, 1], -xy[:, 0]) - heading)
        outward = (xy[:, 0] * torch.cos(heading) + xy[:, 1] * torch.sin(heading)) > 0
        turn = torch.where(near_wall & outward, torch.sign(to_middle) * limit, turn)
        turn = torch.maximum(torch.minimum(turn, limit), -limit)
        # People don't snap from turning left to turning right: the turn rate changes by at most 2 rad/s².
        step = TURN_ACCEL_MAX_RAD_S2 * dt
        turn = self.person_turn_rate + (turn - self.person_turn_rate).clamp(-step, step)
        self.person_turn_rate = turn
        self.person_heading = wrap_angle(heading + turn * dt)
        direction = torch.stack((torch.cos(self.person_heading), torch.sin(self.person_heading)), dim=1)
        self.person_xy = (xy + self.person_speed[:, None] * direction * dt).clamp(-ARENA_M, ARENA_M)

    # ---------- targets ----------

    def mode_target(
        self, xy: torch.Tensor, vel: torch.Tensor, heading: torch.Tensor, turn_rate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Each drone's target point and its velocity, (N, 3) each, for a person at xy moving at vel.

        follow: p − b·h, with h = (cos ψ, sin ψ) and b = 2 m;   velocity v − b·ψ̇·(−sin ψ, cos ψ)
        orbit:  p + r·(cos θ, sin θ), r = 2 m;                    velocity v + r·ω·(−sin θ, cos θ)
        point:  the pointed spot (still), or the follow point while waiting for the first spot
        Height: always 1.5 m, not moving up or down.
        """
        cos_h, sin_h = torch.cos(heading), torch.sin(heading)
        follow_xy = xy - FOLLOW_BEHIND_M * torch.stack((cos_h, sin_h), dim=1)
        follow_v = vel - FOLLOW_BEHIND_M * turn_rate[:, None] * torch.stack((-sin_h, cos_h), dim=1)
        cos_o, sin_o = torch.cos(self.orbit_angle), torch.sin(self.orbit_angle)
        orbit_xy = xy + ORBIT_RADIUS_M * torch.stack((cos_o, sin_o), dim=1)
        orbit_v = vel + ORBIT_RADIUS_M * self.orbit_rate[:, None] * torch.stack((-sin_o, cos_o), dim=1)

        mode = self.mode[:, None]
        target_xy = torch.where(mode == ORBIT, orbit_xy, follow_xy)
        target_v = torch.where(mode == ORBIT, orbit_v, follow_v)
        pointing = (self.mode == POINT) & self.has_spot
        target_xy = torch.where(pointing[:, None], self.spot[:, :2], target_xy)
        target_v = torch.where(pointing[:, None], torch.zeros_like(target_v), target_v)
        height = torch.full_like(target_xy[:, :1], FLY_HEIGHT_M)
        return torch.cat((target_xy, height), dim=1), torch.cat((target_v, torch.zeros_like(height)), dim=1)

    def update_targets(self) -> None:
        """Recompute the true target (from the true person) and the seen one (from the camera estimate)."""
        true_vel = self.person_speed[:, None] * torch.stack(
            (torch.cos(self.person_heading), torch.sin(self.person_heading)), dim=1
        )
        self.target, self.target_vel = self.mode_target(
            self.person_xy, true_vel, self.person_heading, self.person_turn_rate
        )
        self.seen_target, self.seen_target_vel = self.mode_target(
            self.seen_xy, self.seen_vel, self.seen_heading, self.seen_turn_rate
        )

    def point_new_spots(self, which: torch.Tensor) -> None:
        """The persons where `which` (N,) is True point: a spot 2–4 m from where the camera sees them, in a random
        direction, 1.5 m up. (Worked out for everyone, kept where `which`: no slow GPU -> CPU index lookups.)

        Direction: within ±(60° + 120° × difficulty) of the drone's side of the person, so easy spots never make the
        drone fly past the person (at difficulty 1: any direction). Spots stay inside |x|, |y| ≤ 6 m: 8 candidate
        spots are drawn and pulled inside the arena, and the first one still 2–4 m from the person is used (if
        none is, the person points square to the drone's side, inward). A first version flipped spots that left the
        arena back through the person; at the edge that sent the drone straight through them.)"""
        n, candidates = self.n_envs, 8
        drone_side = torch.atan2(self.state[:, 1] - self.seen_xy[:, 1], self.state[:, 0] - self.seen_xy[:, 0])
        cone = math.pi / 3 + (2 * math.pi / 3) * self.difficulty
        angle = drone_side[:, None] + cone * (2 * self.uniform(n, 0.0, 1.0, size=candidates) - 1)  # (N, 8)
        reach = self.uniform(n, *POINT_DISTANCE_M)[:, None, None]
        offsets = reach * torch.stack((torch.cos(angle), torch.sin(angle)), dim=2)  # (N, 8, 2)
        inside = (self.seen_xy[:, None, :] + offsets).clamp(-SPOT_LIMIT_M, SPOT_LIMIT_M)
        offsets = inside - self.seen_xy[:, None, :]
        reach_now = offsets.norm(dim=2)
        fits = (reach_now >= POINT_DISTANCE_M[0]) & (reach_now <= POINT_DISTANCE_M[1])  # (N, 8)
        first = fits.int().argmax(dim=1)  # index of the first candidate that fits (0 if none does)
        offset = offsets.gather(1, first.view(-1, 1, 1).expand(n, 1, 2))[:, 0]
        # None fits: point square to the drone's side (no flying past the person), whichever way is more inward.
        side = torch.stack((-torch.sin(drone_side), torch.cos(drone_side)), dim=1)
        side = torch.where(((side * self.seen_xy).sum(dim=1) > 0)[:, None], -side, side)
        fallback = (self.seen_xy + reach[:, 0] * side).clamp(-SPOT_LIMIT_M, SPOT_LIMIT_M) - self.seen_xy
        offset = torch.where(fits.any(dim=1)[:, None], offset, fallback)
        spot = torch.cat((self.seen_xy + offset, torch.full_like(offset[:, :1], FLY_HEIGHT_M)), dim=1)
        self.spot = torch.where(which[:, None], spot, self.spot)
        self.has_spot = self.has_spot | which
        zero = torch.zeros_like(self.spot_steps)
        self.spot_steps = torch.where(which, zero, self.spot_steps)
        self.spot_close_steps = torch.where(which, zero, self.spot_close_steps)
        self.spot_reached = self.spot_reached & ~which
        self.spot_window_steps = torch.where(which, self.random_steps(n, POINT_EVERY_S), self.spot_window_steps)

    def score_spots(self, which: torch.Tensor, crashed: torch.Tensor | None = None) -> None:
        """Close the books on the current spot where `which` (N,) is True: it counts if it was reached, or if it had
        its fair window (≥ 4 s), or if the drone crashed. A spot cut short by a mode switch or the clock is not
        counted. A spot pointed at on the very step of a crash (spot_steps = 0) isn't counted either."""
        fair = self.spot_reached | (self.spot_steps >= self.min_window_steps)
        if crashed is not None:
            fair = fair | (crashed & (self.spot_steps > 0))
        counted = which & self.has_spot & fair
        self.spots_scored += counted.float()
        self.spots_reached += (counted & self.spot_reached).float()
        self.has_spot = self.has_spot & ~which

    # ---------- the env hooks ----------

    def _task_obs(self) -> torch.Tensor:
        pos = self.state[:, POS]
        person = torch.cat((self.seen_xy, torch.full_like(self.seen_xy[:, :1], PERSON_CENTRE_Z_M)), dim=1)
        person_vel = torch.cat((self.seen_vel, torch.zeros_like(self.seen_vel[:, :1])), dim=1)
        one_hot = torch.nn.functional.one_hot(self.mode, len(MODES)).float()
        age = self.tracker.age_s.clamp(max=AGE_CAP_S)[:, None]
        return torch.cat(
            (self.seen_target - pos, self.seen_target_vel, person - pos, person_vel - self.state[:, VEL], one_hot, age),
            dim=1,
        )

    def drone_yaw(self) -> torch.Tensor:
        """(N,) which way each drone's nose points, seen from above (rad, 0 = +x): the body x axis in the world."""
        rotation = quat_to_matrix(self.state[:, QUAT])
        return torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])

    def person_gap(self) -> torch.Tensor:
        """(N,) metres from each drone to its person's body."""
        return person_distance(self.state[:, POS], self.person_xy)

    def _reward(self) -> torch.Tensor:
        distance = self.distance()
        upright = quat_to_matrix(self.state[:, QUAT])[:, 2, 2]
        spin = (self.state[:, OMEGA] ** 2).sum(dim=1)
        twitch = ((self.action - self.prev_action) ** 2).sum(dim=1)
        gap = self.person_gap()
        too_close = ((SAFE_DISTANCE_M - gap) / (SAFE_DISTANCE_M - CRASH_DISTANCE_M)).clamp(0.0, 1.0)
        reward = (
            self.alive_bonus
            - self.distance_weight * distance.clamp(max=self.distance_cap_m)
            + self.precision_bonus * (1.0 - distance / 0.3).clamp_min(0.0)
            + self.inside_bonus * (distance < SUCCESS_RADIUS_M).float()
            - 0.2 * (1.0 - upright)
            - 0.02 * spin
            - 0.1 * twitch
            - PROXIMITY_WEIGHT * too_close
            - CLOSING_WEIGHT * self.closing_speed() * ((CLOSING_ZONE_M - gap) / (CLOSING_ZONE_M - SAFE_DISTANCE_M)).clamp(0.0, 1.0)
        ).clamp_min(0.0)
        if self.tracker.profile.fov_deg is not None:  # camera: keeping the person in view pays a little
            reward = reward + VIEW_BONUS * self.tracker.visible.float()
        return torch.where(self._crashed(), reward - CRASH_PENALTY, reward)

    def closing_speed(self) -> torch.Tensor:
        """(N,) how fast each drone is closing in on its person, sideways (m/s, 0 if moving apart, at most 3):
        (v_drone − v_person) · û, with û the level unit vector from the drone toward the person."""
        rel = self.person_xy - self.state[:, POS][:, :2]
        toward = rel / rel.norm(dim=1, keepdim=True).clamp_min(1e-6)
        person_vel = self.person_speed[:, None] * torch.stack(
            (torch.cos(self.person_heading), torch.sin(self.person_heading)), dim=1
        )
        return ((self.state[:, VEL][:, :2] - person_vel) * toward).sum(dim=1).clamp(0.0, CLOSING_MAX_MPS)

    def hit_person(self) -> torch.Tensor:
        """(N,) bool: closer than 0.7 m to the person, or below 0.5 m within 2 m (sideways) of them."""
        pos = self.state[:, POS]
        sideways = (pos[:, :2] - self.person_xy).norm(dim=1)
        return (self.person_gap() < CRASH_DISTANCE_M) | ((pos[:, 2] < LOW_HEIGHT_M) & (sideways < LOW_NEAR_M))

    def _crashed(self) -> torch.Tensor:
        upright = quat_to_matrix(self.state[:, QUAT])[:, 2, 2]
        return (
            (self.state[:, 2] < 0.05)
            | (upright < 0.0)
            | self._out_of_bounds()
            | (self.state[:, OMEGA].norm(dim=1) > 40.0)
            | self.hit_person()
        )

    def _out_of_bounds(self) -> torch.Tensor:
        pos = self.state[:, POS]
        return (pos[:, :2].abs() > DRONE_LIMIT_XY_M).any(dim=1) | (pos[:, 2] > DRONE_TOP_M)

    def _post_step(self) -> None:
        """Score the step, then: switch modes, move the orbit and the spots, walk the person, update the camera."""
        pos = self.state[:, POS]
        distance = self.distance()
        gap = self.person_gap()
        self.close_steps += (gap < SAFE_DISTANCE_M).float()
        self.min_person_m = torch.minimum(self.min_person_m, gap)
        settled = self.mode_steps >= self.settle_steps
        in_follow = (self.mode == FOLLOW) & settled
        self.follow_scored += in_follow.float()
        self.follow_good += (in_follow & (distance < FOLLOW_TOLERANCE_M)).float()
        in_orbit = (self.mode == ORBIT) & settled
        radius_error = ((pos[:, :2] - self.person_xy).norm(dim=1) - ORBIT_RADIUS_M).abs()
        self.orbit_scored += in_orbit.float()
        self.orbit_good += (in_orbit & (radius_error < ORBIT_TOLERANCE_M)).float()
        pointing = (self.mode == POINT) & self.has_spot
        near_spot = pointing & ((self.spot - pos).norm(dim=1) < POINT_TOLERANCE_M)
        self.spot_close_steps = torch.where(near_spot, self.spot_close_steps + 1, torch.zeros_like(self.spot_close_steps))
        self.spot_reached |= self.spot_close_steps >= self.hold_steps

        # Mode switches.
        self.mode_steps += 1
        self.spot_steps += 1
        # (Everything below is worked out for all drones and kept with torch.where, with no `if any()` checks:
        # each of those would make the CPU wait for the GPU, every step.)
        if self.switch_modes:
            self.switch_steps_left -= 1
            switch = self.switch_steps_left <= 0
            self.score_spots(switch)
            step = torch.randint(1, len(MODES), (self.n_envs,), generator=self.generator, device=self.device)
            self.mode = torch.where(switch, (self.mode + step) % len(MODES), self.mode)
            self.mode_steps = torch.where(switch, torch.zeros_like(self.mode_steps), self.mode_steps)
            self.switch_steps_left = torch.where(
                switch, self.random_steps(self.n_envs, MODE_SWITCH_S), self.switch_steps_left
            )
            # A new orbit starts where the drone is now, so it doesn't have to cross the circle first.
            rel = pos[:, :2] - self.person_xy
            self.orbit_angle = torch.where(switch, torch.atan2(rel[:, 1], rel[:, 0]), self.orbit_angle)
            self.orbit_rate = torch.where(switch, self.random_orbit_rates(self.n_envs), self.orbit_rate)

        # Pointing: the first spot once the person has stopped, then a new one every 4–6 s.
        in_time = self.step_count < self.max_steps
        in_point = self.mode == POINT
        first = in_point & ~self.has_spot & (self.person_speed < 0.1) & (self.mode_steps >= self.hold_steps) & in_time
        again = in_point & self.has_spot & (self.spot_steps >= self.spot_window_steps) & in_time
        self.score_spots(again)
        self.orbit_angle = wrap_angle(self.orbit_angle + self.orbit_rate * POLICY_DT_S)
        self.move_person()
        self.tracker.step(self.person_record(), self.state[:, POS], self.drone_yaw())
        self.point_new_spots(first | again)  # after the camera update: the person points from where they're seen now
        self.update_targets()

    def _episode_info(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        crashed_all = self._crashed()
        crashed = crashed_all[ids]
        ending = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        ending[ids] = True
        self.score_spots(ending, crashed_all)
        dt = POLICY_DT_S
        follow = self.follow_good[ids] / self.follow_scored[ids].clamp_min(1.0)
        orbit = self.orbit_good[ids] / self.orbit_scored[ids].clamp_min(1.0)
        point = self.spots_reached[ids] / self.spots_scored[ids].clamp_min(1.0)
        present = torch.stack((self.follow_scored[ids] > 0, self.orbit_scored[ids] > 0, self.spots_scored[ids] > 0), 1)
        ratios = torch.stack((follow, orbit, point), dim=1)
        success = (ratios * present).sum(1) / present.sum(1).clamp_min(1)
        success = torch.where(crashed, torch.zeros_like(success), success)
        return {
            "success": success,
            "follow_scored_s": self.follow_scored[ids] * dt,
            "follow_good_s": self.follow_good[ids] * dt,
            "orbit_scored_s": self.orbit_scored[ids] * dt,
            "orbit_good_s": self.orbit_good[ids] * dt,
            "spots": self.spots_scored[ids].clone(),
            "spots_reached": self.spots_reached[ids].clone(),
            "close_s": self.close_steps[ids] * dt,
            "hit_person": (crashed & self.hit_person()[ids]).float(),
            "min_person_m": self.min_person_m[ids].clone(),
            "final_distance_m": self.distance()[ids],
        }

    # ---------- the viewer ----------

    def display_objects(self) -> dict[str, torch.Tensor]:
        """The target, the person (a blue cylinder with a head), the camera's guess (a small red ball) and, while
        pointing, a yellow line from the person's shoulder to the spot."""
        n = self.n_envs
        x, y = self.person_xy[:, 0], self.person_xy[:, 1]
        body_height = PERSON_HEIGHT_M - 0.25
        cylinders = torch.stack(
            (x, y, torch.zeros_like(x), torch.full_like(x, body_height), torch.full_like(x, PERSON_RADIUS_M)), dim=1
        )[:, None, :]
        head = torch.stack((x, y, torch.full_like(x, PERSON_HEIGHT_M - 0.13), torch.full_like(x, 0.13)), dim=1)
        seen = torch.stack(
            (self.seen_xy[:, 0], self.seen_xy[:, 1], torch.full_like(x, PERSON_CENTRE_Z_M), torch.full_like(x, 0.06)),
            dim=1,
        )
        lines = torch.zeros(n, 1, 6, device=self.device)
        pointing = (self.mode == POINT) & self.has_spot
        shoulder = torch.stack((x, y, torch.full_like(x, 1.4)), dim=1)
        lines[:, 0, 0:3] = torch.where(pointing[:, None], shoulder, torch.zeros_like(shoulder))
        lines[:, 0, 3:6] = torch.where(pointing[:, None], self.spot, torch.zeros_like(shoulder))
        return {
            "targets": self.target,
            "cylinders": cylinders,
            "cylinders_colors": ["#4f8cff"],
            "spheres": torch.stack((head, seen), dim=1),
            "spheres_colors": ["#f2c9a0", "#ff5c5c"],
            "lines": lines,
            "lines_colors": ["#ffd43b"],
        }
