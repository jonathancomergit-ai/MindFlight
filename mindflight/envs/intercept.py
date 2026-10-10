"""Phase 2 · P7: intercept thrown balls before they reach a person.

The world (every drone has its own copy, seen from above):

                                                    guard point
        thrower ●  ·  ·  ·  ·  ball  ·  ·  ·  ·  ·  ·  ◎  ·  ▮ person
        6–10 m away, somewhere in an arc              1.2 m      at the origin,
        in front of the person                        in front   facing +x

- **Person**: an upright cylinder at the origin, 0.25 m radius, 1.8 m tall.
- **Guard point**: 1.2 m in front of the person, 1.8 m up. The drone waits here between throws. (Thrown balls arc
  in from above: they pass 1.2 m in front of the person 1.5–2.3 m high.)
- **Throws**: a soft foam ball (25 g, 7 cm across) is thrown at the person's torso from 6–10 m away. The thrower
  stands somewhere in an arc in front of the person (narrow at first, wider as the drone gets better). How long the
  ball is in the air (the flight time) is set by the difficulty: shorter = faster throws. A new ball comes every
  2.5–3 s, so one 15 s episode has about 5 throws and no sample is wasted waiting.

**Ball physics** (batched over all drones): gravity plus quadratic air drag,

    F_drag = ½ · ρ · Cd · A · |v|² against the motion   ->   a = (0, 0, −g) − k · |v| · v,   k = ½ ρ Cd A / m

with ρ = 1.2 kg/m³, Cd = 0.5, A = π r². For our ball k ≈ 0.046 1/m: at 10 m/s drag slows it by 4.6 m/s², about
half of gravity, so drag really matters (a throw aimed without it falls 30–60 cm short). The ball is stepped at the
physics rate (2 ms, 10 steps per policy step) with Heun's method (`ball_step`). Why the physics rate: a 12 m/s ball
moves 24 cm per policy step, more than twice the 9.5 cm touch distance, so checking only every 20 ms could jump right
over a touch. At 2 ms it moves 2.4 cm per check.

**Aiming.** Drag has no neat formula for "which throw lands here in T seconds", so `aim_throws` shoots: throw,
see where the ball is after T seconds, correct the throw by the miss, repeat (10 rounds leave < 1 mm). That is too
slow to do for every throw while training, so the env keeps a **bank** of aimed throws, all with the thrower on the
+x axis, and turns each one to a random direction when it is used. Turning is free because the person is round and
there is no wind.

**Interception** = the drone's 6 cm collision sphere touches the ball: |ball − drone| < 0.06 + 0.035 m, checked at
every 2 ms ball step with the drone's position slid along its path for the step. The ball is then gone (blocked).
**Failure** = the ball touches the person's cylinder or the floor.

**Safety**: the drone's edge (its 6 cm sphere) closer than 0.5 m to the person's body is a crash (−30), exactly
like hitting a wall. So the drone must always stay at least 0.81 m from the person's middle, at body height.

What the drone sees (18 numbers, besides the usual 19). The ball comes from a **camera tracker** that is 20 or 40 ms
late (random per episode) and a little wrong. A real tracker smooths its measurements, so its error changes slowly:
we model it as an offset that is fixed for each throw (1.5 cm, 0.10 m/s) plus a small jitter every step (1.5 cm,
0.05 m/s). The tracker also **predicts** the ball's path from what it measured (`tracker_prediction`: the true path
plus the error its wrong start causes) and reports the point of that path nearest to the drone, among the points
the drone may fly to. With no ball in the air the ball numbers are zeros and the flag is 0.

    0:3    ball − drone (m)                  measured: late and a little wrong
    3:6    ball velocity (m/s)               measured
    6      ball in the air (1) or not (0)
    7:10   nearest point of the predicted path − drone (m): "go here"
    10     time until the ball gets to that point (s): "and be there by then"
    11:14  guard point − drone (m)
    14:17  person's chest (0, 0, 1.2) − drone (m)
    17     distance from the drone to the person's body (m)

Reward per step (50 per second):

    1                                    alive bonus: keeps the reward ≥ 0, so crashing never pays
    while a ball is in the air:
      + 10 · (metres closer to the ball's path this step)    the rest of its true path, 0.9 m or more from the
                                                              person's middle and 0.3–3 m up (where it may fly)
      + 1 · max(0, 1 − distance to the path / 0.3 m)         sit right on the path and the ball runs into you
    when no ball is in the air:
      − 0.2 · min(distance to the guard point, 5 m)          go back and wait (gently: see the doc)
    + 20 for an interception,  − 10 when the ball reaches the person (or the floor)
    + 10 · max(0, 1 − closest the ball came / 0.5 m), once, when a throw ends: near misses earn a little, so there is
      a slope toward catching instead of a cliff at 9.5 cm
    − small spin and twitch terms
    crash: −30 and the episode ends

**Curriculum** (like Level 3's gates): `difficulty` starts at 0 (a narrow ±10° arc, 1.1–1.3 s flights) and goes
up 0.05 every time the smoothed interception rate passes 80 %, to 1 (±45°, 0.8–1.4 s: fast throws and high lobs).
"""

import math
import warnings

import torch

from mindflight.envs.base import POLICY_DT_S, SUBSTEPS, DroneEnv
from mindflight.physics.controller import attitude_action_to_setpoint
from mindflight.physics.dynamics import OMEGA, PHYSICS_DT_S, POS, QUAT, VEL
from mindflight.physics.params import GRAVITY_MPS2

# ---------- the ball: a soft foam ball ----------
BALL_MASS_KG = 0.025  # ASSUMED: a light foam ball, 7 cm across (foam toy balls are 15–40 g)
BALL_RADIUS_M = 0.035  # ASSUMED
BALL_DRAG_COEFF = 0.5  # ASSUMED: a smooth sphere is ≈ 0.47 at these speeds; foam is a little rough
AIR_DENSITY_KGPM3 = 1.2  # ASSUMED: sea level, 20 °C
# k = ½ ρ Cd A / m (1/m): drag deceleration = k · speed²
BALL_DRAG_PER_MASS = 0.5 * AIR_DENSITY_KGPM3 * BALL_DRAG_COEFF * math.pi * BALL_RADIUS_M**2 / BALL_MASS_KG
BALL_DT_S = PHYSICS_DT_S  # the ball is stepped at the physics rate: 10 steps per policy step

# ---------- the person and the guard point ----------
PERSON_RADIUS_M = 0.25
PERSON_HEIGHT_M = 1.8
CHEST = (0.0, 0.0, 1.2)
SAFETY_DISTANCE_M = 0.5  # the drone's edge may never come closer than this to the body
GUARD_POINT = (1.2, 0.0, 1.8)  # 1.8 m up: thrown balls arc in from above, 1.5–2.3 m high when they pass here
DRONE_RADIUS_M = 0.06  # the Crazyflie's collision radius (params.collision_radius_m)
TOUCH_DISTANCE_M = DRONE_RADIUS_M + BALL_RADIUS_M  # centre to centre

# ---------- throws ----------
THROW_DISTANCE_M = (6.0, 10.0)  # thrower to person, horizontally
RELEASE_HEIGHT_M = (1.3, 2.0)  # where the ball leaves the hand
TORSO_HEIGHT_M = (1.0, 1.45)  # aimed at
TORSO_SIDEWAYS_M = 0.1  # aim up to this far left/right of the middle
THROW_GAP_S = (2.5, 3.0)  # time from one throw to the next
FIRST_THROW_S = (1.0, 1.5)  # first throw of an episode
EASY_ARC_DEG = 10.0  # difficulty 0: the thrower is within ±10° of straight ahead
HARD_ARC_DEG = 45.0  # difficulty 1
EASY_FLIGHT_S = (1.1, 1.3)  # difficulty 0: ball in the air this long (s). Longer = a high lob, not a slower throw
HARD_FLIGHT_S = (0.8, 1.4)  # difficulty 1: fast throws and high lobs mixed (a brain trained on 0.9–1.1 s only
# chased lobs into the 0.5 m safety zone: see the doc)
MAX_FLIGHT_S = max(EASY_FLIGHT_S[1], HARD_FLIGHT_S[1])
PATH_POINTS = round(MAX_FLIGHT_S / POLICY_DT_S) + 1  # the stored path: one point per policy step

# ---------- tracker (the camera system that measures the ball) ----------
TRACKER_DELAY_STEPS = (1, 2)  # 20 or 40 ms late (one or two policy steps), random per episode
TRACKER_POS_BIAS_M = 0.015  # ASSUMED: error that stays the same for a whole throw (a filter's error drifts slowly)
TRACKER_POS_JITTER_M = 0.015  # ASSUMED: fresh error every step
TRACKER_VEL_BIAS_MPS = 0.10  # ASSUMED
TRACKER_VEL_JITTER_MPS = 0.05  # ASSUMED
PREDICT_STEPS = 60  # the tracker predicts the path 60 × 20 ms = 1.2 s ahead
PATH_MIN_RADIUS_M = 0.9  # path points closer to the person than this are not targets (the drone may not go there)
PATH_MIN_HEIGHT_M = 0.3
PATH_MAX_HEIGHT_M = 3.0  # the drone crashes above 3.5 m

# ---------- reward ----------
ALIVE_BONUS = 1.0
PATH_PROGRESS_WEIGHT = 10.0
PATH_PRECISION_BONUS = 1.0
GUARD_WEIGHT = 0.2  # gentle: a strong pull back to the guard point taught the drone never to leave it
GUARD_CAP_M = 5.0  # 0.2 × 5 = 1 = the alive bonus, so the reward stays ≥ 0
INTERCEPT_BONUS = 20.0
NEAR_MISS_BONUS = 10.0  # when a throw ends: 10 · max(0, 1 − closest ball-drone distance / 0.5 m)
NEAR_MISS_M = 0.5
HIT_PENALTY = 10.0
CRASH_PENALTY = 30.0
ARENA_RADIUS_M = 5.0  # further than this from the person is a crash
SHOW_EVENT_S = 0.4  # the viewer marks a catch or a hit for this long


def throw_settings(difficulty: float) -> tuple[float, float, float]:
    """Difficulty 0..1 -> (arc half-width (rad), shortest flight time (s), longest flight time (s)), straight-line."""
    d = min(max(difficulty, 0.0), 1.0)
    arc_deg = EASY_ARC_DEG + (HARD_ARC_DEG - EASY_ARC_DEG) * d
    shortest = EASY_FLIGHT_S[0] + (HARD_FLIGHT_S[0] - EASY_FLIGHT_S[0]) * d
    longest = EASY_FLIGHT_S[1] + (HARD_FLIGHT_S[1] - EASY_FLIGHT_S[1]) * d
    return math.radians(arc_deg), shortest, longest


# ---------- ball physics ----------


def ball_acceleration(
    vel: torch.Tensor, drag_per_mass: float = BALL_DRAG_PER_MASS, gravity: float = GRAVITY_MPS2
) -> torch.Tensor:
    """Acceleration of the ball (…, 3) in m/s² from its velocity (…, 3): a = (0, 0, −g) − k · |v| · v."""
    speed = vel.norm(dim=-1, keepdim=True)
    acc = -drag_per_mass * speed * vel
    acc[..., 2] -= gravity
    return acc


def ball_step(
    pos: torch.Tensor,
    vel: torch.Tensor,
    dt: float = BALL_DT_S,
    drag_per_mass: float = BALL_DRAG_PER_MASS,
    gravity: float = GRAVITY_MPS2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance balls (…, 3) by dt seconds with Heun's method (a two-stage Runge-Kutta):

        a1 = a(v)                   the slope now
        a2 = a(v + a1·dt)           the slope at the end, using a1 to guess the end velocity
        a  = ½ (a1 + a2)            their average
        v ← v + a·dt,   p ← p + v·dt + ½·a·dt²

    Without drag a is just gravity, and this is the exact parabola (no error at all, any dt).
    """
    a1 = ball_acceleration(vel, drag_per_mass, gravity)
    a2 = ball_acceleration(vel + a1 * dt, drag_per_mass, gravity)
    acc = 0.5 * (a1 + a2)
    return pos + vel * dt + 0.5 * acc * dt**2, vel + acc * dt


def fly_for(pos: torch.Tensor, vel: torch.Tensor, steps: torch.Tensor, dt: float = BALL_DT_S, sample_every: int = 0):
    """Fly balls (n, 3) for `steps` (n,) ball steps each. Returns (position after its own steps (n, 3), path).

    sample_every > 0 also records a path (n, P, 3): the position every `sample_every` steps from the release
    (P = 1 + max steps // sample_every). After its own steps a ball's path stays where it stopped.
    """
    end = pos.clone()
    longest = int(steps.max().item())
    samples = [pos.clone()] if sample_every else []
    for k in range(1, longest + 1):
        pos, vel = ball_step(pos, vel, dt)
        end = torch.where((steps == k)[:, None], pos, end)
        if sample_every and k % sample_every == 0:
            samples.append(torch.where((steps >= k)[:, None], pos, end))
    path = torch.stack(samples, dim=1) if sample_every else None
    return end, path


def aim_throws(
    release: torch.Tensor, target: torch.Tensor, flight_s: torch.Tensor, dt: float = BALL_DT_S, iterations: int = 10
) -> torch.Tensor:
    """The throw velocity (n, 3) that takes a ball from `release` (n, 3) to `target` (n, 3) in `flight_s` (n,).

    Start from the no-air answer, v = (target − release) / T + (0, 0, ½ g T), then "shoot" a few times: fly the ball
    with drag, see where it is after T seconds, and add miss / T to the throw. Each round leaves about 40 % of the
    miss, so 10 rounds get from metres to well under a millimetre. flight_s is rounded to whole ball steps.
    """
    steps = (flight_s / dt).round().long()
    flight_s = steps.float() * dt
    vel = (target - release) / flight_s[:, None]
    vel[:, 2] += 0.5 * GRAVITY_MPS2 * flight_s
    for _ in range(iterations):
        end, _ = fly_for(release, vel, steps, dt)
        vel = vel + (target - end) / flight_s[:, None]
    return vel


def distance_to_body(points: torch.Tensor) -> torch.Tensor:
    """Distance (…) from points (…, 3) to the person's cylinder (radius 0.25 m, from the floor to 1.8 m); 0 inside.

    How far outside sideways (radial distance − radius) and above it (z − height), each 0 if not outside that way,
    combined with Pythagoras: the distance to the nearest point of the cylinder.
    """
    sideways = (points[..., :2].norm(dim=-1) - PERSON_RADIUS_M).clamp_min(0.0)
    above = (points[..., 2] - PERSON_HEIGHT_M).clamp_min(0.0)
    return torch.sqrt(sideways**2 + above**2)


def predict_path(pos: torch.Tensor, vel: torch.Tensor, steps: int = PREDICT_STEPS, dt: float = POLICY_DT_S):
    """The path (n, steps + 1, 3) that balls at pos, vel (n, 3) will fly, one point every dt: `ball_step` with a
    bigger step (20 ms instead of 2 ms; over 1 s that drifts under a centimetre from the fine version)."""
    points = [pos]
    for _ in range(steps):
        pos, vel = ball_step(pos, vel, dt)
        points.append(pos)
    return torch.stack(points, dim=1)


def allowed_pieces(path: torch.Tensor) -> torch.Tensor:
    """(n, P − 1) bool: which pieces of a path (n, P, 3) the drone may fly to.

    A piece counts if both its ends are 0.3–3 m up and the ball has not yet come within 0.9 m of the person's middle
    (from there on it is too close to the person for the drone, so the rest of the path is out of bounds).
    """
    near_person = (path[..., :2].norm(dim=-1) < PATH_MIN_RADIUS_M).int().cummax(dim=1).values.bool()
    height = path[..., 2]
    ok = ~near_person & (height >= PATH_MIN_HEIGHT_M) & (height <= PATH_MAX_HEIGHT_M)
    return ok[:, :-1] & ok[:, 1:]


def nearest_on_path(points: torch.Tensor, path: torch.Tensor, valid: torch.Tensor):
    """The point of a path nearest to each of `points` (n, 3). The path is straight pieces between samples (n, P, 3);
    valid (n, P − 1) says which pieces count.

    For a piece a→b the nearest point is a + s·(b − a), s = clamp((p − a)·(b − a) / |b − a|², 0, 1).
    Returns (nearest point (n, 3), how far along the path it is, in samples = piece index + s (n,), distance (n,)).
    Rows with no valid piece get distance +inf.
    """
    a, b = path[:, :-1], path[:, 1:]
    ab = b - a
    s = (((points[:, None] - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-9)).clamp(0.0, 1.0)
    nearest = a + s[..., None] * ab
    distance = (points[:, None] - nearest).norm(dim=-1)
    distance = torch.where(valid, distance, torch.full_like(distance, math.inf))
    best, index = distance.min(dim=1)
    rows = torch.arange(points.shape[0], device=points.device)
    return nearest[rows, index], index.float() + s[rows, index], best


def tracker_prediction(
    true_path: torch.Tensor,
    seen_index: torch.Tensor,
    pos_error: torch.Tensor,
    vel_error: torch.Tensor,
    flag: torch.Tensor,
    drone: torch.Tensor,
):
    """What the tracker predicts: (nearest allowed point of its predicted path − drone (n, 3), time until the ball
    gets there (n,)). Zeros when no ball is seen or no part of the predicted path is allowed.

    A tracker predicts by flying its measured ball forward (`predict_path`). Doing that 50 times a second for 4,096
    drones is slow, and it isn't needed: the ball's flight is fixed once thrown, so the prediction is the true path
    from the moment of the measurement (sample `seen_index`, (n,)) plus the error a wrong start causes. To first
    order that error is   Δp(t) = pos_error + vel_error · t,   t = time ahead of the measurement.
    The "time until" counts from the measurement, so it is too long by the tracker's delay, as on a real system.
    """
    index = torch.arange(true_path.shape[1], device=true_path.device)
    ahead_s = ((index[None, :] - seen_index[:, None]) * POLICY_DT_S).clamp_min(0.0)  # (n, P)
    path = true_path + pos_error[:, None] + vel_error[:, None] * ahead_s[..., None]
    valid = (index[None, :-1] >= seen_index[:, None]) & allowed_pieces(path)
    nearest, along, distance = nearest_on_path(drone, path, valid)
    has = (flag > 0) & torch.isfinite(distance)
    offset = torch.where(has[:, None], nearest - drone, torch.zeros_like(drone))
    time_left = (along - seen_index.float()) * POLICY_DT_S
    return offset, torch.where(has, time_left, torch.zeros_like(time_left))


def rotate_z(vectors: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Turn vectors (n, …, 3) about the vertical axis by angle (n,) rad (counter-clockwise seen from above)."""
    c = torch.cos(angle).view(-1, *([1] * (vectors.dim() - 2)))
    s = torch.sin(angle).view(-1, *([1] * (vectors.dim() - 2)))
    x, y = vectors[..., 0], vectors[..., 1]
    return torch.stack((c * x - s * y, s * x + c * y, vectors[..., 2]), dim=-1)


def fly_ball_through_step(
    drone_start: torch.Tensor,
    drone_end: torch.Tensor,
    ball_pos: torch.Tensor,
    ball_vel: torch.Tensor,
    flying: torch.Tensor,
    event_pos: torch.Tensor,
):
    """Fly balls through one policy step (10 ball steps of 2 ms) and check what they touch.

    At ball step k the drone is placed on the straight line from where it was to where it is now,
    p = p0 + (k/10)·(p1 − p0), and three things are checked, in this order, for balls still flying: touching the
    drone (caught), touching the person, touching the floor. A ball that touches something stops flying.

    Returns (ball position, velocity, still flying, caught, hit person or floor, where it touched, closest the ball
    came to the drone's centre this step (inf if it was not flying)), all (N, …).
    """
    caught = torch.zeros_like(flying)
    hit = torch.zeros_like(flying)
    closest = torch.full_like(ball_pos[:, 0], math.inf)
    pos, vel = ball_pos, ball_vel
    for k in range(1, SUBSTEPS + 1):
        pos, vel = ball_step(pos, vel)
        drone = drone_start + (k / SUBSTEPS) * (drone_end - drone_start)
        gap = (pos - drone).norm(dim=1)
        closest = torch.where(flying, torch.minimum(closest, gap), closest)
        touch = flying & (gap < TOUCH_DISTANCE_M)
        arrived = flying & ~touch & ((distance_to_body(pos) < BALL_RADIUS_M) | (pos[:, 2] < BALL_RADIUS_M))
        event_pos = torch.where((touch | arrived)[:, None], pos, event_pos)
        caught = caught | touch
        hit = hit | arrived
        flying = flying & ~(touch | arrived)
    return pos, vel, flying, caught, hit, event_pos, closest


class InterceptEnv(DroneEnv):
    """P7: guard a person against thrown balls. Attitude actions by default. See the top of this file."""

    name = "intercept"
    task_obs_size = 18
    has_curriculum = True

    def __init__(
        self,
        n_envs: int,
        episode_s: float = 15.0,
        device: str | None = None,
        seed: int = 0,
        difficulty: float | None = None,
        start_difficulty: float = 0.0,
        max_tilt_deg: float = 30.0,
        bank_size: int = 4096,
        tracker_noise: bool = True,
        arc_deg: float | None = None,
        flight_s: float | None = None,
        **options,
    ):
        """difficulty: None = curriculum (start at start_difficulty, climb as it learns); a number = fixed.
        max_tilt_deg: the roll/pitch limit handed to the onboard controller (the stock setting is 30°).
        bank_size: how many aimed throws to keep ready (they are turned to a random direction when used).
        tracker_noise: False = the ball is measured without noise (still late; for tests).
        arc_deg, flight_s: fix the throws instead of the difficulty: arc half-width (°), flight time (s, ± 0.1 s).
        For measuring what a policy can do outside its training range (scripts/intercept_report.py)."""
        options.setdefault("action_mode", "attitude")
        super().__init__(n_envs, episode_s, device, seed, **options)
        self.fixed_difficulty = difficulty is not None
        self.difficulty = float(start_difficulty) if difficulty is None else float(difficulty)
        self.smoothed_success = 0.0
        self.max_tilt_rad = math.radians(max_tilt_deg)
        self.bank_size = bank_size
        self.tracker_noise = tracker_noise
        self.arc_deg, self.flight_s = arc_deg, flight_s

        n, dev = n_envs, self.device
        self.guard = torch.tensor(GUARD_POINT, device=dev).expand(n, 3)
        self.chest = torch.tensor(CHEST, device=dev).expand(n, 3)
        self.ball_pos = torch.zeros(n, 3, device=dev)
        self.ball_vel = torch.zeros(n, 3, device=dev)
        self.flying = torch.zeros(n, dtype=torch.bool, device=dev)  # a ball is in the air now
        self.was_flying = torch.zeros(n, dtype=torch.bool, device=dev)  # ...at the start of the last step
        self.ball_age = torch.zeros(n, dtype=torch.long, device=dev)  # policy steps since the throw
        self.path = torch.zeros(n, PATH_POINTS, 3, device=dev)  # the throw's true path, one point per policy step
        self.next_throw = torch.zeros(n, dtype=torch.long, device=dev)  # step_count of the next throw
        self.prev_pos = torch.zeros(n, 3, device=dev)
        # Tracker: the true ball (position, velocity, in-the-air flag) now, 1 step ago and 2 steps ago.
        self.history = torch.zeros(n, 3, 7, device=dev)
        self.delay = torch.ones(n, dtype=torch.long, device=dev)
        # What happened during the last step.
        self.intercepted = torch.zeros(n, dtype=torch.bool, device=dev)
        self.hit = torch.zeros(n, dtype=torch.bool, device=dev)  # reached the person or the floor
        self.crashed_now = torch.zeros(n, dtype=torch.bool, device=dev)
        self.too_close_now = torch.zeros(n, dtype=torch.bool, device=dev)
        # Per-episode counts.
        self.throws = torch.zeros(n, device=dev)  # throws that finished (caught or not)
        self.intercepts = torch.zeros(n, device=dev)
        self.hits = torch.zeros(n, device=dev)
        self.closest_to_body = torch.full((n,), math.inf, device=dev)
        self.miss_distance = torch.full((n,), math.inf, device=dev)  # closest the current ball came to the drone
        # For the viewer: where the last catch / hit happened and when.
        self.event_pos = torch.zeros(n, 3, device=dev)
        self.event_kind = torch.zeros(n, dtype=torch.long, device=dev)  # 0 none, 1 caught, 2 hit
        self.event_step = torch.zeros(n, dtype=torch.long, device=dev)

        self.pos_bias = torch.zeros(n, 3, device=dev)  # the tracker's error for the current throw
        self.vel_bias = torch.zeros(n, 3, device=dev)
        self._fast_fns: dict = {}  # torch.compile'd helpers, made on first use (see _fast)
        self.bank_difficulty: float | None = None
        self.refill_bank()

    # ---------- throws ----------

    def throw_settings(self) -> tuple[float, float, float]:
        """(arc half-width (rad), shortest, longest flight time (s)): from the difficulty, unless fixed by options."""
        arc, shortest, longest = throw_settings(self.difficulty)
        if self.arc_deg is not None:
            arc = math.radians(self.arc_deg)
        if self.flight_s is not None:
            shortest, longest = self.flight_s - 0.1, self.flight_s + 0.1
        return arc, shortest, longest

    def refill_bank(self) -> None:
        """Aim `bank_size` throws at the current difficulty, with the thrower on the +x axis."""
        n = self.bank_size
        _, shortest, longest = self.throw_settings()
        release = torch.zeros(n, 3, device=self.device)
        release[:, 0] = self.uniform(n, *THROW_DISTANCE_M)
        release[:, 2] = self.uniform(n, *RELEASE_HEIGHT_M)
        target = torch.zeros(n, 3, device=self.device)
        target[:, 1] = self.uniform(n, -TORSO_SIDEWAYS_M, TORSO_SIDEWAYS_M)
        target[:, 2] = self.uniform(n, *TORSO_HEIGHT_M)
        flight_s = self.uniform(n, shortest, longest)
        velocity = aim_throws(release, target, flight_s)
        steps = (flight_s / BALL_DT_S).round().long()
        _, path = fly_for(release, velocity, steps, sample_every=SUBSTEPS)
        padded = path[:, -1:].expand(n, PATH_POINTS, 3).clone()  # pad short paths with their last point
        padded[:, : path.shape[1]] = path[:, :PATH_POINTS]
        self.bank = {"release": release, "velocity": velocity, "path": padded, "flight_s": steps.float() * BALL_DT_S}
        self.bank_difficulty = self.difficulty

    def throw(self, ids: torch.Tensor) -> None:
        """Throw a ball at drones `ids`' person: a throw from the bank, turned to a random direction in the arc."""
        n = len(ids)
        if self.bank_difficulty != self.difficulty:
            self.refill_bank()
        arc, _, _ = self.throw_settings()
        pick = torch.randint(0, self.bank_size, (n,), generator=self.generator, device=self.device)
        angle = self.uniform(n, -arc, arc)
        self.ball_pos[ids] = rotate_z(self.bank["release"][pick], angle)
        self.ball_vel[ids] = rotate_z(self.bank["velocity"][pick], angle)
        self.path[ids] = rotate_z(self.bank["path"][pick], angle)
        self.flying.index_fill_(0, ids, True)
        self.miss_distance.index_fill_(0, ids, math.inf)
        if self.tracker_noise:
            self.pos_bias[ids] = TRACKER_POS_BIAS_M * torch.randn(n, 3, generator=self.generator, device=self.device)
            self.vel_bias[ids] = TRACKER_VEL_BIAS_MPS * torch.randn(n, 3, generator=self.generator, device=self.device)
        self.ball_age.index_fill_(0, ids, 0)
        gap = self.uniform(n, *THROW_GAP_S)
        self.next_throw[ids] = self.step_count[ids] + (gap / POLICY_DT_S).round().long()

    def _reset_task(self, ids: torch.Tensor) -> None:
        # index_fill_ sets rows to a plain number without first copying the number to the GPU (and waiting for it).
        n = len(ids)
        start = self.guard[ids] + self.uniform(n, -0.3, 0.3, size=3)
        self.state[ids, POS] = start
        self.state[ids, QUAT] = self.random_tilt(n, math.radians(10))
        self.state[ids, VEL] = self.uniform(n, -0.3, 0.3, size=3)
        self.state[ids, OMEGA] = self.uniform(n, -0.5, 0.5, size=3)
        self.prev_pos[ids] = start
        self.flying.index_fill_(0, ids, False)
        self.was_flying.index_fill_(0, ids, False)
        self.ball_pos.index_fill_(0, ids, 0.0)
        self.ball_vel.index_fill_(0, ids, 0.0)
        self.ball_age.index_fill_(0, ids, 0)
        self.history.index_fill_(0, ids, 0.0)
        self.delay[ids] = torch.randint(
            TRACKER_DELAY_STEPS[0], TRACKER_DELAY_STEPS[1] + 1, (n,), generator=self.generator, device=self.device
        )
        first = self.uniform(n, *FIRST_THROW_S)
        self.next_throw[ids] = (first / POLICY_DT_S).round().long()
        self.intercepted.index_fill_(0, ids, False)
        self.hit.index_fill_(0, ids, False)
        self.crashed_now.index_fill_(0, ids, False)
        self.too_close_now.index_fill_(0, ids, False)
        self.throws.index_fill_(0, ids, 0.0)
        self.intercepts.index_fill_(0, ids, 0.0)
        self.hits.index_fill_(0, ids, 0.0)
        self.closest_to_body.index_fill_(0, ids, math.inf)
        self.miss_distance.index_fill_(0, ids, math.inf)
        self.event_kind.index_fill_(0, ids, 0)

    # ---------- the world step: drone physics, then the ball ----------

    def command(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_mode == "attitude":
            return attitude_action_to_setpoint(action, self.params, self.max_tilt_rad)
        return super().command(action)

    def physics(self, command: torch.Tensor) -> None:
        """Move the drone (base class), then fly the ball through the same 20 ms (`fly_ball_through_step`)."""
        start = self.state[:, POS].clone()
        super().physics(command)
        self.was_flying = self.flying.clone()
        pos, vel, flying, caught, hit, self.event_pos, closest = self._fast(
            fly_ball_through_step, start, self.state[:, POS], self.ball_pos, self.ball_vel, self.flying, self.event_pos
        )
        self.miss_distance = torch.minimum(self.miss_distance, closest)
        lost = flying & (self.ball_age >= PATH_POINTS + 25)  # should never happen: every throw is aimed at the body
        self.ball_pos, self.ball_vel = pos, vel
        self.flying = flying & ~lost
        self.intercepted, self.hit = caught, hit | lost
        self.ball_age = self.ball_age + self.was_flying.long()
        self.event_kind = torch.where(caught, 1, torch.where(self.hit, 2, self.event_kind))
        self.event_step = torch.where(caught | self.hit, self.step_count + 1, self.event_step)

    def _fast(self, fn, *tensors):
        """fn(*tensors), fused by torch.compile on a GPU. The ball's 10 steps and the tracker's 60-step prediction are
        hundreds of tiny GPU jobs each; the compiler merges them into a few. Same maths either way."""
        if fn not in self._fast_fns:
            self._fast_fns[fn] = fn
            if self.fast_physics and self.device.type == "cuda":
                try:
                    compiled = torch.compile(fn, dynamic=False)
                    result = compiled(*tensors)
                    self._fast_fns[fn] = compiled
                    return result
                except Exception as error:  # no triton or no working compiler: use the plain version
                    warnings.warn(f"torch.compile failed, running {fn.__name__} without it: {error}")
        return self._fast_fns[fn](*tensors)

    # ---------- observation ----------

    def measured_ball(self) -> tuple[torch.Tensor, ...]:
        """The tracker's reading, late and a little wrong: the true ball 1–2 steps ago + this throw's fixed error +
        fresh jitter. Returns (position (N, 3), velocity (N, 3), in-the-air flag (N,), position error, velocity
        error)."""
        rows = torch.arange(self.n_envs, device=self.device)
        reading = self.history[rows, self.delay]
        pos, vel, flag = reading[:, 0:3], reading[:, 3:6], reading[:, 6]
        pos_error, vel_error = torch.zeros_like(pos), torch.zeros_like(vel)
        if self.tracker_noise:
            jitter = torch.randn(self.n_envs, 6, generator=self.generator, device=self.device)
            pos_error = self.pos_bias + TRACKER_POS_JITTER_M * jitter[:, :3]
            vel_error = self.vel_bias + TRACKER_VEL_JITTER_MPS * jitter[:, 3:]
        return pos + pos_error, vel + vel_error, flag, pos_error, vel_error

    def _record_ball(self) -> None:
        """Push the true ball state into the tracker's history (index 0 = now)."""
        now = torch.cat((self.ball_pos, self.ball_vel, self.flying.float()[:, None]), dim=1)
        self.history = torch.cat((now[:, None], self.history[:, :-1]), dim=1)

    def _task_obs(self) -> torch.Tensor:
        drone = self.state[:, POS]
        ball, vel, flag, pos_error, vel_error = self.measured_ball()
        seen_index = (self.ball_age - self.delay).clamp_min(0)  # the path sample the reading belongs to
        go_to, time_left = tracker_prediction(self.path, seen_index, pos_error, vel_error, flag, drone)
        on = flag[:, None]
        return torch.cat(
            (
                (ball - drone) * on,
                vel * on,
                flag[:, None],
                go_to,
                time_left[:, None],
                self.guard - drone,
                self.chest - drone,
                distance_to_body(drone)[:, None],
            ),
            dim=1,
        )

    # ---------- reward ----------

    def path_distances(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Distance from the drone (now, and where it was a step ago) to the rest of the ball's true path, (N,) each.

        "The rest" = pieces the ball has not passed yet, and only where the drone may go (≥ 0.9 m from the person's
        middle, 0.3–3 m up). Both distances use the same pieces, so their difference is pure drone movement.
        """
        index = torch.arange(PATH_POINTS - 1, device=self.device)
        not_passed = index[None, :] >= self.ball_age[:, None]  # piece j runs from sample j to j + 1
        valid = not_passed & allowed_pieces(self.path)
        now = nearest_on_path(self.state[:, POS], self.path, valid)[2]
        before = nearest_on_path(self.prev_pos, self.path, valid)[2]
        return now, before

    def _reward(self) -> torch.Tensor:
        pos = self.state[:, POS]
        now, before = self.path_distances()
        has_path = torch.isfinite(now) & self.was_flying
        progress = torch.where(has_path, before - now, torch.zeros_like(now))
        precision = torch.where(has_path, (1.0 - now / 0.3).clamp_min(0.0), torch.zeros_like(now))
        flight_terms = PATH_PROGRESS_WEIGHT * progress + PATH_PRECISION_BONUS * precision
        guard_distance = (self.guard - pos).norm(dim=1)
        idle_terms = -GUARD_WEIGHT * guard_distance.clamp(max=GUARD_CAP_M)
        spin = (self.state[:, OMEGA] ** 2).sum(dim=1)
        twitch = ((self.action - self.prev_action) ** 2).sum(dim=1)
        ended = (self.intercepted | self.hit).float()  # the throw is over: score how close the drone came
        reward = (
            ALIVE_BONUS
            + torch.where(self.was_flying, flight_terms, idle_terms)
            + INTERCEPT_BONUS * self.intercepted.float()
            + NEAR_MISS_BONUS * ((1.0 - self.miss_distance / NEAR_MISS_M).clamp_min(0.0) * ended).nan_to_num(0.0)
            - HIT_PENALTY * self.hit.float()
            - 0.01 * spin
            - 0.05 * twitch
        )
        self.throws += (self.intercepted | self.hit).float()
        self.intercepts += self.intercepted.float()
        self.hits += self.hit.float()
        body = distance_to_body(pos)
        self.closest_to_body = torch.minimum(self.closest_to_body, body)
        upright = 1 - 2 * (self.state[:, 4] ** 2 + self.state[:, 5] ** 2)  # R_zz from the quaternion's x and y
        self.too_close_now = body - DRONE_RADIUS_M < SAFETY_DISTANCE_M
        self.crashed_now = (
            self.too_close_now
            | (pos[:, 2] < 0.05)
            | (pos[:, 2] > 3.5)
            | (pos[:, :2].norm(dim=1) > ARENA_RADIUS_M)
            | (upright < 0.0)
            | (self.state[:, OMEGA].norm(dim=1) > 40.0)
        )
        return torch.where(self.crashed_now, reward - CRASH_PENALTY, reward)

    def _crashed(self) -> torch.Tensor:
        return self.crashed_now  # worked out in _reward, which the base class calls just before this

    def _post_step(self) -> None:
        """Remember where the drone is, throw the next ball when it is due, and update the tracker."""
        self.prev_pos = self.state[:, POS].clone()
        due = (self.step_count >= self.next_throw) & ~self.flying
        ids = due.nonzero(as_tuple=True)[0]
        if len(ids) > 0:
            self.throw(ids)
        self._record_ball()

    def _episode_info(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per episode: throws that finished, interceptions, hits, their ratio ("success"), the closest the drone got
        to the body. A crash with a ball in the air counts that ball as not intercepted."""
        lost_in_air = (self.crashed_now[ids] & self.flying[ids]).float()
        throws = self.throws[ids] + lost_in_air
        return {
            "success": self.intercepts[ids] / throws.clamp_min(1.0),
            "throws": throws,
            "intercepts": self.intercepts[ids],
            "hits": self.hits[ids],
            "closest_to_body_m": self.closest_to_body[ids],
            "crashed_too_close": self.too_close_now[ids].float(),  # which crashes were the safety rule
        }

    def update_curriculum(self, episode_means: dict[str, float]) -> dict[str, float]:
        """Smooth the interception rate; when it is above 80 %, make the throws 0.05 harder (up to 1)."""
        if "episode_success" in episode_means:
            self.smoothed_success = 0.8 * self.smoothed_success + 0.2 * episode_means["episode_success"]
        if not self.fixed_difficulty and self.smoothed_success > 0.8 and self.difficulty < 1.0:
            self.difficulty = min(1.0, round(self.difficulty + 0.05, 2))
            self.smoothed_success = 0.0
        return {"difficulty": self.difficulty}

    # ---------- viewer ----------

    def display_objects(self) -> dict[str, torch.Tensor]:
        """Person (cylinder + head), the ball, a marker where the last catch / hit happened, the guard point, the
        safety ring on the floor (0.81 m from the person's middle) and the rest of the ball's path."""
        n, dev = self.n_envs, self.device
        cylinders = torch.tensor([[0.0, 0.0, 0.0, 1.55, PERSON_RADIUS_M]], device=dev).expand(n, 1, 5)
        recent = (self.step_count - self.event_step) * POLICY_DT_S < SHOW_EVENT_S
        spheres = torch.zeros(n, 5, 4, device=dev)
        spheres[:, 0] = torch.tensor([0.0, 0.0, 1.68, 0.13], device=dev)  # head
        spheres[:, 1, :3] = self.ball_pos
        spheres[:, 1, 3] = BALL_RADIUS_M * self.flying.float()
        spheres[:, 2, :3] = self.event_pos
        spheres[:, 2, 3] = 0.08 * (recent & (self.event_kind == 1)).float()  # caught: green
        spheres[:, 3, :3] = self.event_pos
        spheres[:, 3, 3] = 0.08 * (recent & (self.event_kind == 2)).float()  # hit: red
        spheres[:, 4] = torch.tensor([*GUARD_POINT, 0.025], device=dev)
        ring_segments = 24
        angles = torch.linspace(0, 2 * math.pi, ring_segments + 1, device=dev)
        ring_radius = PERSON_RADIUS_M + SAFETY_DISTANCE_M + DRONE_RADIUS_M
        ring = torch.stack((ring_radius * torch.cos(angles), ring_radius * torch.sin(angles),
                            torch.full_like(angles, 0.01)), dim=1)
        ring_lines = torch.cat((ring[:-1], ring[1:]), dim=1).expand(n, ring_segments, 6)
        # The ball's path from where it is now, every second stored point (drawn only while it flies).
        points = self.path[:, ::2]
        index = torch.arange(points.shape[1] - 1, device=dev) * 2
        ahead = (index[None, :] >= self.ball_age[:, None]) & self.flying[:, None]
        path_lines = torch.cat((points[:, :-1], points[:, 1:]), dim=2) * ahead[..., None]
        return {
            "cylinders": cylinders,
            "cylinders_colors": ["#4f7cac"],
            "spheres": spheres,
            "spheres_colors": ["#e0b48a", "#ff9f43", "#3ddc97", "#ff5c5c", "#9aa5b1"],
            "lines": torch.cat((ring_lines, path_lines), dim=1),
            "lines_colors": ["#ff8a8a"] * ring_segments + ["#ffcc80"] * path_lines.shape[1],
        }
