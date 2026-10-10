"""How the drone knows where the person is (P6, follow-me). One of four sensor profiles, chosen per env.

The drone never sees the true person. It sees the latest *measurement*, which is late, noisy, comes only now and
then, and (for the camera) only when the person is in view:

| Profile | What it is | Rate | Delay | Position noise σ | Other |
|---|---|---|---|---|---|
| `camera` (default) | a camera on the drone's nose + a pose detector | 30 Hz | 60–100 ms | 2 % of the range + 3 cm | sees only within ±30–40° of the nose (60–80° view, random per flight), up to 10 m |
| `gps_rtk` | an RTK GPS on the person (and on the drone) | 10 Hz | 100 ms | 2–3 cm | – |
| `gps` | the person's phone GPS | 1–5 Hz | 100–200 ms | 2–3 m, wandering slowly (+ 0.3 m jitter) | – |
| `perfect` | the truth, now | 50 Hz | 0 | 0 | for tests and comparisons |

Ranges like "60–100 ms" are drawn once per flight. Each measurement also carries a velocity (noise σ 0.2 / 0.05 /
0.3 m/s) and a heading (σ 0.05 / 0.05 / 0.2 rad), as a tracker or the phone would give.

**The phone's wandering error.** Consumer GPS is not just jittery: its error drifts slowly (satellites move,
reflections change), so it can sit 2 m off for many seconds. We model it as an Ornstein–Uhlenbeck random walk:

    e ← e − e · dt / τ + σ · √(2 · dt / τ) · noise          τ = 20 s, σ = 2–3 m per flight

**The tracker.** A real system never uses raw detections: it runs a tracker. Ours is the simplest kind (an
exponential / alpha filter): when a measurement z comes, the old estimate x̂ is first moved forward with its
velocity to the new time, then x̂ ← x̂ + α · (z − x̂). Same for velocity and heading. The camera uses α = 0.3 (its
detections jitter); the GPS profiles use the raw values (RTK is already precise; phone-GPS error is slow, so
smoothing can't remove it).

**Camera view.** The camera looks along the drone's nose (its body x axis, seen from above). The person is in view
when the angle between the nose and the direction to the person is at most half the field of view and they are
within 10 m. Out of view, the drone keeps the last measurement and its `age_s` keeps growing, so the brain can learn
to turn toward the person to keep them in sight.

Simplification: the camera's measurement is turned into a world position with the drone's *true* position (as if
the drone knew exactly where it was). The drone's own position error is a separate model (envs/sensing.py).
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PersonSensor:
    """One sensor profile. Pairs (low, high) are drawn uniformly once per flight."""

    name: str
    rate_hz: tuple[float, float]
    delay_steps: tuple[int, int]  # in policy steps of 20 ms
    noise_m: tuple[float, float]  # white position noise σ per measurement (m)
    noise_per_m: float = 0.0  # plus this × range to the person (camera: 2 %)
    wander_m: tuple[float, float] = (0.0, 0.0)  # slowly wandering error σ (m), phone GPS
    wander_tau_s: float = 20.0
    velocity_noise_mps: float = 0.0
    heading_noise_rad: float = 0.0
    fov_deg: tuple[float, float] | None = None  # camera: horizontal field of view; None = always "in view"
    smoothing: float = 1.0  # tracker: share of each new measurement blended in (1 = use it raw)
    max_range_m: float = math.inf


PERSON_SENSORS = {
    "perfect": PersonSensor("perfect", (50.0, 50.0), (0, 0), (0.0, 0.0)),
    "camera": PersonSensor(
        "camera", (30.0, 30.0), (3, 5), (0.03, 0.03), noise_per_m=0.02, velocity_noise_mps=0.2,
        heading_noise_rad=0.05, fov_deg=(60.0, 80.0), max_range_m=10.0, smoothing=0.3,
    ),
    "gps_rtk": PersonSensor("gps_rtk", (10.0, 10.0), (5, 5), (0.02, 0.03), velocity_noise_mps=0.05,
                            heading_noise_rad=0.05),
    "gps": PersonSensor("gps", (1.0, 5.0), (5, 10), (0.3, 0.3), wander_m=(2.0, 3.0), velocity_noise_mps=0.3,
                        heading_noise_rad=0.2),
}

RECORD_SIZE = 6  # one row of the person's true state: x, y, heading, vx, vy, turn rate


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Angle (rad) wrapped into [−π, π)."""
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


def in_view(drone_xy: torch.Tensor, drone_yaw: torch.Tensor, person_xy: torch.Tensor, half_fov_rad: torch.Tensor,
            max_range_m: float) -> torch.Tensor:
    """(N,) bool: the person is inside the camera's view. |angle(person − drone) − yaw| ≤ half the field of view,
    and the person is within max_range_m (seen from above)."""
    rel = person_xy - drone_xy
    bearing = torch.atan2(rel[:, 1], rel[:, 0])
    return (wrap_angle(bearing - drone_yaw).abs() <= half_fov_rad) & (rel.norm(dim=1) <= max_range_m)


class PersonTracker:
    """The drone's running estimate of its person, for N drones at once.

    After `reset`/`step`, these hold what the brain may use:
        seen_xy (N, 2), seen_vel (N, 2), seen_heading (N,), seen_turn_rate (N,): the latest measurement
        age_s (N,): how old that measurement is (s): its delay, plus time since it came
        visible (N,) bool: the person is in the camera's view right now (always True without a camera)
    """

    def __init__(self, profile: str, n: int, device, generator: torch.Generator, dt_s: float):
        if profile not in PERSON_SENSORS:
            raise ValueError(f"person_sensor must be one of {sorted(PERSON_SENSORS)}, not {profile!r}")
        self.profile = PERSON_SENSORS[profile]
        self.n, self.device, self.generator, self.dt_s = n, device, generator, dt_s
        p = self.profile
        self.history = torch.zeros(n, p.delay_steps[1] + 1, RECORD_SIZE, device=device)  # [:, k] = k steps ago
        self.delay_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.rate_hz = torch.zeros(n, device=device)
        self.phase = torch.zeros(n, device=device)  # a measurement comes each time this passes 1
        self.noise_m = torch.zeros(n, device=device)
        self.wander_sigma_m = torch.zeros(n, device=device)
        self.wander = torch.zeros(n, 2, device=device)
        self.half_fov_rad = torch.full((n,), math.pi, device=device)
        self.seen_xy = torch.zeros(n, 2, device=device)
        self.seen_vel = torch.zeros(n, 2, device=device)
        self.seen_heading = torch.zeros(n, device=device)
        self.seen_turn_rate = torch.zeros(n, device=device)
        self.age_s = torch.zeros(n, device=device)
        self.visible = torch.ones(n, dtype=torch.bool, device=device)

    def uniform(self, n: int, low: float, high: float, size: int | None = None) -> torch.Tensor:
        shape = (n,) if size is None else (n, size)
        return low + (high - low) * torch.rand(shape, generator=self.generator, device=self.device)

    def reset(self, ids: torch.Tensor, record: torch.Tensor, drone_pos: torch.Tensor) -> None:
        """New flights for drones `ids`: fresh per-flight numbers, a history as if the person had always stood
        there, and a first measurement straight away (the flight starts with the person in sight)."""
        p, n = self.profile, len(ids)
        self.history[ids] = record[:, None, :]
        self.delay_steps[ids] = torch.randint(
            p.delay_steps[0], p.delay_steps[1] + 1, (n,), generator=self.generator, device=self.device
        )
        self.rate_hz[ids] = self.uniform(n, *p.rate_hz)
        self.phase[ids] = self.uniform(n, 0.0, 1.0)
        self.noise_m[ids] = self.uniform(n, *p.noise_m)
        self.wander_sigma_m[ids] = self.uniform(n, *p.wander_m)
        self.wander[ids] = self.wander_sigma_m[ids, None] * torch.randn(n, 2, generator=self.generator,
                                                                         device=self.device)
        if p.fov_deg is not None:
            self.half_fov_rad[ids] = torch.deg2rad(self.uniform(n, *p.fov_deg)) / 2
        self.visible[ids] = True
        self._measure(ids, record, drone_pos)

    def _measure(self, ids: torch.Tensor, record: torch.Tensor, drone_pos: torch.Tensor) -> None:
        """Take a measurement of `record` (n, 6) for drones `ids` (no delay applied here) and store it."""
        p, n = self.profile, len(ids)
        noise = torch.randn(n, 5, generator=self.generator, device=self.device)
        rng = (record[:, 0:2] - drone_pos[:, 0:2]).norm(dim=1)
        sigma = self.noise_m[ids] + p.noise_per_m * rng
        self.seen_xy[ids] = record[:, 0:2] + self.wander[ids] + sigma[:, None] * noise[:, 0:2]
        self.seen_vel[ids] = record[:, 3:5] + p.velocity_noise_mps * noise[:, 2:4]
        self.seen_heading[ids] = wrap_angle(record[:, 2] + p.heading_noise_rad * noise[:, 4])
        self.seen_turn_rate[ids] = record[:, 5]
        self.age_s[ids] = self.delay_steps[ids].float() * self.dt_s

    def step(self, record: torch.Tensor, drone_pos: torch.Tensor, drone_yaw: torch.Tensor) -> None:
        """One policy step for everyone: remember the true state `record` (N, 6), let the GPS error wander, and
        take a measurement where one is due and the person is in view; elsewhere keep the old one (it ages).

        Measured = the true state `delay` steps ago + noise. The noise σ (m) is noise_m + noise_per_m × range."""
        p, n, dt = self.profile, self.n, self.dt_s
        self.history = torch.cat((record[:, None, :], self.history[:, :-1]), dim=1)
        if p.wander_m[1] > 0:
            kick = torch.randn(n, 2, generator=self.generator, device=self.device)
            self.wander = (self.wander - self.wander * dt / p.wander_tau_s
                           + self.wander_sigma_m[:, None] * math.sqrt(2 * dt / p.wander_tau_s) * kick)
        self.phase = self.phase + self.rate_hz * dt
        due = self.phase >= 1.0 - 1e-5  # (the small margin keeps 5 × 0.2 counting as 1 in float32)
        self.phase = self.phase - due.float()
        if p.fov_deg is not None:
            self.visible = in_view(drone_pos[:, :2], drone_yaw, record[:, 0:2], self.half_fov_rad, p.max_range_m)
        delayed = self.history.gather(1, self.delay_steps.view(-1, 1, 1).expand(n, 1, RECORD_SIZE))[:, 0]
        update = due & self.visible
        old = (self.seen_xy.clone(), self.seen_vel.clone(), self.seen_heading.clone(), self.seen_turn_rate.clone(),
               self.age_s + dt)
        everyone = torch.arange(n, device=self.device)
        self._measure(everyone, delayed, drone_pos)
        if p.smoothing < 1.0:  # the tracker: blend the new measurement into the old estimate, moved forward
            a, since = p.smoothing, (old[4] - self.age_s)[:, None]  # time between the two measurements (s)
            guess_xy = old[0] + old[1] * since
            self.seen_xy = guess_xy + a * (self.seen_xy - guess_xy)
            self.seen_vel = old[1] + a * (self.seen_vel - old[1])
            guess_heading = old[2] + old[3] * since[:, 0]
            self.seen_heading = wrap_angle(guess_heading + a * wrap_angle(self.seen_heading - guess_heading))
        self.seen_xy = torch.where(update[:, None], self.seen_xy, old[0])
        self.seen_vel = torch.where(update[:, None], self.seen_vel, old[1])
        self.seen_heading = torch.where(update, self.seen_heading, old[2])
        self.seen_turn_rate = torch.where(update, self.seen_turn_rate, old[3])
        self.age_s = torch.where(update, self.age_s, old[4])
