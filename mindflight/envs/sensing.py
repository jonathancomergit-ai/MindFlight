"""What the drone *thinks* its state is: noisy, slightly late estimates instead of the perfect truth.

In the simulator we know exactly where each drone is. A real Crazyflie doesn't. Its state estimator fuses a gyro, an
accelerometer and a position sensor, and the result is a little noisy, a little biased, and a little late by the time
it reaches our brain on the PC over the radio. A brain trained on perfect numbers can fall apart on real ones, so we
train on realistic ones.

A **sensor profile** describes how wrong the estimate is:

| Profile | Position | Velocity | Attitude | Gyro | Delay | Like |
|---|---|---|---|---|---|---|
| `ideal` | exact | exact | exact | exact | none | Phase 1 |
| `lighthouse` | ±5 mm | ±2 cm/s | ±0.5° | ±0.02 rad/s, small bias | 0–20 ms | Lighthouse deck + 2 base stations |
| `flowdeck` | ±2 cm, slow drift | ±5 cm/s | ±1° | ±0.03 rad/s, bias | 20–40 ms | Flow deck v2 (optical flow + height laser) |

ASSUMED: these numbers are typical values from Bitcraze's positioning documentation and Crazyflie papers, rounded,
not measurements of one drone. Once the real drone flies, its logs can replace them.

How each part is made, per drone:
- **noise**: position, velocity and attitude errors *wander smoothly* instead of jumping every step, because the
  estimate is the output of a filter (the Crazyflie's Kalman filter), not a raw sensor. Each error follows
      error ← a · error + σ · √(1 − a²) · noise,      a = exp(−dt / τ)
  which keeps its spread at σ but changes over about τ seconds (`noise_correlation_s`). Gyro noise stays a fresh
  draw every step: the gyro is a raw sensor.
  (The first version drew fresh noise for everything every 20 ms. Hover under the flow-deck profile stalled at
  ~80 % because the brain chased jitter a real estimator wouldn't produce.)
- **bias**: a constant offset per episode (gyro), drawn once at reset.
- **drift** (flowdeck position): a random walk, position_error ← position_error + σ_drift · √dt · noise. Optical
  flow integrates velocity, so its position error grows slowly, like a person walking with eyes closed.
- **delay**: each drone gets a delay of 0, 1 or 2 policy steps (20 ms each) per episode. The brain then sees the
  observation from that many steps ago. Actions can be delayed the same way (the radio goes both ways).
"""

import math
from dataclasses import dataclass

import torch

from mindflight.physics.dynamics import OMEGA, POS, QUAT, VEL
from mindflight.physics.rotations import quat_from_axis_angle, quat_multiply, quat_normalize


@dataclass(frozen=True)
class SensorProfile:
    name: str
    position_noise_m: float = 0.0
    position_drift_m_per_sqrt_s: float = 0.0
    velocity_noise_mps: float = 0.0
    attitude_noise_rad: float = 0.0
    gyro_noise_radps: float = 0.0
    gyro_bias_radps: float = 0.0  # each episode's bias is uniform in ± this
    noise_correlation_s: float = 0.0  # τ of the wandering errors (0 = a fresh draw every step)
    obs_delay_steps: tuple[int, int] = (0, 0)  # (min, max) policy steps, inclusive
    action_delay_steps: tuple[int, int] = (0, 0)


PROFILES = {
    "ideal": SensorProfile("ideal"),
    "lighthouse": SensorProfile(
        "lighthouse",
        position_noise_m=0.005,
        velocity_noise_mps=0.02,
        attitude_noise_rad=math.radians(0.5),
        gyro_noise_radps=0.02,
        gyro_bias_radps=0.02,
        noise_correlation_s=0.1,
        obs_delay_steps=(0, 1),
        action_delay_steps=(0, 1),
    ),
    "flowdeck": SensorProfile(
        "flowdeck",
        position_noise_m=0.02,
        position_drift_m_per_sqrt_s=0.01,
        velocity_noise_mps=0.05,
        attitude_noise_rad=math.radians(1.0),
        gyro_noise_radps=0.03,
        gyro_bias_radps=0.03,
        noise_correlation_s=0.3,
        obs_delay_steps=(1, 2),
        action_delay_steps=(0, 1),
    ),
}


class SensorModel:
    """Makes noisy estimates of the state, and delays observations and actions, for N drones."""

    def __init__(self, profile: str | SensorProfile, n: int, device, generator: torch.Generator, step_dt_s: float):
        self.profile = PROFILES[profile] if isinstance(profile, str) else profile
        self.n = n
        self.device = device
        self.generator = generator
        self.step_dt_s = step_dt_s
        self.max_obs_delay = self.profile.obs_delay_steps[1]
        self.max_action_delay = self.profile.action_delay_steps[1]
        self.gyro_bias = torch.zeros(n, 3, device=device)
        self.position_drift = torch.zeros(n, 3, device=device)
        # The wandering errors, in units of their own σ (each starts at its long-run spread).
        self.position_error = torch.zeros(n, 3, device=device)
        self.velocity_error = torch.zeros(n, 3, device=device)
        self.attitude_error = torch.zeros(n, 3, device=device)  # an axis-angle vector (rad / σ)
        self.obs_delay = torch.zeros(n, dtype=torch.long, device=device)
        self.action_delay = torch.zeros(n, dtype=torch.long, device=device)
        self.obs_history: torch.Tensor | None = None  # (N, max_obs_delay + 1, obs_size), newest first
        self.action_history = torch.zeros(n, self.max_action_delay + 1, 4, device=device)

    @property
    def is_ideal(self) -> bool:
        p = self.profile
        return (
            p.position_noise_m == p.position_drift_m_per_sqrt_s == p.velocity_noise_mps == 0
            and p.attitude_noise_rad == p.gyro_noise_radps == p.gyro_bias_radps == 0
            and self.max_obs_delay == self.max_action_delay == 0
        )

    def _randn(self, *shape) -> torch.Tensor:
        return torch.randn(*shape, generator=self.generator, device=self.device)

    def _randint(self, low: int, high: int, n: int) -> torch.Tensor:
        return torch.randint(low, high + 1, (n,), generator=self.generator, device=self.device)

    def reset(self, ids: torch.Tensor) -> None:
        """New episode for drones `ids`: new gyro bias and delays, drift back to zero, action history to hover (0)."""
        n = len(ids)
        p = self.profile
        self.gyro_bias[ids] = p.gyro_bias_radps * (2 * torch.rand(n, 3, generator=self.generator, device=self.device) - 1)
        self.position_drift[ids] = 0.0
        for error in (self.position_error, self.velocity_error, self.attitude_error):
            error[ids] = self._randn(n, 3)  # start already spread out, as if it had been flying a while
        self.obs_delay[ids] = self._randint(*p.obs_delay_steps, n)
        self.action_delay[ids] = self._randint(*p.action_delay_steps, n)
        self.action_history[ids] = 0.0

    def advance(self) -> None:
        """One policy step of time: the flow-deck drift takes a random-walk step, and the wandering errors move."""
        sigma = self.profile.position_drift_m_per_sqrt_s
        if sigma > 0:
            self.position_drift = self.position_drift + sigma * math.sqrt(self.step_dt_s) * self._randn(self.n, 3)
        tau = self.profile.noise_correlation_s
        a = math.exp(-self.step_dt_s / tau) if tau > 0 else 0.0
        kick = math.sqrt(1.0 - a * a)
        self.position_error = a * self.position_error + kick * self._randn(self.n, 3)
        self.velocity_error = a * self.velocity_error + kick * self._randn(self.n, 3)
        self.attitude_error = a * self.attitude_error + kick * self._randn(self.n, 3)

    def estimate(self, state: torch.Tensor) -> torch.Tensor:
        """A noisy copy of the true state (N, 17): what the drone's state estimator would report."""
        if self.is_ideal:
            return state
        p = self.profile
        est = state.clone()
        est[:, POS] = state[:, POS] + self.position_drift + p.position_noise_m * self.position_error
        est[:, VEL] = state[:, VEL] + p.velocity_noise_mps * self.velocity_error
        est[:, OMEGA] = state[:, OMEGA] + self.gyro_bias + p.gyro_noise_radps * self._randn(self.n, 3)
        if p.attitude_noise_rad > 0:
            # A small rotation: the attitude error vector, as an axis (its direction) and an angle (its length).
            rotation_vector = p.attitude_noise_rad * self.attitude_error
            angle = rotation_vector.norm(dim=1)
            wobble = quat_from_axis_angle(rotation_vector, angle)
            est[:, QUAT] = quat_normalize(quat_multiply(state[:, QUAT], wobble))
        return est

    def delay_observation(self, fresh: torch.Tensor, reset_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Push this step's observation (N, D) into the history; return what each drone actually sees: the
        observation from `obs_delay` steps ago. Drones in `reset_ids` (just reset) get a history full of their
        first observation, so they never see the previous episode."""
        if self.max_obs_delay == 0:
            return fresh
        if self.obs_history is None or self.obs_history.shape[-1] != fresh.shape[-1]:
            self.obs_history = fresh[:, None, :].repeat(1, self.max_obs_delay + 1, 1)
        else:
            self.obs_history = torch.cat((fresh[:, None, :], self.obs_history[:, :-1]), dim=1)
        if reset_ids is not None and len(reset_ids) > 0:
            self.obs_history[reset_ids] = fresh[reset_ids, None, :]
        rows = torch.arange(self.n, device=self.device)
        return self.obs_history[rows, self.obs_delay]

    def delay_action(self, action: torch.Tensor) -> torch.Tensor:
        """Push the chosen action (N, 4); return the one that reaches each drone now (`action_delay` steps old)."""
        if self.max_action_delay == 0:
            return action
        self.action_history = torch.cat((action[:, None, :], self.action_history[:, :-1]), dim=1)
        rows = torch.arange(self.n, device=self.device)
        return self.action_history[rows, self.action_delay]

    def restart_history(self, ids: torch.Tensor, fresh: torch.Tensor) -> None:
        """Drones `ids` just reset: fill their observation history with their first observation of the new episode."""
        if self.obs_history is not None and self.max_obs_delay > 0:
            self.obs_history[ids] = fresh[ids, None, :]
