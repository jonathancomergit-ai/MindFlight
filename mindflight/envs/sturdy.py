"""Level 5 task: fly to a point (Level 2) in gusty wind, on a drone that is never quite the same twice.

Everything from `waypoint.py` stays: targets, reward, success rule, observations. The drone is **not** told about the
wind or its own weight. It has to notice from how it moves, and correct. What changes is the world:

| Each episode, at random | Range |
|---|---|
| mass (and inertia with it) | ±20 % |
| prop strength (thrust and drag torque) | ±10 % |
| motor lag (time constant) | 0.7 × to 1.5 × |
| steady wind | 0 to 2 m/s, any horizontal direction |

On top of the steady wind come **gusts**, changing every step. Each gust component follows a random walk that is
pulled back toward zero (an Ornstein–Uhlenbeck process, a standard way to model turbulence):

    gust ← gust − gust · dt / τ + σ · √(2 · dt / τ) · noise        τ = 0.5 s, σ = 1.5 m/s horizontal, 0.5 m/s up/down

So the wind is "steady + gusts", typically 1–4 m/s and jumping around. Rotor drag (how wind pushes the drone) and
ground effect are on. Both come from `physics/aero.py`.

**Domain randomization** is the lesson: a policy trained on one perfect drone in still air learns that drone. A
policy trained on thousands of slightly different drones in all sorts of wind learns to *fly*, and copes with the
real one. `randomize=False` gives the same task with a standard drone in still air (drag and ground effect still on),
to compare against.
"""

import math

import torch

from mindflight.envs.base import POLICY_DT_S
from mindflight.envs.waypoint import WaypointEnv
from mindflight.physics.aero import DroneVariation

GUST_TAU_S = 0.5
GUST_SIGMA_MPS = (1.5, 1.5, 0.5)
STEADY_WIND_MAX_MPS = 2.0


class SturdyEnv(WaypointEnv):
    """Level 5: waypoint flying with random drones, wind and gusts. Same observation as waypoint (22 numbers)."""

    name = "sturdy"

    def __init__(
        self,
        n_envs: int,
        episode_s: float = 10.0,
        device: str | None = None,
        seed: int = 0,
        randomize: bool = True,
        gust_scale: float = 1.0,
        **options,
    ):
        """randomize: False = standard drone in still air. gust_scale: multiply the gusts and steady wind (2 = a stress
        test with twice the wind it trained on)."""
        super().__init__(n_envs, episode_s, device, seed, **options)
        self.randomize = randomize
        self.gust_scale = gust_scale
        ones = torch.ones(n_envs, device=self.device)
        self.steady_wind = torch.zeros(n_envs, 3, device=self.device)
        self.gust = torch.zeros(n_envs, 3, device=self.device)
        self.gust_sigma = gust_scale * torch.tensor(GUST_SIGMA_MPS, device=self.device)
        self.variation = DroneVariation(
            mass_scale=ones.clone(),
            thrust_scale=ones.clone(),
            motor_tau_scale=ones.clone(),
            wind_mps=torch.zeros(n_envs, 3, device=self.device),
            drag=True,
            ground_effect=True,
        )

    def _reset_task(self, ids: torch.Tensor) -> None:
        super()._reset_task(ids)
        if not self.randomize:
            return
        n = len(ids)
        self.variation.mass_scale[ids] = self.uniform(n, 0.8, 1.2)
        self.variation.thrust_scale[ids] = self.uniform(n, 0.9, 1.1)
        self.variation.motor_tau_scale[ids] = self.uniform(n, 0.7, 1.5)
        direction = self.uniform(n, -math.pi, math.pi)
        speed = self.uniform(n, 0.0, STEADY_WIND_MAX_MPS * self.gust_scale)
        self.steady_wind[ids, 0] = speed * torch.cos(direction)
        self.steady_wind[ids, 1] = speed * torch.sin(direction)
        self.gust[ids] = 0.0
        self.variation.wind_mps[ids] = self.steady_wind[ids]

    def physics(self, rpm_cmd: torch.Tensor) -> None:
        """Move the gusts on one policy step (held for its 10 physics steps), then fly."""
        if self.randomize:
            noise = torch.randn(self.n_envs, 3, generator=self.generator, device=self.device)
            self.gust = (
                self.gust
                - self.gust * POLICY_DT_S / GUST_TAU_S
                + self.gust_sigma * math.sqrt(2 * POLICY_DT_S / GUST_TAU_S) * noise
            )
            self.variation.wind_mps.copy_(self.steady_wind + self.gust)
        super().physics(rpm_cmd)

    def _episode_info(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        info = super()._episode_info(ids)
        info["survived"] = (self.step_count[ids] >= self.max_steps).float()
        return info
