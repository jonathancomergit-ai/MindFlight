"""How many throws can a Crazyflie reach at all? `python scripts/intercept_limits.py`

A physics bound for the intercept task (docs/phase2-intercept.md), no learning involved. For each throw: could a
drone that starts still at the guard point get to *some* point of the ball's path before the ball does?

The drone is modelled as a point that, after a reaction delay, accelerates at its best sideways rate:

    reaction delay   0.15 s = tracker delay (~30 ms) + half a policy step (10 ms) + tilting over (~0.1 s)
    acceleration     g · tan(tilt):  30° roll or pitch  ->  5.7 m/s²
                                     30° roll and 30° pitch at once (a 41° lean, diagonally)  ->  8.6 m/s²

and two ways of meeting the ball:

    touch: fly through the ball's path at full speed; it must be there at the right moment (distance = ½ a t²)
    stop:  get onto the path and stop there, so the ball runs into it (distance = ¼ a t²: speed up, then brake)

Only path points the drone may use count: at least 0.81 m from the person's middle (0.5 m from the body plus the
drone's 6 cm and the body's 25 cm) and before the ball reaches the person. Prints the share of throws reachable.
"""

import math

import torch

from mindflight.envs.base import SUBSTEPS
from mindflight.envs.intercept import (
    BALL_DT_S,
    DRONE_RADIUS_M,
    GUARD_POINT,
    PERSON_RADIUS_M,
    POLICY_DT_S,
    RELEASE_HEIGHT_M,
    SAFETY_DISTANCE_M,
    THROW_DISTANCE_M,
    TORSO_HEIGHT_M,
    TORSO_SIDEWAYS_M,
    TOUCH_DISTANCE_M,
    aim_throws,
    fly_for,
    rotate_z,
)
from mindflight.physics.params import GRAVITY_MPS2

REACTION_S = 0.15
ACCELERATIONS = {"30° one axis": GRAVITY_MPS2 * math.tan(math.radians(30)), "30°+30° diagonal": 8.65}
ARCS_DEG = (10, 20, 30, 45, 60, 90)
FLIGHT_S = (1.6, 1.4, 1.2, 1.0, 0.8, 0.6)


def throws(n: int, arc_deg: float, flight_s: float, generator: torch.Generator):
    """n throws like the env's (flight times ± 0.1 s around flight_s): their paths (n, P, 3) and flight times (n,)."""

    device = generator.device

    def uniform(low, high):
        return low + (high - low) * torch.rand(n, generator=generator, device=device)

    zeros = torch.zeros(n, device=device)
    release = torch.stack((uniform(*THROW_DISTANCE_M), zeros, uniform(*RELEASE_HEIGHT_M)), dim=1)
    target = torch.stack((zeros, uniform(-TORSO_SIDEWAYS_M, TORSO_SIDEWAYS_M), uniform(*TORSO_HEIGHT_M)), dim=1)
    flight = uniform(flight_s - 0.1, flight_s + 0.1)
    velocity = aim_throws(release, target, flight)
    steps = (flight / BALL_DT_S).round().long()
    _, path = fly_for(release, velocity, steps, sample_every=SUBSTEPS)
    angle = math.radians(arc_deg) * (2 * torch.rand(n, generator=generator, device=device) - 1)
    return rotate_z(path, angle), steps.float() * BALL_DT_S


def reachable(path: torch.Tensor, flight: torch.Tensor, accel: float, stop: bool) -> torch.Tensor:
    """(n,) bool: some allowed path point can be reached before the ball gets there."""
    times = torch.arange(path.shape[1], device=path.device) * POLICY_DT_S
    guard = torch.tensor(GUARD_POINT, device=path.device)
    distance = ((path - guard).norm(dim=-1) - TOUCH_DISTANCE_M).clamp_min(0.0)
    needed = REACTION_S + (2 * torch.sqrt(distance / accel) if stop else torch.sqrt(2 * distance / accel))
    safe_radius = PERSON_RADIUS_M + SAFETY_DISTANCE_M + DRONE_RADIUS_M
    allowed = (path[..., :2].norm(dim=-1) >= safe_radius) & (path[..., 2] >= 0.3) & (times[None] <= flight[:, None])
    return (allowed & (needed <= times[None])).any(dim=1)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(0)
    modes = [(name, accel, stop) for name, accel in ACCELERATIONS.items() for stop in (False, True)]
    table = {mode: {} for mode in modes}
    for flight_s in FLIGHT_S:
        for arc in ARCS_DEG:
            path, flight = throws(4000, arc, flight_s, generator)
            for mode in modes:
                table[mode][flight_s, arc] = 100 * reachable(path, flight, mode[1], mode[2]).float().mean().item()
    for name, accel, stop in modes:
        print(f"\n{name} ({accel:.1f} m/s²), {'stop on the path' if stop else 'touch at speed'}: % reachable")
        print("flight s | " + " ".join(f"±{a:>3}°" for a in ARCS_DEG))
        for flight_s in FLIGHT_S:
            cells = " ".join(f"{table[name, accel, stop][flight_s, arc]:5.0f}" for arc in ARCS_DEG)
            print(f"  {flight_s:4.1f}   | {cells}")


if __name__ == "__main__":
    main()
