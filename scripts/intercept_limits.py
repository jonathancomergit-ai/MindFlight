"""How many throws can a Crazyflie reach at all? `python scripts/intercept_limits.py`

A physics bound for the intercept task (docs/phase2-intercept.md), no learning involved. For each throw: could a
drone that starts still at the guard point get to *some* point of the ball's path before the ball does?

The drone is a point that, after a reaction delay, pushes as hard as it may straight toward that point:

    reaction delay   0.15 s = tracker delay (~30 ms) + half a policy step (10 ms) + tilting over (~0.1 s)
    push             thrust up to 1.66 × weight (the most the radio can command), leaning at most `tilt`:
                     30° = roll or pitch alone; 41° = 30° roll and 30° pitch at once (leaning diagonally)

The best acceleration a toward a direction e (unit vector) is the biggest a with thrust T = a·e + g·up and
    tilt:    |a·e_horizontal| ≤ tan(tilt) · (a·e_z + g)   ->   a ≤ tan(tilt)·g / (|e_h| − tan(tilt)·e_z)
    thrust:  |a·e + g·up| ≤ T_max                          ->   a ≤ −g·e_z + √(g²·e_z² − g² + T_max²)
(so climbing is slower than sinking: straight up the drone has only 0.66 g to spare).

Two ways of meeting the ball, over a distance d:
    touch: fly through the ball's path at full speed; be there at the right moment     t = √(2d / a)
    stop:  get onto the path and stop, so the ball runs into you (push, then brake)  t = √(2d · (1/a + 1/a_brake))

Only path points the drone may use count: at least 0.81 m from the person's middle (0.5 m from the body plus the
drone's 6 cm and the body's 25 cm), 0.3–3 m up, and before the ball reaches the person. Prints the share of throws
for which some such point can be reached in time.
"""

import math

import torch

from mindflight.envs.base import SUBSTEPS
from mindflight.envs.intercept import (
    BALL_DT_S,
    DRONE_RADIUS_M,
    GUARD_POINT,
    HARD_ARC_DEG,
    HARD_FLIGHT_S,
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
from mindflight.physics.params import CRAZYFLIE, GRAVITY_MPS2

REACTION_S = 0.15
MAX_THRUST_G = CRAZYFLIE.max_command_thrust_n / CRAZYFLIE.weight_n  # ≈ 1.66
TILTS_DEG = {"30° (roll or pitch alone)": 30.0, "41° (roll + pitch, diagonal)": 40.9}
ARCS_DEG = (10, 20, 30, 45, 60, 90)
FLIGHT_S = (1.4, 1.2, 1.0, 0.9, 0.8, 0.6)


def throws(n: int, arc_deg: float, flight_s: tuple[float, float], generator: torch.Generator):
    """n throws like the env's, flight times in flight_s (s): their paths (n, P, 3) and flight times (n,)."""
    device = generator.device

    def uniform(low, high):
        return low + (high - low) * torch.rand(n, generator=generator, device=device)

    zeros = torch.zeros(n, device=device)
    release = torch.stack((uniform(*THROW_DISTANCE_M), zeros, uniform(*RELEASE_HEIGHT_M)), dim=1)
    target = torch.stack((zeros, uniform(-TORSO_SIDEWAYS_M, TORSO_SIDEWAYS_M), uniform(*TORSO_HEIGHT_M)), dim=1)
    flight = uniform(*flight_s)
    velocity = aim_throws(release, target, flight)
    steps = (flight / BALL_DT_S).round().long()
    _, path = fly_for(release, velocity, steps, sample_every=SUBSTEPS)
    angle = math.radians(arc_deg) * (2 * torch.rand(n, generator=generator, device=device) - 1)
    return rotate_z(path, angle), steps.float() * BALL_DT_S


def best_acceleration(direction: torch.Tensor, tilt_deg: float) -> torch.Tensor:
    """Biggest acceleration (m/s²) the drone can make along unit vectors (…, 3), within the tilt and thrust limits."""
    g, tan_tilt, t_max = GRAVITY_MPS2, math.tan(math.radians(tilt_deg)), MAX_THRUST_G * GRAVITY_MPS2
    e_h = direction[..., :2].norm(dim=-1)
    e_z = direction[..., 2]
    denominator = e_h - tan_tilt * e_z
    by_tilt = torch.where(denominator > 1e-6, tan_tilt * g / denominator.clamp_min(1e-6), torch.full_like(e_z, 1e9))
    by_thrust = -g * e_z + torch.sqrt((g * e_z) ** 2 - g**2 + t_max**2)
    return torch.minimum(by_tilt, by_thrust)


def reachable(path: torch.Tensor, flight: torch.Tensor, tilt_deg: float, stop: bool) -> torch.Tensor:
    """(n,) bool: some allowed path point can be reached before the ball gets there."""
    times = torch.arange(path.shape[1], device=path.device) * POLICY_DT_S
    offset = path - torch.tensor(GUARD_POINT, device=path.device)
    distance = (offset.norm(dim=-1) - TOUCH_DISTANCE_M).clamp_min(1e-6)
    direction = offset / offset.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    push = best_acceleration(direction, tilt_deg)
    if stop:
        needed = REACTION_S + torch.sqrt(2 * distance * (1 / push + 1 / best_acceleration(-direction, tilt_deg)))
    else:
        needed = REACTION_S + torch.sqrt(2 * distance / push)
    safe_radius = PERSON_RADIUS_M + SAFETY_DISTANCE_M + DRONE_RADIUS_M
    height = path[..., 2]
    allowed = (path[..., :2].norm(dim=-1) >= safe_radius) & (height >= 0.3) & (height <= 3.0)
    allowed &= times[None] <= flight[:, None]
    return (allowed & (needed <= times[None])).any(dim=1)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(0)
    modes = [(name, tilt, stop) for name, tilt in TILTS_DEG.items() for stop in (False, True)]
    table = {mode: {} for mode in modes}
    for flight_s in FLIGHT_S:
        for arc in ARCS_DEG:
            path, flight = throws(2000, arc, (flight_s - 0.1, flight_s + 0.1), generator)
            for mode in modes:
                table[mode][flight_s, arc] = 100 * reachable(path, flight, mode[1], mode[2]).float().mean().item()
    print(f"max thrust {MAX_THRUST_G:.2f} × weight, reaction {REACTION_S} s")
    path, flight = throws(4000, HARD_ARC_DEG, HARD_FLIGHT_S, generator)
    print(f"hardest training setting (±{HARD_ARC_DEG:.0f}°, {HARD_FLIGHT_S[0]}–{HARD_FLIGHT_S[1]} s):")
    for name, tilt, stop in modes:
        share = 100 * reachable(path, flight, tilt, stop).float().mean().item()
        print(f"  lean {name}, {'stop on the path' if stop else 'touch at speed'}: {share:.0f} % reachable")
    for name, tilt, stop in modes:
        print(f"\nlean {name}, {'stop on the path' if stop else 'touch at speed'}: % of throws reachable")
        print("flight s ±0.1 | " + " ".join(f"±{a:>3}°" for a in ARCS_DEG))
        for flight_s in FLIGHT_S:
            cells = " ".join(f"{table[name, tilt, stop][flight_s, arc]:5.0f}" for arc in ARCS_DEG)
            print(f"  {flight_s:4.1f}        | {cells}")


if __name__ == "__main__":
    main()
