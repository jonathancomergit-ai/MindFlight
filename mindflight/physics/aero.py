"""Aerodynamic extras (Level 5): air drag, wind, ground effect, and per-drone differences for domain randomization.

The basic model in `dynamics.py` flies in a perfect vacuum: no air pushes back, there is no wind, and the floor
doesn't change the props' thrust. These add the two effects from the same source as our parameters (cf2x.urdf and
gym-pybullet-drones, which cite Förster 2015 and Shi et al. 2019), plus wind:

**Rotor drag.** A spinning prop moving sideways through the air gets pushed back. The faster the props spin and the
faster the drone moves *relative to the air*, the bigger the push. In the body frame:

    F_drag = −(Σ ω_i) · diag(k_xy, k_xy, k_z) · v_air_body       (N)

ω_i = each prop's speed in rad/s (rpm · 2π / 60), v_air = velocity relative to the air = v − wind, in m/s.
At hover (Σ ω ≈ 6,060 rad/s) that is about 5.6 mN per m/s sideways. A 3 m/s gust pushes with 17 mN, about 6 % of
the drone's weight: enough to blow it off its spot, not enough to overpower it.

**Wind** enters only through drag: the air moves, so v_air changes, so the drag pushes the drone along.

**Ground effect.** Close to the floor the props' downwash has nowhere to go, so they make extra thrust:

    extra thrust = Σ F_i · c_ge · (r_prop / (4 · h))²          (N, along body z)

h = height above the floor, clipped to at least h_clip (about 3.8 cm) so it stays finite. With c_ge = 11.37 that is
+27 % thrust at h_clip, +3.8 % at 10 cm and under 0.1 % above 70 cm. It is skipped when the drone is tipped past 90°.

**Per-drone differences** (`DroneVariation`): mass and inertia scaled together (a heavier drone of the same shape),
thrust coefficient scaled (weaker or stronger motors and props), and the motor time constant scaled (laggier or
snappier motors). Training on many slightly different drones makes a policy that copes with the real one.
"""

import math
from dataclasses import dataclass

import torch

from mindflight.physics.device_cache import cached_on_device
from mindflight.physics.params import CRAZYFLIE, CrazyflieParams
from mindflight.physics.rotations import quat_rotate, quat_rotate_inverse

RPM_TO_RADPS = 2 * math.pi / 60


@dataclass
class DroneVariation:
    """Per-drone differences and the air around each drone. Every tensor is (N,) or (N, 3); None means "standard".

    mass_scale:  mass and inertia × this (1.2 = 20 % heavier)
    thrust_scale: thrust and drag-torque coefficients × this (0.9 = 10 % weaker props)
    motor_tau_scale: motor time constant × this (1.5 = 50 % laggier)
    wind_mps: wind velocity, world frame (m/s), (N, 3)
    drag: apply rotor drag (needed for wind to do anything)
    ground_effect: apply ground effect
    world_id, downwash: apply downwash between drones that share a world (see `downwash_world`)
    com_offset_m: the centre of mass sits (dx, dy) off the middle (an off-centre battery). The thrust then pushes
        through a point that is not the centre of mass, which twists the drone:
            τ = r × F,  r = (−dx, −dy, 0) from the centre of mass to the middle, F = (0, 0, T)
            -> τx = −dy · T,  τy = dx · T            (a heavy nose, dx > 0, pitches the nose down)
    group_size: if set, the drones come in consecutive worlds of exactly this many (world = index // group_size),
        and downwash is worked out per world with `downwash_grouped` (cost N · G instead of N²). Packs use this.
    """

    mass_scale: torch.Tensor | None = None
    thrust_scale: torch.Tensor | None = None
    motor_tau_scale: torch.Tensor | None = None
    wind_mps: torch.Tensor | None = None
    drag: bool = False
    ground_effect: bool = False
    world_id: torch.Tensor | None = None  # (N,) long: drones with the same id share the air (for downwash)
    downwash: bool = False
    com_offset_m: torch.Tensor | None = None  # (N, 2): centre of mass off the geometric centre in x, y (body, m)
    group_size: int | None = None  # drones per world, worlds in consecutive blocks (see `downwash_grouped`)


@cached_on_device
def drag_coeffs_on(device: torch.device, dtype: torch.dtype, params: CrazyflieParams = CRAZYFLIE) -> torch.Tensor:
    """(k_xy, k_xy, k_z) as a (3,) tensor already on `device` (made once: see device_cache.py)."""
    return torch.tensor([params.drag_coeff_xy, params.drag_coeff_xy, params.drag_coeff_z], device=device, dtype=dtype)


def rotor_drag_world(
    quat: torch.Tensor,
    velocity_mps: torch.Tensor,
    rpm: torch.Tensor,
    wind_mps: torch.Tensor | None = None,
    params: CrazyflieParams = CRAZYFLIE,
) -> torch.Tensor:
    """Rotor drag force in the world frame, (N, 3) in N. See the formula at the top of this file.

    quat (N, 4) body -> world; velocity_mps (N, 3) world; rpm (N, 4); wind_mps (N, 3) world or None (still air).
    """
    air_velocity = velocity_mps if wind_mps is None else velocity_mps - wind_mps
    air_velocity_body = quat_rotate_inverse(quat, air_velocity)
    total_radps = (rpm * RPM_TO_RADPS).sum(dim=1, keepdim=True)  # Σ ω_i, (N, 1)
    coeff = drag_coeffs_on(air_velocity.device, air_velocity.dtype, params)
    drag_body = -total_radps * coeff * air_velocity_body
    return quat_rotate(quat, drag_body)


def ground_effect_clip_height_m(params: CrazyflieParams = CRAZYFLIE) -> float:
    """h_clip from gym-pybullet-drones: 0.25 · r_prop · √(15 · max_rpm² · kf · c_ge / max_thrust), about 3.8 cm."""
    max_thrust_n = 4 * params.kf_n_per_rpm2 * params.max_rpm**2
    return 0.25 * params.prop_radius_m * math.sqrt(
        15 * params.max_rpm**2 * params.kf_n_per_rpm2 * params.ground_effect_coeff / max_thrust_n
    )


def ground_effect_thrust(
    height_m: torch.Tensor, thrust_n: torch.Tensor, upright: torch.Tensor, params: CrazyflieParams = CRAZYFLIE
) -> torch.Tensor:
    """Extra thrust from flying close to the floor, (N,) in N, along body z. See the top of this file.

    height_m (N,) above the floor; thrust_n (N,) the props' normal total thrust; upright (N,) = R_zz = cos(tilt).
    """
    h = height_m.clamp_min(ground_effect_clip_height_m(params))
    extra = thrust_n * params.ground_effect_coeff * (params.prop_radius_m / (4 * h)) ** 2
    return torch.where(upright > 0, extra, torch.zeros_like(extra))


def downwash_world(
    position_m: torch.Tensor, world_id: torch.Tensor, params: CrazyflieParams = CRAZYFLIE, min_gap_m: float = 0.1
) -> torch.Tensor:
    """Push from the air thrown down by drones flying above, (N, 3) in N (only the z part is non-zero).

    For every pair in the same world where drone j is above drone i (Δz = z_j − z_i > 0, Δxy = sideways distance):

        F_z on i = −α · exp(−½ · (Δxy / β)²),   α = c1 · (r_prop / (4 · Δz))²,   β = c2 · Δz + c3

    α is how hard the column of air pushes (weaker the further below you are), and the Gaussian says how wide the
    column is (β grows with Δz). This is the empirical fit from gym-pybullet-drones, used as-is. It is only sensible
    when the drones are a fair way apart: Δz is clipped to at least `min_gap_m` (10 cm) and |β| to at least 2 cm,
    because the raw formula blows up as Δz → 0 and its β crosses zero near Δz = 0.69 m. Even at Δz = 10 cm the raw
    α is 7.5 N, about 28 × a Crazyflie's weight, so the total push on a drone is capped at its own weight (ASSUMED:
    a column of air from above can cancel the drone's lift, not throw it at the floor at several g).

    Cost: every pair of drones, N × N. Fine for a race of a few drones; too much for 4,096 training drones, which
    each fly alone in their own world anyway.
    """
    dz = position_m[None, :, 2] - position_m[:, None, 2]  # (N, N): z_j − z_i, j = column
    dxy = (position_m[None, :, :2] - position_m[:, None, :2]).norm(dim=-1)
    above = (dz > 0) & (world_id[None, :] == world_id[:, None])
    push = downwash_pair_push(dz, dxy, above, params, min_gap_m).sum(dim=1)
    push = push.clamp(max=params.weight_n)  # ASSUMED cap, see above
    return torch.stack((torch.zeros_like(push), torch.zeros_like(push), -push), dim=1)


def downwash_pair_push(
    dz: torch.Tensor, dxy: torch.Tensor, above: torch.Tensor, params: CrazyflieParams = CRAZYFLIE, min_gap_m: float = 0.1
) -> torch.Tensor:
    """Downward push (N, ≥ 0) on drone i from drone j, for any shape of pair tensors: α · exp(−½ (Δxy / β)²) where
    `above` (j is above i and they share the air), else 0. The formula and its clips are in `downwash_world`."""
    dz = dz.clamp_min(min_gap_m)
    alpha = params.downwash_coeff_1 * (params.prop_radius_m / (4 * dz)) ** 2
    beta = params.downwash_coeff_2 * dz + params.downwash_coeff_3
    beta = torch.where(beta.abs() < 0.02, torch.full_like(beta, 0.02), beta)
    return torch.where(above, alpha * torch.exp(-0.5 * (dxy / beta) ** 2), torch.zeros_like(dz))


def downwash_grouped(
    position_m: torch.Tensor, params: CrazyflieParams = CRAZYFLIE, min_gap_m: float = 0.1
) -> torch.Tensor:
    """`downwash_world` for drones that come in worlds of the same size G: position_m is (W, G, 3), one row of G
    drones per world, and the result is the push (W, G, 3) in N (only z is non-zero). Same formula and caps.

    Only the G × G pairs inside each world are worked out: W · G² pairs instead of (W · G)². For 4,096 drones in
    worlds of 6 that is 25 thousand pairs instead of 17 million, so downwash can stay on while training.
    """
    dz = position_m[:, None, :, 2] - position_m[:, :, None, 2]  # (W, G, G): z_j − z_i, j = last index
    dxy = (position_m[:, None, :, :2] - position_m[:, :, None, :2]).norm(dim=-1)
    push = downwash_pair_push(dz, dxy, dz > 0, params, min_gap_m).sum(dim=-1)  # (W, G)
    push = push.clamp(max=params.weight_n)  # ASSUMED cap, see downwash_world
    return torch.stack((torch.zeros_like(push), torch.zeros_like(push), -push), dim=-1)


def aero_force_world(
    quat: torch.Tensor,
    position_m: torch.Tensor,
    velocity_mps: torch.Tensor,
    rpm: torch.Tensor,
    thrust_body_n: torch.Tensor,
    variation: DroneVariation,
    params: CrazyflieParams = CRAZYFLIE,
) -> torch.Tensor:
    """The extra forces `variation` switches on, summed, in the world frame, (N, 3) in N: drag (with wind),
    downwash and ground effect. Zeros if none is on. Called by `dynamics.step`."""
    force = torch.zeros_like(velocity_mps)
    if variation.drag:
        force = force + rotor_drag_world(quat, velocity_mps, rpm, variation.wind_mps, params)
    if variation.downwash and variation.group_size is not None:  # worlds of G in a row: the cheap way
        group = variation.group_size
        force = force + downwash_grouped(position_m.reshape(-1, group, 3), params).reshape(-1, 3)
    elif variation.downwash and variation.world_id is not None:
        force = force + downwash_world(position_m, variation.world_id, params)
    if variation.ground_effect:
        x, y = quat[:, 1], quat[:, 2]
        upright = 1 - 2 * (x * x + y * y)  # R_zz = cos(tilt)
        extra_n = ground_effect_thrust(position_m[:, 2], thrust_body_n[:, 2], upright, params)
        extra_body = torch.stack((torch.zeros_like(extra_n), torch.zeros_like(extra_n), extra_n), dim=1)
        force = force + quat_rotate(quat, extra_body)
    return force
