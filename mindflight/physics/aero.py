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
    """

    mass_scale: torch.Tensor | None = None
    thrust_scale: torch.Tensor | None = None
    motor_tau_scale: torch.Tensor | None = None
    wind_mps: torch.Tensor | None = None
    drag: bool = False
    ground_effect: bool = False


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
    coeff = air_velocity.new_tensor([params.drag_coeff_xy, params.drag_coeff_xy, params.drag_coeff_z])
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
