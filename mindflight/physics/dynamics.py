"""Rigid-body step: move N drones forward in time by one physics step.

The whole swarm is one tensor `state` shaped (N, 17), one row per drone (PLAN.md -> Conventions):

    columns 0:3    position          (m, world frame)
    columns 3:7    orientation       quaternion (w, x, y, z), body -> world
    columns 7:10   linear velocity   (m/s, world frame)
    columns 10:13  angular velocity  (rad/s, **body** frame)
    columns 13:17  motor speeds      (RPM, m0..m3)

Each `step`:
1. Motors chase the command (`motor_step`), then the props make a force and a torque (`rotor_wrench`).
2. Newton for the center of mass (world frame):  a = R · thrust_body / m + (0, 0, −g)
3. Euler's rigid-body equation (body frame):      ω̇ = I⁻¹ (τ − ω × Iω)
4. Semi-implicit Euler: update the velocities first, then move with the *new* velocities.
5. A floor at z = 0.
"""

import torch

from mindflight.physics.motors import motor_step, rotor_wrench
from mindflight.physics.params import CRAZYFLIE, GRAVITY_MPS2, CrazyflieParams
from mindflight.physics.rotations import quat_integrate, quat_rotate

PHYSICS_DT_S = 0.002  # 500 Hz, PLAN.md -> Timing

# Where each part of the state lives, so code reads state[:, VEL] instead of state[:, 7:10].
POS = slice(0, 3)
QUAT = slice(3, 7)
VEL = slice(7, 10)
OMEGA = slice(10, 13)
RPM = slice(13, 17)
STATE_SIZE = 17


def make_state(
    n: int,
    device: torch.device | str | None = None,
    pos: torch.Tensor | None = None,
    rpm: torch.Tensor | None = None,
) -> torch.Tensor:
    """N drones at rest: identity orientation (level, nose along world +x), zero linear and angular velocity.

    n: number of drones. device: defaults to "cuda" when a GPU is available, else "cpu".
    pos: start position in m, (N, 3) or one (3,) shared by all; default the origin.
    rpm: start motor speeds in RPM, (N, 4) or one (4,) shared by all; default 0 (motors off).
    Returns a float32 state tensor shaped (N, 17).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.zeros(n, STATE_SIZE, dtype=torch.float32, device=device)
    state[:, QUAT.start] = 1.0  # quaternion (1, 0, 0, 0) = no rotation
    if pos is not None:
        state[:, POS] = torch.as_tensor(pos, dtype=torch.float32, device=device)
    if rpm is not None:
        state[:, RPM] = torch.as_tensor(rpm, dtype=torch.float32, device=device)
    return state


def step(
    state: torch.Tensor, rpm_cmd: torch.Tensor, dt: float = PHYSICS_DT_S, params: CrazyflieParams = CRAZYFLIE
) -> torch.Tensor:
    """Advance every drone by `dt` seconds and return the new (N, 17) state (the input is not changed).

    state: (N, 17) as described at the top of this file. rpm_cmd: (N, 4) commanded motor speeds in RPM.

    Linear (world frame), with R the body -> world rotation from the quaternion:
        a = R · (0, 0, total_thrust) / m + (0, 0, −g)                         (m/s²)
    Angular (body frame), with the diagonal inertia I = diag(Ixx, Iyy, Izz):
        ω̇ = I⁻¹ (τ − ω × Iω)                                                  (rad/s²)
    The ω × Iω term is the gyroscopic effect: a body spinning about one axis resists being tipped about another.

    Semi-implicit (symplectic) Euler, which keeps orbits and oscillations from slowly gaining energy:
        v ← v + a·dt,   p ← p + v·dt,   ω ← ω + ω̇·dt,   q ← q ⊗ exp(½ ω dt)

    Floor: a drone that ends below z = 0 is put back on z = 0 with zero linear velocity. Envs decide whether that
    counts as a crash.
    """
    position_m = state[:, POS]
    quat = state[:, QUAT]
    velocity_mps = state[:, VEL]
    omega_radps = state[:, OMEGA]

    # 1. Motors, then the force and torque they make (both in the body frame).
    rpm = motor_step(state[:, RPM], rpm_cmd, dt, params)
    thrust_body_n, torque_body_nm = rotor_wrench(rpm, params)

    # 2. Linear acceleration in the world frame.
    gravity_mps2 = state.new_tensor([0.0, 0.0, -GRAVITY_MPS2])
    accel_mps2 = quat_rotate(quat, thrust_body_n) / params.mass_kg + gravity_mps2

    # 3. Angular acceleration in the body frame. I is diagonal, so I·ω and I⁻¹·x are element-wise.
    inertia_kgm2 = state.new_tensor([params.ixx_kgm2, params.iyy_kgm2, params.izz_kgm2])
    angular_momentum = inertia_kgm2 * omega_radps  # I·ω, kg·m²/s
    gyroscopic_nm = torch.linalg.cross(omega_radps, angular_momentum, dim=-1)  # ω × Iω
    angular_accel_radps2 = (torque_body_nm - gyroscopic_nm) / inertia_kgm2

    # 4. Semi-implicit Euler: new velocities first, then move with them.
    new_velocity_mps = velocity_mps + accel_mps2 * dt
    new_position_m = position_m + new_velocity_mps * dt
    new_omega_radps = omega_radps + angular_accel_radps2 * dt
    new_quat = quat_integrate(quat, new_omega_radps, dt)  # also renormalizes to unit length

    # 5. Floor at z = 0.
    below_floor = new_position_m[:, 2:3] < 0.0  # (N, 1)
    new_position_m = torch.cat((new_position_m[:, :2], new_position_m[:, 2:3].clamp_min(0.0)), dim=1)
    new_velocity_mps = torch.where(below_floor, torch.zeros_like(new_velocity_mps), new_velocity_mps)

    return torch.cat((new_position_m, new_quat, new_velocity_mps, new_omega_radps, rpm), dim=1)
