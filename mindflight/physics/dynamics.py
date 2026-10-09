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
4. Move forward in time:
   - Position and linear velocity: the force is held constant over the step, so the exact constant-acceleration
     formulas apply (p ← p + v·dt + ½·a·dt², v ← v + a·dt).
   - Orientation and angular velocity: RK4 (fourth-order Runge-Kutta, explained below).
5. A floor at z = 0.

Why RK4 for rotation? Even with no torque, the gyroscopic term ω × Iω keeps turning the direction of ω, so ω moves
along a curve. One Euler step follows a straight line from the start of the step and lands a little outside that
curve, and the error piles up: a tumbling drone slowly gains spin energy it never had. RK4 instead asks "how fast
are things changing?" four times inside the step (at the start, twice at the middle, at the end) and takes a
weighted average of those slopes, 1/6 · (k1 + 2·k2 + 2·k3 + k4). Its error shrinks like dt⁴ instead of dt, so at
500 Hz a tumble keeps its energy and angular momentum to well under 0.1 %.
"""

import torch

from mindflight.physics.motors import motor_step, rotor_wrench
from mindflight.physics.params import CRAZYFLIE, GRAVITY_MPS2, CrazyflieParams
from mindflight.physics.rotations import quat_multiply, quat_normalize, quat_rotate

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


def rotation_rates(
    quat: torch.Tensor, omega_radps: torch.Tensor, torque_body_nm: torch.Tensor, inertia_kgm2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """How fast orientation and body angular velocity are changing right now: returns (q̇, ω̇).

    quat: (N, 4) body -> world. omega_radps: (N, 3) body frame, rad/s. torque_body_nm: (N, 3) in N·m.
    inertia_kgm2: (3,) the diagonal (Ixx, Iyy, Izz) in kg·m².

        q̇ = ½ q ⊗ (0, ω)                  (1/s; body-frame ω multiplies on the right)
        ω̇ = I⁻¹ (τ − ω × Iω)              (rad/s²)

    I is diagonal, so I·ω and I⁻¹·x are element-wise. The ω × Iω term is the gyroscopic effect: a body spinning
    about one axis resists being tipped about another.
    """
    omega_quat = torch.cat((torch.zeros_like(omega_radps[:, :1]), omega_radps), dim=1)  # (0, ω)
    quat_rate = 0.5 * quat_multiply(quat, omega_quat)
    angular_momentum = inertia_kgm2 * omega_radps  # I·ω, kg·m²/s
    gyroscopic_nm = torch.linalg.cross(omega_radps, angular_momentum, dim=-1)  # ω × Iω
    angular_accel_radps2 = (torque_body_nm - gyroscopic_nm) / inertia_kgm2
    return quat_rate, angular_accel_radps2


def step(
    state: torch.Tensor, rpm_cmd: torch.Tensor, dt: float = PHYSICS_DT_S, params: CrazyflieParams = CRAZYFLIE
) -> torch.Tensor:
    """Advance every drone by `dt` seconds and return the new (N, 17) state (the input is not changed).

    state: (N, 17) as described at the top of this file. rpm_cmd: (N, 4) commanded motor speeds in RPM.

    The motor force and torque are worked out once, at the start of the step, and held constant over it.

    Linear (world frame), with R the body -> world rotation from the quaternion at the start of the step:
        a = R · (0, 0, total_thrust) / m + (0, 0, −g)                         (m/s²)
        p ← p + v·dt + ½·a·dt²,   v ← v + a·dt      (exact when a is constant over the step)

    Angular (body frame), with the diagonal inertia I = diag(Ixx, Iyy, Izz), see `rotation_rates`:
        q̇ = ½ q ⊗ (0, ω),   ω̇ = I⁻¹ (τ − ω × Iω)
    q and ω are advanced together with RK4. Each k is a pair of rates (q̇, ω̇) from `rotation_rates`:
        k1 = rates at (q, ω)                          the slope at the start
        k2 = rates at (q, ω) + ½·dt·k1                the slope at the middle, using k1 to get there
        k3 = rates at (q, ω) + ½·dt·k2                the middle again, using the better guess k2
        k4 = rates at (q, ω) + dt·k3                  the slope at the end
        (q, ω) ← (q, ω) + dt/6 · (k1 + 2·k2 + 2·k3 + k4)
    The gyroscopic term is recomputed inside every k. q is renormalized to unit length at the end.

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

    # 3. Position and linear velocity: exact for an acceleration that is constant over the step.
    new_position_m = position_m + velocity_mps * dt + 0.5 * accel_mps2 * dt**2
    new_velocity_mps = velocity_mps + accel_mps2 * dt

    # 4. Orientation and angular velocity together, with RK4 (torque held constant over the step).
    inertia_kgm2 = state.new_tensor([params.ixx_kgm2, params.iyy_kgm2, params.izz_kgm2])
    quat_k1, omega_k1 = rotation_rates(quat, omega_radps, torque_body_nm, inertia_kgm2)
    quat_k2, omega_k2 = rotation_rates(
        quat + 0.5 * dt * quat_k1, omega_radps + 0.5 * dt * omega_k1, torque_body_nm, inertia_kgm2
    )
    quat_k3, omega_k3 = rotation_rates(
        quat + 0.5 * dt * quat_k2, omega_radps + 0.5 * dt * omega_k2, torque_body_nm, inertia_kgm2
    )
    quat_k4, omega_k4 = rotation_rates(quat + dt * quat_k3, omega_radps + dt * omega_k3, torque_body_nm, inertia_kgm2)
    new_quat = quat_normalize(quat + dt / 6.0 * (quat_k1 + 2.0 * quat_k2 + 2.0 * quat_k3 + quat_k4))
    new_omega_radps = omega_radps + dt / 6.0 * (omega_k1 + 2.0 * omega_k2 + 2.0 * omega_k3 + omega_k4)

    # 5. Floor at z = 0.
    below_floor = new_position_m[:, 2:3] < 0.0  # (N, 1)
    new_position_m = torch.cat((new_position_m[:, :2], new_position_m[:, 2:3].clamp_min(0.0)), dim=1)
    new_velocity_mps = torch.where(below_floor, torch.zeros_like(new_velocity_mps), new_velocity_mps)

    return torch.cat((new_position_m, new_quat, new_velocity_mps, new_omega_radps, rpm), dim=1)
