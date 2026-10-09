"""Motor model: how fast the 4 propellers spin, and what force and twist they put on the drone.

Two steps, both batched over N drones:
1. `motor_step`: a real motor can't change speed instantly. Its RPM chases the commanded RPM with a first-order lag.
2. `rotor_wrench`: each spinning prop pushes the drone up (thrust) and twists it (torque). We add up all 4.

Motor layout, spin directions and torque signs follow PLAN.md -> "Motor layout" (X configuration, body frame FLU:
x forward, y left, z up).
"""

import math

import torch

from mindflight.physics.device_cache import cached_on_device
from mindflight.physics.params import CRAZYFLIE, CrazyflieParams

# Each motor's (x, y) position as a multiple of d = arm / sqrt(2). Order is m0, m1, m2, m3.
# Kept as plain Python numbers too, so `motor_layout_on` can build the GPU copy without touching CPU tensors.
MOTOR_POSITION_SIGN_ROWS = (
    (+1.0, -1.0),  # m0 front-right
    (-1.0, -1.0),  # m1 back-right
    (-1.0, +1.0),  # m2 back-left
    (+1.0, +1.0),  # m3 front-left
)
MOTOR_POSITION_SIGNS = torch.tensor(MOTOR_POSITION_SIGN_ROWS)  # (4, 2)

# Each motor's (x, y) position in meters for the Crazyflie, shape (4, 2).
MOTOR_POSITIONS = MOTOR_POSITION_SIGNS * CRAZYFLIE.arm_xy_m

# Spin direction seen from above: +1 = counter-clockwise (CCW), -1 = clockwise (CW). Shape (4,).
MOTOR_SPIN_VALUES = (+1.0, -1.0, +1.0, -1.0)
MOTOR_SPIN = torch.tensor(MOTOR_SPIN_VALUES)


@cached_on_device
def motor_layout_on(
    device: torch.device, dtype: torch.dtype, params: CrazyflieParams = CRAZYFLIE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The motor layout as tensors already on `device`: (motor_x_m, motor_y_m, spin), each shaped (4,).

    motor_x_m, motor_y_m: each motor's position in meters, (±d, ±d) with d = arm / √2 (PLAN.md -> Motor layout).
    spin: +1 for CCW, −1 for CW.

    Made once per (device, dtype, params) and reused, so `rotor_wrench` never copies from the CPU during a step
    (see `device_cache.py` for why that matters). Read-only: every caller shares these tensors.
    """
    d = params.arm_xy_m
    motor_x_m = torch.tensor([x_sign * d for x_sign, _ in MOTOR_POSITION_SIGN_ROWS], device=device, dtype=dtype)
    motor_y_m = torch.tensor([y_sign * d for _, y_sign in MOTOR_POSITION_SIGN_ROWS], device=device, dtype=dtype)
    spin = torch.tensor(MOTOR_SPIN_VALUES, device=device, dtype=dtype)
    return motor_x_m, motor_y_m, spin


def motor_step(
    rpm: torch.Tensor, rpm_cmd: torch.Tensor, dt: float, params: CrazyflieParams = CRAZYFLIE
) -> torch.Tensor:
    """Advance the motor speeds by one time step of `dt` seconds.

    A first-order lag with time constant tau: the motor closes a fixed fraction of the gap to the command each step,
        rpm_new = rpm + (rpm_cmd - rpm) * (1 - exp(-dt / tau))
    This is the exact solution of d(rpm)/dt = (rpm_cmd - rpm) / tau over one step, so it is correct for any dt.
    After tau seconds the motor has covered 1 - 1/e = 63.2 % of a jump.

    The command and the result are clamped to [0, max_rpm]: a motor can't spin backwards or past full throttle.

    rpm, rpm_cmd: (N, 4) in RPM. Returns the new (N, 4) RPM.
    """
    rpm_cmd = rpm_cmd.clamp(0.0, params.max_rpm)
    fraction_closed = 1.0 - math.exp(-dt / params.motor_time_constant_s)
    new_rpm = rpm + (rpm_cmd - rpm) * fraction_closed
    return new_rpm.clamp(0.0, params.max_rpm)


def rotor_wrench(rpm: torch.Tensor, params: CrazyflieParams = CRAZYFLIE) -> tuple[torch.Tensor, torch.Tensor]:
    """Total force and torque the 4 props put on the drone, in the body frame.

    Per motor i:  thrust F_i = kf * rpm_i^2 (N, along body +z),  drag torque Q_i = km * rpm_i^2 (N*m).

    thrust_body = (0, 0, F0 + F1 + F2 + F3)
    torque_body:
        roll  tau_x = sum of  y_i * F_i  = d * (-F0 - F1 + F2 + F3)   (from tau = r x F, r = (x_i, y_i, 0))
        pitch tau_y = sum of -x_i * F_i  = d * (-F0 + F1 + F2 - F3)
        yaw   tau_z = sum of -spin_i * Q_i = -Q0 + Q1 - Q2 + Q3        (a CCW prop pushes the body CW)

    rpm: (N, 4). Returns thrust_body (N, 3) in N and torque_body (N, 3) in N*m.
    """
    thrust_per_motor_n = params.kf_n_per_rpm2 * rpm**2  # (N, 4)
    drag_torque_per_motor_nm = params.km_nm_per_rpm2 * rpm**2  # (N, 4)

    motor_x_m, motor_y_m, spin = motor_layout_on(rpm.device, rpm.dtype, params)  # each (4,), already on rpm's device

    zeros = torch.zeros_like(rpm[:, 0])
    thrust_body = torch.stack([zeros, zeros, thrust_per_motor_n.sum(dim=1)], dim=1)

    roll_nm = (motor_y_m * thrust_per_motor_n).sum(dim=1)
    pitch_nm = (-motor_x_m * thrust_per_motor_n).sum(dim=1)
    yaw_nm = (-spin * drag_torque_per_motor_nm).sum(dim=1)
    torque_body = torch.stack([roll_nm, pitch_nm, yaw_nm], dim=1)

    return thrust_body, torque_body
