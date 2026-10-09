"""A stand-in for the drone's own flight controller: turn an attitude command into motor speeds, 500 times a second.

On a real Crazyflie the PC (our brain) never sets motor speeds. It sends a **setpoint** over the radio: a collective
thrust, a roll angle, a pitch angle and a yaw rate. The drone's firmware then holds that attitude with its own fast
controller. To train brains that can fly the real thing, the simulator needs the same layer in between:

    brain (50 Hz) ──setpoint──▶ controller (500 Hz) ──motor speeds──▶ physics

The controller is cascaded, like the Crazyflie's default PID controller:

1. **Angle loop** (outer). How far is the drone from the roll/pitch it was asked for? Ask for a turn rate that
   closes the gap:      rate_wanted = angle_gain · (angle_wanted − angle)            (rad/s)
   Yaw is commanded as a rate directly, as on the Crazyflie.
2. **Rate loop** (inner). How far is the actual turn rate (gyro) from the wanted one? Ask for an angular
   acceleration that closes it, with an integral term that cancels steady pushes (wind, an off-centre battery):
                          accel_wanted = rate_gain · error + rate_i_gain · ∫ error dt  (rad/s²)
                          torque = I · accel_wanted                                     (N·m)
3. **Mixing.** Which 4 motor thrusts give this total thrust and these 3 torques? Invert the motor table in PLAN.md:
       F0 = T/4 − τx/4d − τy/4d − τz/4c        F1 = T/4 − τx/4d + τy/4d + τz/4c
       F2 = T/4 + τx/4d + τy/4d − τz/4c        F3 = T/4 + τx/4d − τy/4d + τz/4c
   with d = arm/√2 and c = km/kf. Each F is clamped to [0, max] and becomes rpm = √(F / kf).

This works in physical units with our own gains, chosen for a similar response (about 0.3 s to tilt 20°); it does
not copy the firmware's numbers, which work in PWM units. The deploy step (Phase 2, P4) maps our setpoint onto the
firmware's (roll and pitch in degrees, yaw rate in deg/s, thrust 10001–60000).

Everything is batched over N drones. The integral is the controller's only memory: an (N, 3) tensor the caller keeps
and resets when an episode starts.
"""

import math
from dataclasses import dataclass

import torch

from mindflight.physics.dynamics import OMEGA, QUAT
from mindflight.physics.params import CRAZYFLIE, CrazyflieParams
from mindflight.physics.rotations import quat_to_euler

# Attitude action a in [−1, 1]⁴ -> setpoint (PLAN.md -> Phase 2 conventions).
THRUST_SPAN = 1.25  # T = hover · (1 + 1.25 · a): a = +1 is 2.25 × hover = the Crazyflie's full thrust
MAX_TILT_RAD = math.radians(30)
MAX_YAW_RATE_RADPS = math.radians(180)


@dataclass(frozen=True)
class ControllerGains:
    """Gains of the cascaded controller (see the top of this file). Units: 1/s for angle and rate gains, 1/s² for
    the integral gain, rad/s for the integral limit (it can't wind up past this much rate error × 1 s)."""

    angle_gain: float = 10.0
    rate_gain_xy: float = 25.0
    rate_i_gain_xy: float = 20.0
    rate_gain_z: float = 12.0
    rate_i_gain_z: float = 5.0
    integral_limit: float = 2.0


GAINS = ControllerGains()


def attitude_action_to_setpoint(action: torch.Tensor, params: CrazyflieParams = CRAZYFLIE) -> torch.Tensor:
    """Brain action (N, 4) in [−1, 1] -> setpoint (N, 4): thrust (N), roll (rad), pitch (rad), yaw rate (rad/s).

        thrust = weight · (1 + 1.25 · a0), clamped to [0, the most the radio can command]
                 (the firmware's top thrust command is 60000 of 65535: about 1.66 × weight for the Crazyflie,
                 less than the motors' 2.25 ×, so a brain must not learn to count on more)
        roll = 30° · a1,   pitch = 30° · a2,   yaw rate = 180°/s · a3
    """
    a = action.clamp(-1.0, 1.0)
    max_thrust_n = min(params.thrust_to_weight * params.weight_n, params.max_command_thrust_n)
    thrust = (params.weight_n * (1.0 + THRUST_SPAN * a[:, 0])).clamp(0.0, max_thrust_n)
    return torch.stack((thrust, MAX_TILT_RAD * a[:, 1], MAX_TILT_RAD * a[:, 2], MAX_YAW_RATE_RADPS * a[:, 3]), dim=1)


def mix(thrust_n: torch.Tensor, torque_nm: torch.Tensor, params: CrazyflieParams = CRAZYFLIE) -> torch.Tensor:
    """Total thrust (N,) and body torques (N, 3) -> the 4 motor speed commands (N, 4) in RPM (see "Mixing" above)."""
    d = params.arm_xy_m
    c = params.km_nm_per_rpm2 / params.kf_n_per_rpm2  # spin torque per newton of thrust (m)
    quarter = thrust_n / 4
    roll = torque_nm[:, 0] / (4 * d)
    pitch = torque_nm[:, 1] / (4 * d)
    yaw = torque_nm[:, 2] / (4 * c)
    per_motor_n = torch.stack(
        (quarter - roll - pitch - yaw, quarter - roll + pitch + yaw, quarter + roll + pitch - yaw, quarter + roll - pitch + yaw),
        dim=1,
    )
    max_per_motor_n = params.kf_n_per_rpm2 * params.max_rpm**2
    return torch.sqrt(per_motor_n.clamp(0.0, max_per_motor_n) / params.kf_n_per_rpm2)


def controller_step(
    state: torch.Tensor,
    setpoint: torch.Tensor,
    integral: torch.Tensor,
    dt: float,
    params: CrazyflieParams = CRAZYFLIE,
    gains: ControllerGains = GAINS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One controller update: (state (N, 17), setpoint (N, 4), integral (N, 3)) -> (rpm command (N, 4), new integral).

    The setpoint is thrust (N), roll (rad), pitch (rad), yaw rate (rad/s), as made by `attitude_action_to_setpoint`.
    Roll and pitch come from the quaternion (Z-Y-X Euler angles); using their errors directly as body-rate requests
    is the small-angle shortcut the Crazyflie firmware also takes.
    """
    euler = quat_to_euler(state[:, QUAT])  # roll, pitch, yaw
    omega = state[:, OMEGA]  # body rates (rad/s): the gyro

    rate_wanted = torch.stack(
        (
            gains.angle_gain * (setpoint[:, 1] - euler[:, 0]),
            gains.angle_gain * (setpoint[:, 2] - euler[:, 1]),
            setpoint[:, 3],
        ),
        dim=1,
    )
    error = rate_wanted - omega
    integral = (integral + error * dt).clamp(-gains.integral_limit, gains.integral_limit)

    accel_xy = gains.rate_gain_xy * error[:, :2] + gains.rate_i_gain_xy * integral[:, :2]
    accel_z = gains.rate_gain_z * error[:, 2:] + gains.rate_i_gain_z * integral[:, 2:]
    inertia = torch.stack(
        (torch.full_like(omega[:, 0], params.ixx_kgm2), torch.full_like(omega[:, 0], params.iyy_kgm2),
         torch.full_like(omega[:, 0], params.izz_kgm2)),
        dim=1,
    )
    torque = inertia * torch.cat((accel_xy, accel_z), dim=1)
    return mix(setpoint[:, 0], torque, params), integral
