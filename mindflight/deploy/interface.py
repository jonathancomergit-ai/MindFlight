"""The ONE place where our numbers become the Crazyflie's numbers, and back.

Everything else in `mindflight/deploy/` works in our units (PLAN.md -> Conventions). Only this file knows the
firmware's units and signs. If a sign is wrong, it is wrong here, and `tests/test_deploy_interface.py` should catch it.

## Setpoint: PC -> drone

We send the setpoint with cflib's `Commander.send_setpoint(roll, pitch, yawrate, thrust)` (the "RPYT" commander).

| Number | Ours (what the brain's action means) | cflib `send_setpoint` argument | So we send |
|---|---|---|---|
| roll | rad, + = right side down (FLU body) | degrees, + = right side down | `+degrees(roll)` |
| pitch | rad, + = nose down | degrees, + = nose down (see below) | `+degrees(pitch)` |
| yaw rate | rad/s, + = counter-clockwise from above | deg/s, + = **clockwise** from above | `−degrees(yaw_rate)` |
| thrust | newtons, total of 4 motors | whole number 10001..60000 | see "Thrust" |

Why pitch keeps its sign although two things flip it:
1. cflib packs `-pitch` into the radio packet (`struct.pack('<fffH', roll, -pitch, yawrate, thrust)`, cflib 0.1.34
   `crazyflie/commander.py`).
2. The firmware's pitch is "legacy": + = nose UP. `kalman_core.c` saves `.pitch = -pitch` "adjusted for the legacy
   CF2 body coordinate system", and `controller_pid.c` feeds the rate loop `-sensors->gyro.y` to match.
Two flips cancel: cflib pitch + = nose down = forward, same as ours.

Why yaw rate flips: `crtp_commander_rpyt.c` says "legacy rate input is inverted" and does
`setpoint->attitudeRate.yaw = -values->yaw`. Inside, yaw rate is counter-clockwise like ours, so cflib's is clockwise.

Roll is never flipped (firmware roll = standard right-hand roll about +x forward = right side down).

## Thrust

Stock firmware (battery compensation on, 2.1+ propellers, the default build) turns the thrust number into newtons
in a straight line (`motors.c`, `motorsCompensateBatteryVoltage`):

    thrust per motor (N) = command / 65535 · 0.12 N        (THRUST_MAX in platform_defaults_cf2.h)

so it is proportional. We anchor the line at the drone's hover command, which you can measure on the real drone:

    command = thrust_n / weight_n · hover_thrust_command

With the default numbers that is the firmware line exactly (hover_thrust_command = weight / 0.48 N · 65535). While
flying we keep the command in [10001, 60000] (cflib's "next to no power" .. "full power"): below the firmware's
THRUST_MIN (≈ 7000) it stops the motors, and a motor stop in the air also resets the attitude controller.
"Motors off" is a separate, deliberate command: thrust 0.

## State estimate: drone -> PC

We log these firmware variables (all `float`, 4 bytes) at 100 Hz, the firmware's fastest log rate (log periods are
counted in 10 ms steps). One log packet carries at most 26 bytes of data (cflib `LogConfig.MAX_LEN`), so 6 floats
per block at most: three blocks.

| Variable | Unit, frame | Ours |
|---|---|---|
| stateEstimate.x, .y, .z | m, world (z up, x forward at start, y left) | position, same |
| stateEstimate.vx, .vy, .vz | m/s, world | velocity, same |
| stateEstimate.qw, .qx, .qy, .qz | body -> world quaternion, *not* legacy-adjusted | quaternion (w, x, y, z), same |
| gyro.x, .y, .z | **deg/s**, body (x forward, y left, z up) | angular velocity rad/s, so `radians(...)` |
| pm.vbat | V | battery volts |
"""

import math
from dataclasses import dataclass

from mindflight.physics.controller import MAX_TILT_RAD, MAX_YAW_RATE_RADPS
from mindflight.physics.params import CrazyflieParams

THRUST_COMMAND_FULL_SCALE = 65535  # the firmware's thrust number for "THRUST_MAX per motor"

# ---------- logging ----------

LOG_PERIOD_MS = 10  # 100 Hz: the fastest the firmware logs (periods are in 10 ms units)
LOG_BLOCK_MAX_BYTES = 26  # cflib LogConfig.MAX_LEN: data bytes in one log packet
FLOAT_BYTES = 4
LOG_BLOCKS: dict[str, list[str]] = {
    "position_velocity": [
        "stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
        "stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz",
    ],
    "attitude": ["stateEstimate.qw", "stateEstimate.qx", "stateEstimate.qy", "stateEstimate.qz"],
    "rates_battery": ["gyro.x", "gyro.y", "gyro.z", "pm.vbat"],
}


@dataclass(frozen=True)
class Estimate:
    """The drone's latest state estimate, in OUR units (see the table at the top).

    time_s: PC clock when the newest log block of this estimate arrived. oldest_s: when the oldest one arrived
    (the estimate is only as fresh as its oldest part). firmware_ms: the drone's clock on the newest block.
    """

    time_s: float
    oldest_s: float
    firmware_ms: int
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    gyro_radps: tuple[float, float, float]
    battery_v: float

    def numbers(self) -> list[float]:
        """Every number in it (for a NaN check)."""
        return [*self.position_m, *self.velocity_mps, *self.quaternion_wxyz, *self.gyro_radps, self.battery_v]

    @property
    def tilt_rad(self) -> float:
        """Angle between the drone's up axis and world up: acos(R_zz), R_zz = 1 − 2(x² + y²) (rad)."""
        w, x, y, z = self.quaternion_wxyz
        norm2 = w * w + x * x + y * y + z * z
        r_zz = 1.0 - 2.0 * (x * x + y * y) / max(norm2, 1e-12)
        return math.acos(max(-1.0, min(1.0, r_zz)))


def estimate_from_log(values: dict[str, float], time_s: float, oldest_s: float, firmware_ms: int) -> Estimate:
    """Firmware log variables (names as in LOG_BLOCKS) -> an Estimate in our units. Only gyro changes: deg/s -> rad/s."""
    v = values
    return Estimate(
        time_s=time_s,
        oldest_s=oldest_s,
        firmware_ms=firmware_ms,
        position_m=(v["stateEstimate.x"], v["stateEstimate.y"], v["stateEstimate.z"]),
        velocity_mps=(v["stateEstimate.vx"], v["stateEstimate.vy"], v["stateEstimate.vz"]),
        quaternion_wxyz=(v["stateEstimate.qw"], v["stateEstimate.qx"], v["stateEstimate.qy"], v["stateEstimate.qz"]),
        gyro_radps=(math.radians(v["gyro.x"]), math.radians(v["gyro.y"]), math.radians(v["gyro.z"])),
        battery_v=v["pm.vbat"],
    )


# ---------- setpoint ----------


@dataclass(frozen=True)
class FirmwareSetpoint:
    """The four numbers of cflib `Commander.send_setpoint`: degrees, degrees, deg/s (+ = clockwise), thrust integer."""

    roll_deg: float
    pitch_deg: float
    yawrate_deg: float
    thrust: int


MOTORS_OFF = FirmwareSetpoint(0.0, 0.0, 0.0, 0)  # also the "unlock" packet: the firmware needs one thrust-0 first


def thrust_newtons_to_command(thrust_n: float, params: CrazyflieParams) -> int:
    """Total thrust (N) -> the firmware's thrust integer, for flying: command = thrust / weight · hover command,
    rounded and kept in [thrust_command_min, thrust_command_max]. NaN gives the minimum (never full power)."""
    if not math.isfinite(thrust_n):
        return params.thrust_command_min
    command = round(thrust_n / params.weight_n * params.hover_thrust_command)
    return int(min(max(command, params.thrust_command_min), params.thrust_command_max))


def thrust_command_to_newtons(command: float, params: CrazyflieParams) -> float:
    """The inverse line: thrust integer -> total thrust (N) we expect, command / hover command · weight."""
    return command / params.hover_thrust_command * params.weight_n


def firmware_hover_command(params: CrazyflieParams) -> float:
    """Hover command straight from the firmware's thrust line: weight / (4 · THRUST_MAX per motor) · 65535."""
    return params.weight_n / (4 * params.firmware_motor_thrust_max_n) * THRUST_COMMAND_FULL_SCALE


def setpoint_to_firmware(
    thrust_n: float, roll_rad: float, pitch_rad: float, yaw_rate_radps: float, params: CrazyflieParams
) -> FirmwareSetpoint:
    """Our setpoint (the output of `attitude_action_to_setpoint`) -> cflib `send_setpoint` numbers.

        roll_deg = +degrees(roll)    pitch_deg = +degrees(pitch)    yawrate_deg = −degrees(yaw_rate)
        thrust = thrust_newtons_to_command(thrust_n)

    Roll and pitch are clamped to ±30° and yaw rate to ±180 °/s (the action's own range: a second fence).
    A NaN angle becomes 0 (level).
    """

    def clamp(value: float, limit: float) -> float:
        return 0.0 if not math.isfinite(value) else max(-limit, min(limit, value))

    return FirmwareSetpoint(
        roll_deg=math.degrees(clamp(roll_rad, MAX_TILT_RAD)),
        pitch_deg=math.degrees(clamp(pitch_rad, MAX_TILT_RAD)),
        yawrate_deg=-math.degrees(clamp(yaw_rate_radps, MAX_YAW_RATE_RADPS)),
        thrust=thrust_newtons_to_command(thrust_n, params),
    )
