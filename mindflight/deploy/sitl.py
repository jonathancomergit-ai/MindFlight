"""A fake Crazyflie for software-in-the-loop (SITL) tests: our simulator, dressed up as the real drone's firmware.

The runner can't tell it from the real thing: same API (`CrazyflieLink`), same log variables, same setpoint numbers.
Inside, 500 times a simulated second:

```
 runner ──send_setpoint──▶ [uplink delay 5–20 ms] ──▶ firmware stand-in ──▶ our onboard controller ──▶ physics
 runner ◀── log blocks ◀── [downlink delay 5–20 ms] ◀── noisy estimate every 10 ms (100 Hz) ◀──────────┘
```

What the firmware stand-in copies from the real firmware (bitcraze/crazyflie-firmware, master, read 2026-10-09):

| Rule | Real firmware | Here |
|---|---|---|
| Packet | cflib packs `<fffH`: roll, **−pitch**, yawrate, thrust | `send_setpoint` packs the same bytes |
| Thrust lock | `crtp_commander_rpyt.c`: thrust is ignored until a packet with thrust 0 arrives after boot | `_receive_rpyt` |
| Too little thrust | below MIN_THRUST = THRUST_MIN / THRUST_MAX · 65535 ≈ 7000 the setpoint thrust is 0 | same |
| Yaw rate | `attitudeRate.yaw = -values->yaw` ("legacy rate input is inverted") | same |
| Pitch | setpoint pitch is "legacy": + = nose up (`kalman_core.c` stores `-pitch`) | same, then flipped to ours |
| Thrust → newtons | `motors.c`: per motor = command / 65535 · 0.12 N, capped by what the battery can give | same (`thrust_scale` can make the drone stronger or weaker than the model, to test calibration) |
| Watchdog | `supervisor.c`: no setpoint for 0.5 s while flying -> level out (roll = pitch = 0, yaw rate 0, keep thrust); for 2 s -> motors off and **locked** (reboot needed) | same |
| Tumble | tilted past 60° for 1 s (or upside down 0.1 s) while flying -> crashed, motors off | same |
| Stop packet | generic commander `stopType`: setpoint all zeros | same |
| Thrust 0 | `controller_pid.c`: thrust 0 -> no motor output, controller reset | same |

What it does NOT copy: the firmware's PID gains (we use our own controller, `physics/controller.py`), and the
firmware turns a yaw rate into a moving yaw-angle target while ours tracks the rate directly.

Estimator noise is simple and local on purpose (Gaussian on position, velocity, attitude, gyro). Fuller sensor
profiles belong to the training side (Phase 2 · P3).
"""

import heapq
import math
import struct
import time
from dataclasses import dataclass

import numpy as np
import torch

from mindflight.deploy.interface import LOG_BLOCKS, LOG_PERIOD_MS, THRUST_COMMAND_FULL_SCALE
from mindflight.deploy.link import CrazyflieLink
from mindflight.physics.controller import GAINS, controller_step
from mindflight.physics.dynamics import OMEGA, PHYSICS_DT_S, POS, QUAT, VEL, make_state, step
from mindflight.physics.params import CrazyflieParams, load_drone
from mindflight.physics.rotations import quat_from_axis_angle, quat_multiply, quat_to_euler, quat_to_matrix

RPYT_FORMAT = "<fffH"  # cflib Commander.send_setpoint: struct.pack('<fffH', roll, -pitch, yawrate, thrust)
WATCHDOG_LEVEL_S = 0.5  # supervisor.c COMMANDER_WDT_TIMEOUT_STABILIZE
WATCHDOG_CUT_S = 2.0  # supervisor.c COMMANDER_WDT_TIMEOUT_SHUTDOWN
IS_FLYING_HOLD_S = 2.0  # supervisor.c IS_FLYING_HYSTERESIS_THRESHOLD: "flying" until motors were off this long
TUMBLE_TILT_RAD = math.radians(60)  # SUPERVISOR_TUMBLE_CHECK_ACCEPTED_TILT_ACCZ 0.5 = cos(60°)
TUMBLE_TILT_S = 1.0  # ..._ACCEPTED_TILT_TIME 1000 ms
TUMBLE_UPSIDE_DOWN_S = 0.1  # ..._ACCEPTED_UPSIDEDOWN_TIME 100 ms (acc z < −0.2 g)
PRIORITY_DISABLE, PRIORITY_CRTP = 0, 2  # commander.h: priority is DISABLE from boot until the first setpoint


@dataclass
class SitlConfig:
    """Knobs of the fake drone. Numbers marked ASSUMED are guesses that look like a real Crazyflie, not measured."""

    uplink_delay_s: tuple[float, float] = (0.005, 0.020)  # ASSUMED: radio one-way delay, uniform in this range
    downlink_delay_s: tuple[float, float] = (0.005, 0.020)  # ASSUMED
    position_noise_m: float = 0.005  # ASSUMED: estimate noise (standard deviation), Lighthouse-like
    velocity_noise_mps: float = 0.02  # ASSUMED
    attitude_noise_rad: float = math.radians(0.3)  # ASSUMED: per axis, as a small random rotation
    gyro_noise_radps: float = math.radians(0.5)  # ASSUMED
    battery_noise_v: float = 0.005  # ASSUMED
    battery_full_v: float = 4.15  # open-circuit volts of a charged 1-cell LiPo (4.2 V right off the charger)
    battery_start_v: float | None = None  # open-circuit volts at the start; None = full
    battery_empty_v: float = 3.0  # open-circuit volts when empty (firmware: 3.0 V is "critical")
    battery_hover_s: float = 420.0  # ASSUMED: full -> empty at hover thrust (Bitcraze: about 7 min of flight)
    battery_sag_v: float = 0.25  # ASSUMED: voltage drop under hover load (more thrust, more drop)
    thrust_scale: float = 1.0  # real thrust / the firmware's model (e.g. 0.9 = props worn, drone 10 % weaker)
    start_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)  # on the floor
    start_yaw_rad: float = 0.0
    seed: int = 0
    realtime: bool = False  # True: sleep_until also waits on the wall clock, so you can watch it
    clock_start_s: float = 1000.0  # what now() says at connect. Not 0, like a real clock, so code can't assume 0


@dataclass
class FirmwareSetpoint:
    """The setpoint inside the firmware, in the firmware's own units: degrees, legacy pitch (+ = nose up),
    yaw rate deg/s counter-clockwise, thrust integer (0 = motors off). timestamp_s: when it arrived."""

    roll_deg: float = 0.0
    pitch_legacy_deg: float = 0.0
    yaw_rate_degps: float = 0.0
    thrust: int = 0
    timestamp_s: float = 0.0


class SitlCrazyflie(CrazyflieLink):
    """A simulated Crazyflie with the same API as `RealCrazyflie`. Time is simulated: `sleep_until(t)` runs the
    simulation up to t (fast, repeatable), unless config.realtime is set."""

    def __init__(self, drone: "str | CrazyflieParams | None" = None, config: SitlConfig | None = None):
        super().__init__()
        self.params = load_drone(drone)
        self.config = config or SitlConfig()
        self.connected = False
        self._reset()

    # ---------- setup ----------

    def _reset(self) -> None:
        c = self.config
        self.rng = np.random.default_rng(c.seed)
        self.ticks = 0  # physics steps since connect; time = start + ticks · 2 ms (an integer count: no drift)
        state = make_state(1, "cpu", pos=list(c.start_position_m))
        up = torch.tensor([[0.0, 0.0, 1.0]])
        state[:, QUAT] = quat_from_axis_angle(up, torch.tensor([c.start_yaw_rad]))
        self.state = state
        self.integral = torch.zeros(1, 3)
        self.rpm_command = torch.zeros(1, 4)
        self.thrust_n = 0.0  # total thrust the firmware is asking the motors for right now (N)
        # Firmware memory.
        self.setpoint = FirmwareSetpoint()
        self.priority = PRIORITY_DISABLE
        self.thrust_locked = True
        self.supervisor = "ready"  # ready | flying | level_out | locked | crashed
        self._last_thrust_s: float | None = None
        self._tilted_since_s: float | None = None
        self._upside_down_since_s: float | None = None
        # Battery.
        self.battery_open_v = c.battery_start_v if c.battery_start_v is not None else c.battery_full_v
        self.battery_v = self.battery_open_v
        # Radio.
        self.link_up = True
        self._uplink: list[tuple[float, int, str, bytes]] = []  # heap of (arrival time, order, kind, data)
        self._downlink: list[tuple[float, int, str, int, dict]] = []
        self._order = 0
        self._last_up_arrival = -math.inf
        self._last_down_arrival = -math.inf
        self.received: list[tuple[float, int]] = []  # (arrival time, thrust after the lock rule) per setpoint packet
        self.events: list[tuple[float, str]] = []  # supervisor changes, for tests and curious people
        self._wall_start = time.monotonic()

    def connect(self) -> None:
        self._reset()
        self.connected = True

    def close(self) -> None:
        self.connected = False

    # ---------- the link API ----------

    def now(self) -> float:
        return self.config.clock_start_s + self.ticks * PHYSICS_DT_S

    def sleep_until(self, time_s: float) -> None:
        self.advance_to(time_s)
        if self.config.realtime:
            delay = (self._wall_start + time_s - self.config.clock_start_s) - time.monotonic()
            if delay > 0:
                time.sleep(delay)

    def send_setpoint(self, roll_deg: float, pitch_deg: float, yawrate_deg: float, thrust: int) -> None:
        """Exactly what cflib does: check the thrust range, then pack roll, −pitch, yawrate, thrust."""
        if thrust > 0xFFFF or thrust < 0:
            raise ValueError("Thrust must be between 0 and 0xFFFF")
        self._send_packet("rpyt", struct.pack(RPYT_FORMAT, roll_deg, -pitch_deg, yawrate_deg, int(thrust)))

    def send_stop_setpoint(self) -> None:
        self._send_packet("stop", b"\x00")

    # ---------- test helpers ----------

    def advance(self, seconds: float) -> None:
        """Let the simulation run on by itself (nobody sending) for `seconds`."""
        self.advance_to(self.now() + seconds)

    def set_link(self, up: bool) -> None:
        """Cut (False) or restore (True) the radio. While cut, packets both ways are lost."""
        self.link_up = up

    def kick(self, delta_velocity_mps: tuple[float, float, float]) -> None:
        """A sudden push (a gust, a bump): add this to the drone's velocity (m/s, world)."""
        self.state[:, VEL] += torch.tensor([delta_velocity_mps], dtype=torch.float32)

    @property
    def position_m(self) -> tuple[float, float, float]:
        return tuple(self.state[0, POS].tolist())

    @property
    def euler_rad(self) -> tuple[float, float, float]:
        """True roll, pitch, yaw (rad, ours: + roll = right side down, + pitch = nose down, + yaw = CCW)."""
        return tuple(quat_to_euler(self.state[:, QUAT])[0].tolist())

    @property
    def motors_on(self) -> bool:
        """True if the firmware is driving the motors (any motor command above zero)."""
        return bool((self.rpm_command > 0).any())

    @property
    def true_hover_command(self) -> float:
        """The thrust integer that really holds this fake drone up (what calibrate-hover should find)."""
        per_motor_n = self.params.weight_n / 4 / self.config.thrust_scale
        return per_motor_n / self.params.firmware_motor_thrust_max_n * THRUST_COMMAND_FULL_SCALE

    # ---------- radio ----------

    def _send_packet(self, kind: str, data: bytes) -> None:
        if not self.link_up:
            return
        delay = self.rng.uniform(*self.config.uplink_delay_s)
        arrival = max(self.now() + delay, self._last_up_arrival)  # the radio keeps packets in order
        self._last_up_arrival = arrival
        self._order += 1
        heapq.heappush(self._uplink, (arrival, self._order, kind, data))

    def _deliver_uplink(self) -> None:
        while self._uplink and self._uplink[0][0] <= self.now() + 1e-9:
            _, _, kind, data = heapq.heappop(self._uplink)
            if self.supervisor in ("locked", "crashed"):
                continue  # crtpCommanderBlock: setpoints are not taken while the drone can't fly
            if kind == "rpyt":
                self._receive_rpyt(data)
            else:  # stop: crtp_commander_generic.c memsets the setpoint to zero
                self.setpoint = FirmwareSetpoint(timestamp_s=self.now())
                self.priority = PRIORITY_CRTP
                self.received.append((self.now(), 0))

    def _receive_rpyt(self, data: bytes) -> None:
        """crtpCommanderRpytDecodeSetpoint (crtp_commander_rpyt.c), default modes: roll/pitch angle, yaw rate."""
        roll, pitch, yaw, thrust = struct.unpack(RPYT_FORMAT, data)
        if self.priority == PRIORITY_DISABLE:
            self.thrust_locked = True
        if thrust == 0:
            self.thrust_locked = False
        min_command = self.params.firmware_motor_thrust_min_n / self.params.firmware_motor_thrust_max_n * 65535
        thrust_command = 0 if (self.thrust_locked or thrust < int(min_command)) else thrust
        self.setpoint = FirmwareSetpoint(
            roll_deg=roll,
            pitch_legacy_deg=pitch,  # what cflib put on the radio: −(cflib pitch). Legacy: + = nose up.
            yaw_rate_degps=-yaw,  # "legacy rate input is inverted"
            thrust=thrust_command,
            timestamp_s=self.now(),
        )
        self.priority = PRIORITY_CRTP
        self.received.append((self.now(), thrust_command))

    def _send_logs(self) -> None:
        """One estimate sample, sent as the three log blocks of interface.LOG_BLOCKS, each with its own delay."""
        if not self.link_up:
            return
        c, rng = self.config, self.rng
        s = self.state[0]
        position = s[POS].numpy() + rng.normal(0.0, c.position_noise_m, 3)
        velocity = s[VEL].numpy() + rng.normal(0.0, c.velocity_noise_mps, 3)
        wobble = torch.tensor(rng.normal(0.0, c.attitude_noise_rad, 3), dtype=torch.float32)[None]
        angle = wobble.norm(dim=1)
        noisy_turn = quat_from_axis_angle(wobble if angle > 0 else torch.tensor([[0.0, 0.0, 1.0]]), angle)
        quat = quat_multiply(self.state[:, QUAT], noisy_turn)[0].numpy()
        gyro_degps = np.degrees(s[OMEGA].numpy() + rng.normal(0.0, c.gyro_noise_radps, 3))
        values = {
            "stateEstimate.x": position[0], "stateEstimate.y": position[1], "stateEstimate.z": position[2],
            "stateEstimate.vx": velocity[0], "stateEstimate.vy": velocity[1], "stateEstimate.vz": velocity[2],
            "stateEstimate.qw": quat[0], "stateEstimate.qx": quat[1],
            "stateEstimate.qy": quat[2], "stateEstimate.qz": quat[3],
            "gyro.x": gyro_degps[0], "gyro.y": gyro_degps[1], "gyro.z": gyro_degps[2],
            "pm.vbat": self.battery_v + rng.normal(0.0, c.battery_noise_v),
        }
        firmware_ms = round(self.ticks * PHYSICS_DT_S * 1000)  # the drone's own clock: ms since it booted
        for name, variables in LOG_BLOCKS.items():
            arrival = max(self.now() + rng.uniform(*c.downlink_delay_s), self._last_down_arrival)
            self._last_down_arrival = arrival
            self._order += 1
            block = {v: float(values[v]) for v in variables}
            heapq.heappush(self._downlink, (arrival, self._order, name, firmware_ms, block))

    def _deliver_downlink(self) -> None:
        while self._downlink and self._downlink[0][0] <= self.now() + 1e-9:
            _, _, name, firmware_ms, block = heapq.heappop(self._downlink)
            self._on_log_block(name, firmware_ms, block)

    # ---------- firmware ----------

    def _set_supervisor(self, new_state: str) -> None:
        if new_state != self.supervisor:
            self.events.append((self.now(), new_state))
            self.supervisor = new_state

    def _supervised_setpoint(self) -> FirmwareSetpoint:
        """supervisor.c: the watchdog and the tumble check, applied to the current setpoint."""
        now = self.now()
        if self.supervisor in ("locked", "crashed"):
            return FirmwareSetpoint(timestamp_s=now)
        if self.setpoint.thrust > 0:
            self._last_thrust_s = now
        flying = self._last_thrust_s is not None and now - self._last_thrust_s < IS_FLYING_HOLD_S
        age = now - self.setpoint.timestamp_s
        if flying and age > WATCHDOG_CUT_S:
            self._set_supervisor("locked")  # ExceptFreeFall -> Locked: "reboot required"
            return FirmwareSetpoint(timestamp_s=now)
        if flying and self._tumbled(now):
            self._set_supervisor("crashed")
            return FirmwareSetpoint(timestamp_s=now)
        if flying and age > WATCHDOG_LEVEL_S:
            self._set_supervisor("level_out")
            sp = self.setpoint
            return FirmwareSetpoint(0.0, 0.0, 0.0, sp.thrust, sp.timestamp_s)  # level, keep the thrust
        self._set_supervisor("flying" if flying else "ready")
        return self.setpoint

    def _tumbled(self, now: float) -> bool:
        r_zz = float(quat_to_matrix(self.state[:, QUAT])[0, 2, 2])
        tilted = r_zz < math.cos(TUMBLE_TILT_RAD)
        upside_down = r_zz < -0.2
        self._tilted_since_s = (self._tilted_since_s or now) if tilted else None
        self._upside_down_since_s = (self._upside_down_since_s or now) if upside_down else None
        return (tilted and now - self._tilted_since_s > TUMBLE_TILT_S) or (
            upside_down and now - self._upside_down_since_s > TUMBLE_UPSIDE_DOWN_S
        )

    def _motor_volts_to_thrust_n(self, volts: float) -> float:
        p = self.params
        return (
            p.motor_volts_to_thrust_c0
            + p.motor_volts_to_thrust_c1 * volts
            + p.motor_volts_to_thrust_c2 * volts**2
            + p.motor_volts_to_thrust_c3 * volts**3
        )

    def _setpoint_in_our_units(self, sp: FirmwareSetpoint) -> tuple[float, float, float, float]:
        """Firmware setpoint -> (total thrust N, roll rad, pitch rad, yaw rate rad/s) in OUR conventions.

        thrust per motor = command / 65535 · THRUST_MAX, but no more than the battery can push through the motor
        at full duty (VMOTOR2THRUST polynomial at the loaded battery voltage); times config.thrust_scale.
        roll: same sign. pitch: legacy (+ = nose up) -> ours (+ = nose down) is a minus. yaw rate: already CCW.
        """
        p = self.params
        per_motor_n = sp.thrust / THRUST_COMMAND_FULL_SCALE * p.firmware_motor_thrust_max_n
        if per_motor_n < p.firmware_motor_thrust_min_n:
            per_motor_n = 0.0  # motorsCompensateBatteryVoltage returns 0 below THRUST_MIN
        per_motor_n = min(per_motor_n, max(self._motor_volts_to_thrust_n(self.battery_v), 0.0))
        thrust_n = 4 * per_motor_n * self.config.thrust_scale
        return thrust_n, math.radians(sp.roll_deg), -math.radians(sp.pitch_legacy_deg), math.radians(sp.yaw_rate_degps)

    def _update_battery(self, dt: float) -> None:
        """Open-circuit volts fall in a straight line with charge used; volts under load sag with thrust:
            d(open_v)/dt = −(full − empty) · load / hover_s,   battery_v = open_v − sag · load,   load = T / weight
        """
        c = self.config
        load = self.thrust_n / self.params.weight_n
        self.battery_open_v = max(
            c.battery_empty_v - 0.3, self.battery_open_v - (c.battery_full_v - c.battery_empty_v) * load * dt / c.battery_hover_s
        )
        self.battery_v = self.battery_open_v - c.battery_sag_v * load

    @torch.no_grad()
    def _tick(self) -> None:
        """One 2 ms step of the drone: radio in, firmware, controller, physics, battery, logs out."""
        self._deliver_uplink()
        thrust_n, roll, pitch, yaw_rate = self._setpoint_in_our_units(self._supervised_setpoint())
        self.thrust_n = thrust_n
        if thrust_n <= 0.0:
            self.rpm_command = torch.zeros(1, 4)
            self.integral = torch.zeros(1, 3)  # controller_pid.c: thrust 0 -> controllerPidReinitialize
        else:
            setpoint = torch.tensor([[thrust_n, roll, pitch, yaw_rate]], dtype=torch.float32)
            self.rpm_command, self.integral = controller_step(
                self.state, setpoint, self.integral, PHYSICS_DT_S, self.params, GAINS
            )
        self.state = step(self.state, self.rpm_command, PHYSICS_DT_S, self.params)
        self._update_battery(PHYSICS_DT_S)
        self.ticks += 1
        if self.ticks % round(LOG_PERIOD_MS / 1000 / PHYSICS_DT_S) == 0:
            self._send_logs()
        self._deliver_downlink()

    def advance_to(self, time_s: float) -> None:
        """Run the simulation until now() reaches time_s (whole 2 ms steps)."""
        while self.now() + 0.5 * PHYSICS_DT_S < time_s:
            self._tick()
