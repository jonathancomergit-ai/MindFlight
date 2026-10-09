"""What every Crazyflie link can do, real or fake. The runner only talks to this.

Two links share this API:

| Link | File | What is on the other end |
|---|---|---|
| `SitlCrazyflie` | `sitl.py` | our simulator, dressed up as the firmware (radio delay, watchdog, unlock rule, noise) |
| `RealCrazyflie` | `crazyflie.py` | a real Crazyflie over a Crazyradio, through cflib |

A subclass fills in the radio part: `connect`, `close`, `send_setpoint`, `send_stop_setpoint`, `now`, `sleep_until`,
and calls `_on_log_block` whenever a log packet arrives. Everything above that (putting log blocks together into one
estimate, unlocking, motors off) is written once, here, so the exact code that will fly is the code SITL tests.
"""

import threading
import time

from mindflight.deploy.interface import LOG_BLOCKS, MOTORS_OFF, Estimate, FirmwareSetpoint, estimate_from_log


class CrazyflieLink:
    """The shared part of a link. Log blocks may arrive on another thread (cflib), so they are kept under a lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._blocks: dict[str, tuple[float, int, dict[str, float]]] = {}  # name -> (PC time, firmware ms, values)
        self.lost = False  # set when the radio link is known to be gone (cflib's connection_lost)

    # ---------- the radio part: each link fills these in ----------

    def connect(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def send_setpoint(self, roll_deg: float, pitch_deg: float, yawrate_deg: float, thrust: int) -> None:
        """cflib `Commander.send_setpoint`: roll and pitch in degrees, yaw rate in deg/s (+ = clockwise), thrust
        0..65535. Use `send` with a FirmwareSetpoint from interface.py instead of calling this by hand."""
        raise NotImplementedError

    def send_stop_setpoint(self) -> None:
        """cflib `Commander.send_stop_setpoint`: the firmware replaces the setpoint with all zeros: motors stop."""
        raise NotImplementedError

    def now(self) -> float:
        """Seconds on this link's clock (wall clock for the real drone, simulated time for SITL)."""
        raise NotImplementedError

    def sleep_until(self, time_s: float) -> None:
        """Wait until `now()` reaches time_s. SITL runs its simulation forward instead of waiting."""
        raise NotImplementedError

    def reset_estimator(self) -> None:
        """Ask the drone to restart its position estimate (real: Kalman reset). Default: nothing to do."""

    # ---------- shared ----------

    def _on_log_block(self, name: str, firmware_ms: int, values: dict[str, float]) -> None:
        """Called by the subclass for every log packet that arrives: remember it with the time it came in."""
        with self._lock:
            self._blocks[name] = (self.now(), firmware_ms, dict(values))

    def latest_estimate(self) -> Estimate | None:
        """The newest value of every logged variable, as one Estimate. None until every block has arrived once."""
        with self._lock:
            if set(self._blocks) != set(LOG_BLOCKS):
                return None
            values: dict[str, float] = {}
            for _, _, block_values in self._blocks.values():
                values.update(block_values)
            arrived = [t for t, _, _ in self._blocks.values()]
            newest_ms = max(ms for _, ms, _ in self._blocks.values())
        return estimate_from_log(values, time_s=max(arrived), oldest_s=min(arrived), firmware_ms=newest_ms)

    def wait_for_estimate(self, timeout_s: float = 3.0) -> Estimate | None:
        """Wait (on the link's clock) until a full estimate is in, up to timeout_s. None if it never came."""
        deadline = self.now() + timeout_s
        while self.now() < deadline:
            estimate = self.latest_estimate()
            if estimate is not None:
                return estimate
            self.sleep_until(self.now() + 0.02)
        return self.latest_estimate()

    def send(self, setpoint: FirmwareSetpoint) -> None:
        self.send_setpoint(setpoint.roll_deg, setpoint.pitch_deg, setpoint.yawrate_deg, setpoint.thrust)

    def unlock(self, seconds: float = 0.3, rate_hz: float = 50.0) -> None:
        """The firmware ignores thrust until it has seen one setpoint with thrust exactly 0 ("thrust lock",
        crtp_commander_rpyt.c). Send zero setpoints for `seconds`: motors stay off, and the drone is then ready."""
        end = self.now() + seconds
        while self.now() < end:
            self.send(MOTORS_OFF)
            self.sleep_until(self.now() + 1.0 / rate_hz)

    def motors_off(self) -> None:
        """Stop the motors now: a thrust-0 setpoint and a stop setpoint (either one alone stops them)."""
        self.send(MOTORS_OFF)
        self.send_stop_setpoint()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class WallClock:
    """Real time for real links: time.monotonic and time.sleep."""

    @staticmethod
    def now() -> float:
        return time.monotonic()

    @staticmethod
    def sleep_until(time_s: float) -> None:
        delay = time_s - time.monotonic()
        if delay > 0:
            time.sleep(delay)
