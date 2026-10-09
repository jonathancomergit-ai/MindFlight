"""The real link: a Crazyflie over a Crazyradio, through Bitcraze's cflib.

Kept thin on purpose. It only moves bytes: setpoints out, log blocks in. Everything above it (log blocks -> one
estimate, unlock, motors off, the flight loop, the unit and sign conversions) is shared with the SITL fake and
tested there.

cflib pieces used (cflib 0.1.34):

| cflib | What for |
|---|---|
| `cflib.crtp.init_drivers()` | find the Crazyradio |
| `SyncCrazyflie(uri, cf=Crazyflie(rw_cache=...))`, `open_link()` | connect and download the variable lists |
| `LogConfig(name, period_in_ms=10)` + `add_variable(name, "float")` | the 3 log blocks of interface.LOG_BLOCKS, 100 Hz |
| `Commander.send_setpoint(roll, pitch, yawrate, thrust)` | the setpoint (client X-mode stays OFF: a 2.x is an X frame already) |
| `Commander.send_stop_setpoint()` | motors off |
| `supervisor.send_arming_request(True)` | arm, for firmware built to require it (stock 2.1+ auto-arms) |
| `param.set_value("kalman.resetEstimation", ...)` | restart the position estimate before takeoff |
| `connection_lost` callback | mark the link lost, so the runner stops at once |

This file cannot be tested without hardware. `tests/test_deploy_*.py` test everything that calls it.
"""

from pathlib import Path

from mindflight.deploy.interface import LOG_BLOCKS, LOG_PERIOD_MS
from mindflight.deploy.link import CrazyflieLink, WallClock

DEFAULT_URI = "radio://0/80/2M/E7E7E7E7E7"  # the factory address of a new Crazyflie


class RealCrazyflie(CrazyflieLink):
    """A real Crazyflie. uri: its radio address. arm: send an arming request after connecting."""

    def __init__(self, uri: str = DEFAULT_URI, arm: bool = True, cache_dir: str | Path = "runs/cflib_cache"):
        super().__init__()
        self.uri = uri
        self.arm = arm
        self.cache_dir = Path(cache_dir)
        self._scf = None
        self._log_configs = []

    def connect(self) -> None:
        try:
            import cflib.crtp
            from cflib.crazyflie import Crazyflie
            from cflib.crazyflie.log import LogConfig
            from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
        except ImportError as error:
            raise ImportError("the real drone needs cflib: pip install -e .[deploy]") from error

        cflib.crtp.init_drivers()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._scf = SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache=str(self.cache_dir)))
        self._scf.open_link()
        cf = self._scf.cf
        cf.connection_lost.add_callback(self._on_lost)
        cf.disconnected.add_callback(self._on_lost)
        cf.commander.set_client_xmode(False)
        for name, variables in LOG_BLOCKS.items():
            config = LogConfig(name=name, period_in_ms=LOG_PERIOD_MS)
            for variable in variables:
                config.add_variable(variable, "float")
            cf.log.add_config(config)  # raises if a block is too big or a variable is missing
            config.data_received_cb.add_callback(self._on_data)
            config.start()
            self._log_configs.append(config)
        if self.arm:
            supervisor = getattr(cf, "supervisor", None)
            if supervisor is not None:
                supervisor.send_arming_request(True)
            else:
                cf.platform.send_arming_request(True)

    def _on_data(self, timestamp_ms: int, data: dict, config) -> None:
        self._on_log_block(config.name, timestamp_ms, data)

    def _on_lost(self, uri: str, message: str = "") -> None:
        self.lost = True

    def reset_estimator(self, settle_s: float = 2.0) -> None:
        """Restart the Kalman filter at the current spot and give it settle_s to converge (Bitcraze examples do this)."""
        param = self._scf.cf.param
        param.set_value("kalman.resetEstimation", "1")
        self.sleep_until(self.now() + 0.1)
        param.set_value("kalman.resetEstimation", "0")
        self.sleep_until(self.now() + settle_s)

    def close(self) -> None:
        if self._scf is None:
            return
        try:
            self.motors_off()
            for config in self._log_configs:
                config.stop()
        finally:
            self._scf.close_link()
            self._scf = None

    def send_setpoint(self, roll_deg: float, pitch_deg: float, yawrate_deg: float, thrust: int) -> None:
        self._scf.cf.commander.send_setpoint(roll_deg, pitch_deg, yawrate_deg, int(thrust))

    def send_stop_setpoint(self) -> None:
        self._scf.cf.commander.send_stop_setpoint()

    def now(self) -> float:
        return WallClock.now()

    def sleep_until(self, time_s: float) -> None:
        WallClock.sleep_until(time_s)
