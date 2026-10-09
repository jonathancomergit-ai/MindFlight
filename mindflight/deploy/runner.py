"""The offboard flight loop: the trained brain flies a Crazyflie (real or SITL), 50 times a second.

Every 20 ms:

| Step | What happens |
|---|---|
| 1. read | the newest state estimate the drone logged (position, velocity, attitude, gyro, battery) |
| 2. safety | kill key? link lost? NaN? tipped over? outside the fence? battery low? |
| 3. mission | where should the drone be now (takeoff ramp, hover, next waypoint, landing ramp)? |
| 4. observe | build the observation EXACTLY like training: a "shadow" copy of the training env gets the estimate and the target, then its own `observe()` runs |
| 5. brain | policy(observation) -> action in [−1, 1]⁴ |
| 6. setpoint | `attitude_action_to_setpoint` (the same function training used) -> newtons and radians |
| 7. send | interface.py turns that into cflib's degrees and thrust integer; the link sends it |

The shadow env is the trick that keeps deploy correct when observations change: we never copy the observation code,
we call it. Its simulated sensor noise and delays are switched off (sensors="ideal", real_world=False), because the
real estimate already has real noise and real delay.

Safety stops:

| Trigger | Action |
|---|---|
| kill key (k) or Ctrl+C | motors off at once (the drone drops: that is the point of a kill switch) |
| estimate older than 0.3 s, or link lost | motors off (best effort); the drone's own watchdog levels it at 0.5 s and cuts the motors at 2 s |
| NaN in the estimate or the action, or tilt past 55 deg | motors off |
| outside the geofence box | land, inside the box (more than 0.5 m outside: motors off) |
| battery below `low_battery_v` for 1 s | land |

Every flight is written to runs/flights/<date-time>.jsonl: one line per step, plus events and the result.
"""

import json
import math
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from mindflight.deploy.interface import Estimate, setpoint_to_firmware
from mindflight.deploy.link import CrazyflieLink
from mindflight.deploy.policy import FlightPolicy
from mindflight.envs import make_env
from mindflight.envs.base import POLICY_DT_S
from mindflight.physics.controller import attitude_action_to_setpoint
from mindflight.physics.dynamics import OMEGA, POS, QUAT, VEL
from mindflight.physics.params import CrazyflieParams, load_drone

# ---------- the mission ----------


@dataclass
class Fence:
    """The box the drone must stay in (m, world frame). The floor is the bottom."""

    x_m: tuple[float, float] = (-1.5, 1.5)
    y_m: tuple[float, float] = (-1.5, 1.5)
    z_max_m: float = 2.0

    def distance_outside(self, position: tuple[float, float, float]) -> float:
        """How far outside the box the point is (m); 0 inside."""
        x, y, z = position
        dx = max(self.x_m[0] - x, 0.0, x - self.x_m[1])
        dy = max(self.y_m[0] - y, 0.0, y - self.y_m[1])
        dz = max(0.0, z - self.z_max_m)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def clamp(self, position: tuple[float, float, float], margin_m: float = 0.0) -> tuple[float, float, float]:
        """The nearest point at least margin_m inside the box."""
        x, y, z = position
        return (
            min(max(x, self.x_m[0] + margin_m), self.x_m[1] - margin_m),
            min(max(y, self.y_m[0] + margin_m), self.y_m[1] - margin_m),
            min(z, self.z_max_m - margin_m),
        )


@dataclass
class Mission:
    """Take off straight up, hover, visit the waypoints in order, land where you are. Distances in m, times in s."""

    takeoff_height_m: float = 0.5
    climb_speed_mps: float = 0.3  # the target rises this fast during takeoff
    hover_s: float = 3.0
    waypoints: list[tuple[float, float, float]] = field(default_factory=list)
    reach_radius_m: float = 0.10  # a waypoint is reached after settle_s in a row this close
    settle_s: float = 0.5
    waypoint_timeout_s: float = 10.0  # give up on a waypoint after this long (it is marked missed)
    descend_speed_mps: float = 0.3  # the target sinks this fast during landing
    touchdown_height_m: float = 0.06  # below this the motors go off
    land_timeout_s: float = 4.0  # extra time allowed after the landing ramp before motors off anyway
    max_target_step_m: float = 0.6  # the brain never sees a target further than this (keeps it in its training range)
    fence: Fence = field(default_factory=Fence)

    def check(self, edge_margin_m: float = 0.2) -> None:
        """Refuse a mission that would fly outside its own fence (raises ValueError)."""
        points = [(0.0, 0.0, self.takeoff_height_m)] + [tuple(w) for w in self.waypoints]
        for point in points:
            if self.fence.distance_outside(point) > 0 or self.fence.clamp(point, edge_margin_m) != tuple(point):
                raise ValueError(f"waypoint {point} is not {edge_margin_m} m inside the fence {self.fence}")
            if point[2] < 0.2:
                raise ValueError(f"waypoint {point} is lower than 0.2 m")

    @classmethod
    def from_json(cls, path: str | Path) -> "Mission":
        """A mission file: {"takeoff_height_m": 0.5, "waypoints": [[x, y, z], ...], "fence": {"x_m": [-1, 1], ...}}.
        Any field above may be given; the rest keep their defaults."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        fence = Fence(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.pop("fence", {}).items()})
        waypoints = [tuple(w) for w in raw.pop("waypoints", [])]
        return cls(**raw, waypoints=waypoints, fence=fence)


@dataclass
class Safety:
    max_tilt_deg: float = 55.0  # the brain asks for at most 30 deg; past 55 deg something is badly wrong
    link_timeout_s: float = 0.3  # the estimate comes at 100 Hz; 0.3 s without one = link lost
    low_battery_s: float = 1.0  # battery must be low this long (not one noisy sample) before we land
    hard_fence_margin_m: float = 0.5  # this far outside the fence: motors off, not land
    max_start_height_m: float = 0.3  # must start on (or near) the floor
    max_start_tilt_deg: float = 15.0
    min_start_battery_v: float = 3.5  # resting volts before takeoff; below: charge it first


@dataclass
class FlightResult:
    outcome: str  # landed | killed | stopped | link_lost | refused
    reason: str
    duration_s: float = 0.0
    waypoints_reached: list[bool] = field(default_factory=list)
    waypoint_closest_m: list[float] = field(default_factory=list)
    hover_thrust_commands: list[int] = field(default_factory=list)  # steady hover samples, for calibrate-hover
    final_position_m: tuple[float, float, float] | None = None
    max_tilt_deg: float = 0.0
    log_path: str = ""


# ---------- the observation, built by the training code itself ----------


class ShadowEnv:
    """A 1-drone copy of the training env that never steps. We write the real estimate into its state and call its
    own `observe()`, so the observation matches training column for column, whatever the task puts in it."""

    def __init__(self, task: str, env_kwargs: dict, params: CrazyflieParams):
        kwargs = dict(env_kwargs)
        if "sensors" in kwargs:
            kwargs["sensors"] = "ideal"  # the estimate is already real and noisy: don't add simulated noise
        if "real_world" in kwargs:
            kwargs["real_world"] = False  # ...nor simulated delays: the radio has real ones
        kwargs["drone"] = params
        self.env = make_env(task, 1, device="cpu", seed=0, **kwargs)
        if self.env.action_mode != "attitude":
            raise ValueError(f"this brain uses action_mode={self.env.action_mode!r}; a real Crazyflie needs 'attitude'")
        if not hasattr(self.env, "target"):
            raise ValueError(f"task {task!r} has no single target position; deploy flies hover/waypoint-style tasks")
        self.env.reset()

    @torch.no_grad()
    def observe(self, estimate: Estimate, target_m: tuple[float, float, float], last_action: torch.Tensor) -> torch.Tensor:
        env = self.env
        q = torch.tensor(estimate.quaternion_wxyz, dtype=torch.float32)
        env.state[0, POS] = torch.tensor(estimate.position_m, dtype=torch.float32)
        env.state[0, QUAT] = q / q.norm().clamp_min(1e-9)
        env.state[0, VEL] = torch.tensor(estimate.velocity_mps, dtype=torch.float32)
        env.state[0, OMEGA] = torch.tensor(estimate.gyro_radps, dtype=torch.float32)
        env.target[0] = torch.tensor(target_m, dtype=torch.float32)
        env.action[0] = last_action[0]  # the env's observation includes the action taken last step
        return env.observe()


# ---------- flight log ----------


class FlightLog:
    """One JSON object per line: {"type": "start" | "tick" | "event" | "end", "t": seconds, ...}."""

    def __init__(self, folder: str | Path):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = folder / f"{stamp}.jsonl"
        n = 1
        while path.exists():
            path = folder / f"{stamp}-{n}.jsonl"
            n += 1
        self.path = path
        self._file = path.open("w", encoding="utf-8")

    def write(self, kind: str, t: float, **data) -> None:
        self._file.write(json.dumps({"type": kind, "t": round(t, 4), **data}, default=_rounded) + "\n")

    def close(self) -> None:
        self._file.close()


def _rounded(value):
    return str(value)


def _r(values, digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in values]


# ---------- the runner ----------


class FlightRunner:
    """Flies one mission with `policy` over `link`. `fly()` returns a FlightResult.

    params: the drone (thrust scale, hover command, battery limit); default: the brain's drone file.
    on_tick(runner, t, estimate): optional hook after every step (tests use it to pull the kill switch).
    """

    def __init__(
        self,
        link: CrazyflieLink,
        policy: FlightPolicy,
        mission: Mission | None = None,
        params: CrazyflieParams | None = None,
        safety: Safety | None = None,
        log_dir: str | Path = "runs/flights",
        print_fn: Callable[[str], None] = print,
        on_tick: Callable[["FlightRunner", float, Estimate], None] | None = None,
        status_every_s: float = 0.5,
    ):
        self.link = link
        self.policy = policy
        self.mission = mission or Mission()
        self.mission.check()
        self.params = params or load_drone(policy.env_kwargs.get("drone"))
        self.safety = safety or Safety()
        self.log_dir = log_dir
        self.print = print_fn
        self.on_tick = on_tick
        self.status_every_s = status_every_s
        self.shadow = ShadowEnv(policy.task, policy.env_kwargs, self.params)
        self._kill_reason: str | None = None
        self._send_lock = threading.Lock()

    # ---------- kill switch (any thread) ----------

    def request_kill(self, reason: str = "kill key") -> None:
        """Motors off NOW, from any thread. The loop then stops sending anything else."""
        with self._send_lock:
            if self._kill_reason is None:
                self._kill_reason = reason
            self.link.motors_off()

    # ---------- helpers ----------

    def _event(self, t: float, text: str) -> None:
        self.print(f">>> {t:6.2f} s  {text}")
        self.log.write("event", t, text=text)

    def _stop_motors(self, repeats: int = 5) -> None:
        """Motors off, sent a few times over 0.1 s in case a radio packet is lost."""
        for _ in range(repeats):
            with self._send_lock:
                self.link.motors_off()
            self.link.sleep_until(self.link.now() + 0.02)

    def _carrot(self, position, target) -> tuple[float, float, float]:
        """The target the brain sees: the real one, but no further than max_target_step_m from the drone."""
        offset = [t - p for t, p in zip(target, position)]
        length = math.sqrt(sum(o * o for o in offset))
        scale = min(1.0, self.mission.max_target_step_m / length) if length > 1e-9 else 1.0
        return tuple(p + o * scale for p, o in zip(position, offset))

    def _preflight(self, estimate: Estimate | None) -> str | None:
        """Why we must not take off, or None if all is well."""
        s = self.safety
        if estimate is None:
            return "no state estimate from the drone (is the positioning deck on and the estimator running?)"
        if not all(math.isfinite(v) for v in estimate.numbers()):
            return "the state estimate has NaN in it"
        if estimate.position_m[2] > s.max_start_height_m:
            return f"the drone is not on the floor (z = {estimate.position_m[2]:.2f} m)"
        if math.degrees(estimate.tilt_rad) > s.max_start_tilt_deg:
            return f"the drone is tilted {math.degrees(estimate.tilt_rad):.0f} deg (put it on flat ground)"
        if estimate.battery_v < s.min_start_battery_v:
            return f"battery {estimate.battery_v:.2f} V is below {s.min_start_battery_v} V: charge it"
        if self.mission.fence.distance_outside(estimate.position_m) > 0:
            return f"the drone starts outside the fence at {_r(estimate.position_m, 2)}"
        return None

    # ---------- the flight ----------

    def fly(self) -> FlightResult:
        self.log = FlightLog(self.log_dir)
        self.log.write(
            "start", self.link.now(), policy=self.policy.source, task=self.policy.task,
            env_kwargs=self.policy.env_kwargs, drone=asdict(self.params), mission=asdict(self.mission),
            safety=asdict(self.safety),
        )
        try:
            result = self._fly()
        except KeyboardInterrupt:
            self.request_kill("Ctrl+C")
            self._stop_motors()
            result = FlightResult("killed", "Ctrl+C")
        except BaseException:
            self._stop_motors()
            raise
        result.log_path = str(self.log.path)
        self.log.write("end", self.link.now(), **asdict(result))
        self.log.close()
        self.print(f"=== {result.outcome.upper()}: {result.reason}   (log: {result.log_path})")
        return result

    def _fly(self) -> FlightResult:
        link, mission, safety = self.link, self.mission, self.safety
        estimate = link.wait_for_estimate(timeout_s=3.0)
        refusal = self._preflight(estimate)
        if refusal is not None:
            self._event(link.now(), f"NOT TAKING OFF: {refusal}")
            return FlightResult("refused", refusal)

        home = estimate.position_m
        self.print(f"--- {self.policy.source} on {self.params.name}: start at {_r(home, 2)} m, battery {estimate.battery_v:.2f} V")
        link.unlock(seconds=0.3)  # thrust-0 packets: the firmware's thrust lock opens, motors stay off

        result = FlightResult("landed", "mission complete")
        result.waypoints_reached = [False] * len(mission.waypoints)
        result.waypoint_closest_m = [math.inf] * len(mission.waypoints)
        start_s = link.now()
        phase, phase_start_s = "takeoff", 0.0  # phase times count from start_s, like t
        phase_from_z = home[2]
        land_xy = home[:2]
        waypoint = 0
        close_since_s: float | None = None
        battery_filtered_v = estimate.battery_v
        low_battery_since_s: float | None = None
        last_action = torch.zeros(1, 4)
        last_status_s = -math.inf
        next_step_s = link.now()

        def start_landing(now: float, why: str, position) -> None:
            nonlocal phase, phase_start_s, phase_from_z, land_xy
            if phase == "land":
                return
            self._event(now, f"LANDING: {why}")
            if why != "mission complete":
                result.reason = why
            phase, phase_start_s, phase_from_z = "land", now, position[2]
            land_xy = mission.fence.clamp(position, margin_m=0.2)[:2]

        while True:
            now = link.now()
            t = now - start_s

            # ---- 1. kill switch ----
            if self._kill_reason is not None:
                self._event(t, f"KILL ({self._kill_reason}): motors off")
                self._stop_motors()
                result.outcome, result.reason = "killed", self._kill_reason
                break

            # ---- 2. read + safety ----
            estimate = link.latest_estimate()
            if link.lost or estimate is None or now - estimate.oldest_s > safety.link_timeout_s:
                age = math.inf if estimate is None else now - estimate.oldest_s
                self._event(t, f"LINK LOST (no estimate for {age:.2f} s): motors off; the drone's watchdog cuts at 2 s")
                self._stop_motors()
                result.outcome, result.reason = "link_lost", f"no estimate for {age:.2f} s"
                break
            position = estimate.position_m
            if not all(math.isfinite(v) for v in estimate.numbers()):
                self._event(t, "STOP: NaN in the state estimate: motors off")
                self._stop_motors()
                result.outcome, result.reason = "stopped", "NaN in the estimate"
                break
            tilt_deg = math.degrees(estimate.tilt_rad)
            result.max_tilt_deg = max(result.max_tilt_deg, tilt_deg)
            if tilt_deg > safety.max_tilt_deg:
                self._event(t, f"STOP: tilted {tilt_deg:.0f} deg (> {safety.max_tilt_deg:.0f} deg): motors off")
                self._stop_motors()
                result.outcome, result.reason = "stopped", f"tilt {tilt_deg:.0f} deg"
                break
            outside_m = mission.fence.distance_outside(position)
            if outside_m > safety.hard_fence_margin_m:
                self._event(t, f"STOP: {outside_m:.2f} m outside the fence: motors off")
                self._stop_motors()
                result.outcome, result.reason = "stopped", f"{outside_m:.2f} m outside the fence"
                break
            if outside_m > 0 and phase != "land":
                start_landing(t, f"geofence: {_r(position, 2)} is outside the box", position)
            battery_filtered_v += (estimate.battery_v - battery_filtered_v) * min(1.0, POLICY_DT_S / 0.5)
            if battery_filtered_v < self.params.low_battery_v:
                low_battery_since_s = t if low_battery_since_s is None else low_battery_since_s
                if t - low_battery_since_s >= safety.low_battery_s and phase != "land":
                    start_landing(t, f"low battery ({battery_filtered_v:.2f} V)", position)
            else:
                low_battery_since_s = None

            # ---- 3. mission: where should we be? ----
            in_phase_s = t - phase_start_s
            height = mission.takeoff_height_m
            if phase == "takeoff":
                target = (home[0], home[1], min(phase_from_z + mission.climb_speed_mps * in_phase_s, height))
                if target[2] >= height and abs(position[2] - height) < 0.1:
                    phase, phase_start_s = "hover", t
                    self._event(t, f"HOVER at {height} m")
                elif in_phase_s > height / mission.climb_speed_mps + 6.0:
                    start_landing(t, "takeoff took too long", position)
            elif phase == "hover":
                target = (home[0], home[1], height)
                if in_phase_s > 1.5 and abs(estimate.velocity_mps[2]) < 0.1 and abs(position[2] - height) < 0.15:
                    result.hover_thrust_commands.append(self._last_thrust)
                if in_phase_s >= mission.hover_s:
                    if mission.waypoints:
                        phase, phase_start_s, close_since_s = "waypoint", t, None
                        self._event(t, f"WAYPOINT 1/{len(mission.waypoints)} {mission.waypoints[0]}")
                    else:
                        start_landing(t, "mission complete", position)
            elif phase == "waypoint":
                goal = tuple(mission.waypoints[waypoint])
                target = goal
                distance = math.dist(position, goal)
                result.waypoint_closest_m[waypoint] = min(result.waypoint_closest_m[waypoint], distance)
                close_since_s = (close_since_s if close_since_s is not None else t) if distance < mission.reach_radius_m else None
                reached = close_since_s is not None and t - close_since_s >= mission.settle_s
                if reached or in_phase_s > mission.waypoint_timeout_s:
                    result.waypoints_reached[waypoint] = reached
                    self._event(t, f"waypoint {waypoint + 1} {'REACHED' if reached else 'MISSED (timeout)'}")
                    waypoint += 1
                    if waypoint == len(mission.waypoints):
                        start_landing(t, "mission complete", position)
                    else:
                        phase, phase_start_s, close_since_s = "waypoint", t, None
                        self._event(t, f"WAYPOINT {waypoint + 1}/{len(mission.waypoints)} {mission.waypoints[waypoint]}")
            if phase == "land":
                in_phase_s = t - phase_start_s
                target = (land_xy[0], land_xy[1], max(phase_from_z - mission.descend_speed_mps * in_phase_s, 0.0))
                ramp_s = phase_from_z / mission.descend_speed_mps
                if position[2] < mission.touchdown_height_m or in_phase_s > ramp_s + mission.land_timeout_s:
                    self._event(t, f"TOUCHDOWN at z = {position[2]:.3f} m: motors off")
                    self._stop_motors()
                    break

            # ---- 4-7. observe, brain, setpoint, send ----
            obs = self.shadow.observe(estimate, self._carrot(position, target), last_action)
            action = self.policy(obs)
            if not torch.isfinite(action).all():
                self._event(t, "STOP: the brain gave NaN: motors off")
                self._stop_motors()
                result.outcome, result.reason = "stopped", "NaN action"
                break
            thrust_n, roll, pitch, yaw_rate = attitude_action_to_setpoint(action, self.params)[0].tolist()
            command = setpoint_to_firmware(thrust_n, roll, pitch, yaw_rate, self.params)
            with self._send_lock:
                if self._kill_reason is not None:
                    continue  # killed while we were thinking: send nothing but motors off (top of the loop)
                link.send(command)
            self._last_thrust = command.thrust
            last_action = action

            self.log.write(
                "tick", t, phase=phase, pos=_r(position), vel=_r(estimate.velocity_mps),
                quat=_r(estimate.quaternion_wxyz), gyro=_r(estimate.gyro_radps), vbat=round(estimate.battery_v, 3),
                age=round(now - estimate.oldest_s, 4), target=_r(target), action=_r(action[0].tolist()),
                setpoint=_r((thrust_n, roll, pitch, yaw_rate)), command=_r(
                    (command.roll_deg, command.pitch_deg, command.yawrate_deg), 3) + [command.thrust],
            )
            if t - last_status_s >= self.status_every_s:
                last_status_s = t
                self.print(
                    f"{t:6.2f} s  {phase.upper():8s} pos {position[0]:5.2f} {position[1]:5.2f} {position[2]:5.2f} m"
                    f"  to target {math.dist(position, target):4.2f} m  tilt {tilt_deg:3.0f} deg"
                    f"  bat {estimate.battery_v:4.2f} V  thrust {command.thrust:5d}"
                )
            if self.on_tick is not None:
                self.on_tick(self, t, estimate)

            next_step_s += POLICY_DT_S
            if next_step_s < link.now():
                next_step_s = link.now()  # running late (slow PC): don't try to catch up with a burst
            link.sleep_until(next_step_s)

        result.duration_s = link.now() - start_s
        final = link.latest_estimate()
        result.final_position_m = tuple(_r(final.position_m)) if final is not None else None
        return result

    _last_thrust = 0


def hover_command_from(result: FlightResult) -> float | None:
    """The average thrust command while hovering steadily: that is what really holds the drone up."""
    samples = result.hover_thrust_commands
    return sum(samples) / len(samples) if len(samples) >= 25 else None


def calibrate_hover(runner_factory: Callable[[Mission], FlightRunner], height_m: float = 0.5, hover_s: float = 6.0):
    """Take off, hover `hover_s` seconds, land, and measure the hover thrust command.

    runner_factory(mission) -> a FlightRunner for that mission. Returns (hover command or None, FlightResult).
    While the drone hovers still (vertical speed < 0.1 m/s), thrust = weight, so the average command sent then is
    the drone's true hover command, even if the brain was given a wrong one: it simply pushed the throttle until
    the drone stopped sinking.
    """
    runner = runner_factory(Mission(takeoff_height_m=height_m, hover_s=hover_s))
    result = runner.fly()
    return hover_command_from(result), result


# ---------- keyboard kill switch ----------


class KeyboardKill:
    """Press k (or K, or space) to kill the motors. A background thread watches the keyboard; Ctrl+C also works
    (Python raises KeyboardInterrupt in the flight loop). Does nothing when the input is not a terminal."""

    KEYS = ("k", "K", " ")

    def __init__(self, runner: FlightRunner):
        self.runner = runner
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        if sys.stdin is not None and sys.stdin.isatty():
            self._thread = threading.Thread(target=self._watch, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()

    def _watch(self) -> None:
        if sys.platform == "win32":
            import msvcrt

            while not self._stop.is_set():
                if msvcrt.kbhit() and msvcrt.getwch() in self.KEYS:
                    self.runner.request_kill("kill key")
                time.sleep(0.005)
        else:
            import select
            import termios
            import tty

            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while not self._stop.is_set():
                    ready, _, _ = select.select([sys.stdin], [], [], 0.01)
                    if ready and sys.stdin.read(1) in self.KEYS:
                        self.runner.request_kill("kill key")
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
