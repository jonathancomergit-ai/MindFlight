"""MindFlight Hub: every brain, the race, live training and the physics test in one page. `python -m mindflight.hub`
(or double-click `MindFlight Hub.bat`), and the browser opens by itself.

The page gets a sidebar with every show. Click one and the server swaps what it streams: no restarting, no
terminal commands. Each show is built the same way its own command builds it (`play`, `race`, `demo`,
`train --view`, `deploy --sitl`), so what you see here is exactly what those commands show.

How the swap works: `HubSource` is a `FrameSource` (see viewer/server.py) that hands every call on to the show
running now. A switch builds the new show, then swaps it in and bumps `generation`. The server sees the new
generation and sends the page everything again; the page sees it in the "hub" message and clears the old scene,
charts and leaderboard first.

Extra routes (besides the viewer's own):
- `POST /show/{id}`        switch to a show (the sidebar's buttons)
- `POST /speed?value=0.5`  sim seconds per real second for the show running now (and the ones after it)
"""

import argparse
import socket
import threading
import time
import webbrowser
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import HTTPException

from mindflight.demo import FourGroupDemo
from mindflight.envs.base import POLICY_DT_S
from mindflight.play import EpisodeTally
from mindflight.race import DEFAULT_ENTRANTS, Entrant, RaceSource
from mindflight.rl.ppo import PPO, PPOConfig
from mindflight.rl.train import RUNS_DIR, load_policy, train
from mindflight.viewer.__main__ import HOST
from mindflight.viewer.live import CHARTS
from mindflight.viewer.policy_source import SHOW, PolicySource, env_for_showing
from mindflight.viewer.server import create_app
from mindflight.viewer.sim_source import SimSource

POLICIES = Path("policies")
SPEEDS = (0.25, 0.5, 1.0, 2.0)
# The bigger shows need about 20 ms of CPU per brain step, and real time allows 20 ms: on the GPU they keep up.
# Follow-me is the exception (its person tracker is faster on the CPU).
SHOW_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CPU_TASKS = {"follow"}
# Most sim time one frame may run (s). One frame's worth (1/30 s) plus a little: if a show can't keep up, it runs a
# bit slow but stays smooth, instead of stuttering through big catch-up jumps.
MAX_CATCH_UP_S = 0.04
MISSION_WAYPOINTS = [(0.5, 0.0, 0.6), (0.5, 0.5, 0.8), (0.0, 0.5, 0.6)]


@dataclass(frozen=True)
class Show:
    """One entry in the sidebar."""

    id: str
    group: str
    title: str
    blurb: str
    score: str = ""
    policy: str | None = None  # a brain in policies/; the show is hidden if the file is missing


SHOWS = [
    Show("pack", "Phase 2 · the vision", "Packs of 6",
         "Four packs fly to new formations. Each drone picks its own slot, so paths never cross. Realistic sensors.",
         "97.3 % in formation", "real_pack.pt"),
    Show("intercept", "Phase 2 · the vision", "Intercept",
         "Guard the person (grey) and knock thrown foam balls out of the air.", "93.1 % caught", "intercept.pt"),
    Show("follow", "Phase 2 · the vision", "Follow-me",
         "Follow, orbit, or fly where the person points. The mode switches as they go. (Follow mode: 76 %,"
         " being improved.)",
         "orbit 97 % · point 98.5 %", "follow.pt"),
    Show("mission", "Ready for the real drone", "Fake Crazyflie",
         "A full mission (take off, 3 waypoints, land) through the exact code that will fly the real drone, with"
         " radio delay and sensor noise.", "real flight code", "real_waypoint.pt"),
    Show("real_waypoint", "Ready for the real drone", "Real-world waypoints",
         "Flow-deck sensors with drift and delay; every drone a bit heavier, weaker or off-balance.",
         "95.6 % ≤ 15 cm", "real_waypoint.pt"),
    Show("real_hover", "Ready for the real drone", "Real-world hover",
         "Hold 1 m on noisy, late Flow-deck readings, with a draining battery.", "96.4 % ≤ 15 cm",
         "real_hover.pt"),
    Show("race", "Phase 1 · skills", "Race", "Three brains, same course, one try each. Fastest lap wins.",
         "racer won 96 of 100"),
    Show("gates", "Phase 1 · skills", "Gates", "Fly 6 gates in order on a new course every round.",
         "95.8 %, 3.7 s laps", "gates.pt"),
    Show("obstacles", "Phase 1 · skills", "Pillar field", "Cross a pillar field seeing only by laser rangefinder.",
         "100 %", "obstacles.pt"),
    Show("sturdy", "Phase 1 · skills", "Sturdy", "Fly to points through gusts, with ±20 % mass.", "97.4 %",
         "sturdy.pt"),
    Show("waypoint", "Phase 1 · skills", "Fly to a point", "Reach a random point and settle within 10 cm.",
         "99.3 %", "waypoint.pt"),
    Show("hover", "Phase 1 · skills", "Hover", "Hold 1 m for 10 s from a random start.", "100 %", "hover.pt"),
    Show("train", "Learn", "Watch it learn: hover",
         "A brand-new brain learns to hover from scratch on the GPU. About a minute. Charts on the right."),
    Show("physics", "Learn", "Raw physics test",
         "No brain: fixed motor speeds. One group hovers, one falls, one tips over, one spins."),
]


@dataclass
class Built:
    """A show, ready to stream."""

    source: object  # a FrameSource
    groups: list[dict] = field(default_factory=list)  # legend: drone groups and their colours
    on_reset: Callable[[], None] | None = None
    note: str = ""  # shown next to the restart button
    stop: Callable[[], None] | None = None  # called when switching away (stops live training)


# ---------- the shows ----------


def build_policy_show(path: Path) -> Built:
    """What `python -m mindflight.play --policy <path>` shows."""
    task = load_policy(path, "cpu")[1]["task"]
    device = "cpu" if task in CPU_TASKS else SHOW_DEVICE
    model, checkpoint = load_policy(path, device)
    layout, n, spacing_m, round_s = SHOW.get(task, ("grid", 16, 4.0, None))
    env = env_for_showing(task, dict(checkpoint.get("env_kwargs", {})), n, device)
    source = PolicySource(env, model, layout=layout, spacing_m=spacing_m, round_s=round_s)
    header = [f"{task} · {path.name} (iteration {checkpoint.get('iteration', '?')})"]
    source.on_episode_end = EpisodeTally(source, header)
    return Built(source, on_reset=source.reset)


def build_race() -> Built:
    """What `python -m mindflight.race` shows."""
    entrants = [Entrant(name, path, note) for name, path, note in DEFAULT_ENTRANTS if Path(path).exists()]
    models = [load_policy(e.path, "cpu")[0] for e in entrants]
    source = RaceSource(entrants, models, drones_each=2)
    source.info_lines = ["Each brain flies 2 drones on the same course. One try per round. Fastest lap wins."]
    groups = [
        {"name": e.name, "motors": e.path, "expect": e.note, "first": 2 * i, "count": 2} for i, e in enumerate(entrants)
    ]
    return Built(source, groups=groups, on_reset=source.restart)


def build_physics() -> Built:
    """What `python -m mindflight.demo --n 64` shows."""
    demo = FourGroupDemo(64)
    source = SimSource(demo.start_state, demo.step, restart_after_s=3.0)
    return Built(source, groups=demo.config()["groups"], on_reset=source.reset, note="starts over every 3 s")


class _StopTraining(Exception):
    pass


def build_training(task: str = "hover", iterations: int = 300) -> Built:
    """What `python -m mindflight.train --task hover --view` shows. Training runs in a thread; switching away stops
    it at the end of the iteration it is on. Runs go to runs/hub/<task>-<date-time>/, never into policies/."""
    layout, n, spacing_m, round_s = SHOW.get(task, ("grid", 16, 4.0, None))
    env = env_for_showing(task, {}, n, "cpu")
    model = PPO(env.obs_size, env.action_size, PPOConfig(), "cpu").model.eval()  # the learner's shape, untrained
    source = PolicySource(env, model, layout=layout, spacing_m=spacing_m, round_s=round_s)
    source.chart_series = [c for c in CHARTS if c["key"] != "difficulty" or getattr(env, "has_curriculum", False)]
    title = f"learning {task} from scratch: the newest brain is copied in after every iteration"
    source.info_lines = [title, "starting up (the GPU warms up for a few seconds)..."]
    stop = threading.Event()

    def on_iteration(_env, metrics: dict, learner) -> None:
        if stop.is_set():
            raise _StopTraining
        source.set_policy({k: v.detach().to("cpu") for k, v in learner.state_dict().items()})
        point = {"iteration": metrics["iteration"]}
        for chart in source.chart_series:
            if chart["key"] in metrics:
                point[chart["key"]] = round(float(metrics[chart["key"]]), 4)
        source.metric_points.append(point)
        line = f"iteration {metrics['iteration']} of {iterations} · {metrics['samples'] / 1e6:.1f} M practice steps"
        if "episode_success" in metrics:
            line += f" · success {100 * metrics['episode_success']:.0f}%"
        source.info_lines = [title, line]

    def run() -> None:
        run_dir = RUNS_DIR / "hub" / f"{task}-{time.strftime('%Y%m%d-%H%M%S')}"
        try:
            train(task, iterations=iterations, run_dir=run_dir, on_iteration=on_iteration, log=lambda _: None)
            source.info_lines = source.info_lines + [f"done! saved in {run_dir}"]
        except _StopTraining:
            pass
        except Exception as error:  # show it in the page rather than dying quietly in a thread
            source.info_lines = source.info_lines + [f"training stopped: {error}"]

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    def stop_training() -> None:
        stop.set()
        thread.join(timeout=10)

    return Built(source, on_reset=source.reset, stop=stop_training, note="restart = new starting spots, same brain")


class ReplaySource:
    """Plays back a recorded flight in real time, then loops: poses (T, 7) at 50 Hz, plus timed text lines."""

    def __init__(self, poses: np.ndarray, lines: list[tuple[float, str]], scene: dict, header: list[str],
                 hold_s: float = 2.0, clock: Callable[[], float] = time.perf_counter):
        self.poses = poses.astype(np.float32)
        self.lines = lines
        self.scene_message = {"type": "scene", **scene}
        self.header = header
        self.hold_s = hold_s
        self.speed = 1.0
        self.n_drones = 1
        self._clock = clock
        self.reset()

    @property
    def duration_s(self) -> float:
        return len(self.poses) * POLICY_DT_S

    def reset(self) -> None:
        self.t = 0.0
        self._last = self._clock()

    def frame(self, time_s: float) -> np.ndarray:
        now = self._clock()
        self.t += min(max(now - self._last, 0.0), 0.1) * self.speed
        self._last = now
        if self.t > self.duration_s + self.hold_s:
            self.t = 0.0
        index = min(int(self.t / POLICY_DT_S), len(self.poses) - 1)
        return self.poses[index : index + 1].copy()

    def state_messages(self) -> list[dict]:
        said = [text for t, text in self.lines if t <= self.t]
        return [self.scene_message, {"type": "info", "lines": self.header + [f"t = {self.t:5.1f} s"] + said[-6:]}]


def record_mission(policy_path: Path = POLICIES / "real_waypoint.pt") -> tuple:
    """What `python -m mindflight.deploy --policy policies/real_waypoint.pt --sitl --waypoints ...` does, recorded.
    The fake Crazyflie runs as fast as it can, which is still about 20 s of work, so the hub does it in the
    background at startup (`MISSION`). Returns what ReplaySource needs."""
    from mindflight.deploy.policy import load_flight_policy
    from mindflight.deploy.runner import FlightRunner, Mission
    from mindflight.deploy.sitl import SitlConfig, SitlCrazyflie
    from mindflight.physics.dynamics import PHYSICS_DT_S
    from mindflight.physics.params import load_drone

    class RecordingSitl(SitlCrazyflie):
        """The fake Crazyflie, writing down its true pose 50 times a second."""

        def _tick(self) -> None:
            super()._tick()
            if self.ticks % round(POLICY_DT_S / PHYSICS_DT_S) == 0:
                self.recorded.append(self.state[0, 0:7].tolist())

        def _reset(self) -> None:
            super()._reset()
            self.recorded: list[list[float]] = []

    policy = load_flight_policy(policy_path)
    params = load_drone(policy.env_kwargs.get("drone"))
    mission = Mission(waypoints=list(MISSION_WAYPOINTS))
    link = RecordingSitl(params, SitlConfig(seed=int(time.time()) % 10_000))
    lines: list[tuple[float, str]] = []
    with link:
        link.reset_estimator()
        start_s = link.now()
        runner = FlightRunner(link, policy, mission, params=params, log_dir=RUNS_DIR / "hub" / "flights",
                              print_fn=lambda text: lines.append((link.now() - start_s, text.split("(log:")[0])))
        result = runner.fly()
    poses = np.array(link.recorded, dtype=np.float32)
    f = mission.fence
    x0, x1, y0, y1, z1 = f.x_m[0], f.x_m[1], f.y_m[0], f.y_m[1], f.z_max_m
    grey = "#5a6472"
    box = [[x0, y0, 0, x1, y0, 0], [x1, y0, 0, x1, y1, 0], [x1, y1, 0, x0, y1, 0], [x0, y1, 0, x0, y0, 0],
           [x0, y0, z1, x1, y0, z1], [x1, y0, z1, x1, y1, z1], [x1, y1, z1, x0, y1, z1], [x0, y1, z1, x0, y0, z1],
           [x0, y0, 0, x0, y0, z1], [x1, y0, 0, x1, y0, z1], [x1, y1, 0, x1, y1, z1], [x0, y1, 0, x0, y1, z1]]
    path = [(0.0, 0.0, mission.takeoff_height_m)] + list(MISSION_WAYPOINTS)
    legs = [[*a, *b, "#f1c21b"] for a, b in zip(path, path[1:])]
    scene = {
        "targets": [list(w) for w in MISSION_WAYPOINTS],
        "lines": [[*seg, grey] for seg in box] + legs,
    }
    reached = sum(result.waypoints_reached)
    header = [
        f"fake Crazyflie · {policy_path.name} · radio delay 5–20 ms each way, sensor noise on",
        f"result: {result.outcome}, {reached} of {len(MISSION_WAYPOINTS)} waypoints · grey box = geofence",
    ]
    return poses, lines, scene, header


class MissionRecording:
    """One fake-Crazyflie flight per hub run, recorded in a background thread."""

    def __init__(self):
        self._future: Future | None = None
        self._lock = threading.Lock()

    def start(self, policy_path: Path) -> None:
        with self._lock:
            if self._future is None:
                self._future = ThreadPoolExecutor(1).submit(record_mission, policy_path)

    def get(self, policy_path: Path) -> tuple:
        self.start(policy_path)
        return self._future.result()


MISSION = MissionRecording()


def build_mission(policy_path: Path) -> Built:
    source = ReplaySource(*MISSION.get(policy_path))
    return Built(source, on_reset=source.reset, note="replays the recorded flight from the start")


def build(show: Show) -> Built:
    if show.id == "race":
        return build_race()
    if show.id == "physics":
        return build_physics()
    if show.id == "train":
        return build_training()
    if show.id == "mission":
        return build_mission(POLICIES / show.policy)
    return build_policy_show(POLICIES / show.policy)


def available_shows() -> list[Show]:
    """Shows whose brains are on disk (all of them in a normal checkout)."""
    return [s for s in SHOWS if s.policy is None or (POLICIES / s.policy).exists()]


# ---------- the hub ----------


class HubSource:
    """A FrameSource that streams whichever show is picked. See the note at the top of this file."""

    def __init__(self, shows: list[Show], first: str, builder: Callable[[Show], Built] = build):
        self.shows = {s.id: s for s in shows}
        self.order = [s.id for s in shows]
        self.builder = builder
        self.speed = 1.0
        self.generation = 0
        self.loading: str | None = None
        self.error = ""
        self._switch_lock = threading.Lock()  # one switch at a time
        self._kept: dict[str, Built] = {}  # shows you've left keep flying here, so going back is instant
        self.current_id = first
        self.current = builder(self.shows[first])
        self._apply_speed()

    # FrameSource: hand everything on to the show running now.
    @property
    def n_drones(self) -> int:
        return self.current.source.n_drones

    @property
    def env(self):
        return getattr(self.current.source, "env", None)

    def frame(self, time_s: float) -> np.ndarray:
        return self.current.source.frame(time_s)

    @property
    def metric_points(self) -> list[dict]:
        return getattr(self.current.source, "metric_points", [])

    def hub_message(self) -> dict:
        return {
            "type": "hub",
            "generation": self.generation,
            "current": self.current_id,
            "loading": self.loading,
            "error": self.error,
            "speed": self.speed,
            "speeds": list(SPEEDS),
            "resettable": self.current.on_reset is not None,
            "note": self.current.note,
            "groups": self.current.groups,
            "shows": [
                {"id": s.id, "group": s.group, "title": s.title, "blurb": s.blurb, "score": s.score}
                for s in (self.shows[i] for i in self.order)
            ],
        }

    def state_messages(self) -> list[dict]:
        built = self.current  # read once: a switch may swap it while we work
        messages = [self.hub_message()]
        if hasattr(built.source, "state_messages"):
            messages += built.source.state_messages()
        return messages

    def reset(self) -> None:
        if self.current.on_reset is not None:
            self.current.on_reset()

    def set_speed(self, speed: float) -> None:
        self.speed = speed
        self._apply_speed()

    def _apply_speed(self) -> None:
        source = self.current.source
        if hasattr(source, "speed"):
            source.speed = self.speed
        if hasattr(source, "max_catch_up_s"):
            source.max_catch_up_s = MAX_CATCH_UP_S * self.speed

    def switch(self, show_id: str) -> None:
        """Swap in the show: the one kept from last time, or a new one. Runs in a worker thread, so the old show
        keeps streaming while a new one is built. Live training isn't kept: leaving it stops it."""
        if show_id not in self.shows:
            raise KeyError(show_id)
        with self._switch_lock:
            old, old_id = self.current, self.current_id
            self.loading = show_id
            self.error = ""
            try:
                if old.stop is not None:  # stop live training first: it shares the GPU
                    old.stop()
                built = self._kept.pop(show_id, None) or self.builder(self.shows[show_id])
            except Exception as error:
                self.error = f"couldn't start {self.shows[show_id].title}: {error}"
                raise
            finally:
                self.loading = None
            if old.stop is None:
                self._kept[old_id] = old
            self.current, self.current_id = built, show_id
            self._apply_speed()
            self.generation += 1


def create_hub_app(hub: HubSource):
    app = create_app(hub, on_reset=hub.reset)

    @app.post("/show/{show_id}")
    def show(show_id: str) -> dict:  # a plain def: FastAPI runs it in a worker thread, the stream goes on
        if show_id not in hub.shows:
            raise HTTPException(404, f"no show {show_id!r}")
        try:
            hub.switch(show_id)
        except Exception as error:
            raise HTTPException(500, str(error)) from error
        return {"ok": True, "current": hub.current_id}

    @app.post("/speed")
    async def speed(value: float) -> dict:
        if value not in SPEEDS:
            raise HTTPException(400, f"speed must be one of {SPEEDS}")
        hub.set_speed(value)
        return {"ok": True, "speed": value}

    return app


def free_port(first: int, tries: int = 20) -> int:
    """The first port from `first` on that nothing on this machine is listening on (play or race may be running)."""
    for port in range(first, first + tries):
        with socket.socket() as probe:
            if probe.connect_ex((HOST, port)) != 0:
                return port
    raise SystemExit(f"ports {first}-{first + tries - 1} are all busy")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m mindflight.hub", description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8000, help="port on 127.0.0.1 (default 8000)")
    parser.add_argument("--first", default="pack", help="the show to start on (default pack)")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    args = parser.parse_args()

    shows = available_shows()
    if args.first not in {s.id for s in shows}:
        parser.error(f"--first must be one of: {', '.join(s.id for s in shows)}")
    print("MindFlight Hub: loading...")
    mission = next((s for s in shows if s.id == "mission"), None)
    if mission is not None:
        MISSION.start(POLICIES / mission.policy)  # record the fake-Crazyflie flight while you watch something else
    hub = HubSource(shows, args.first)
    port = free_port(args.port)
    url = f"http://{HOST}:{port}/"
    print(f"MindFlight Hub: {url}  ({len(shows)} shows, Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_hub_app(hub), host=HOST, port=port, log_level="warning")


if __name__ == "__main__":
    main()
