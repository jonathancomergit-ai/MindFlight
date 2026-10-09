"""FastAPI server that streams drone poses to the Three.js page in `static/`.

Routes:
- `GET /`        the viewer page (`static/index.html`)
- `GET /static/` the page's JavaScript
- `GET /config`  JSON with the drone count and the drone size the page draws (from `physics/params.py`),
                 plus any `extra_config` the caller adds (the demo adds its drone groups for the legend)
- `POST /reset`  start the sim over (only when `create_app` got an `on_reset` function; the page's R key)
- `/ws`          WebSocket that sends frames at about 30 per second

Frame format (the contract the simulator uses to talk to the viewer):
    One **binary** WebSocket message per frame. It is a little-endian float32 array of N × 7 values,
    one row of 7 per drone, in drone order:

        x, y, z, qw, qx, qy, qz

    Position is in metres in the world frame (Z up, x forward, y left). The quaternion is (w, x, y, z) and rotates
    body → world, the same as columns 0:7 of the simulator state in PLAN.md. The page works out N as
    byte_length / (7 × 4), so N may change between frames.

Frames come from a `FrameSource`: anything with an `n_drones` attribute and a `frame(time_s)` method returning an
(N, 7) float32 array. `FakeSwarm` below is one (no physics). `sim_source.SimSource` runs the real simulator in real
time (`python -m mindflight.demo`). `policy_source.PolicySource` flies trained policies on a task
(`python -m mindflight.play`, `python -m mindflight.train --view`).

Text messages (optional). Besides the binary pose frames, a source may send JSON **text** messages, for things that
change rarely or aren't drone poses. A source opts in with either or both of:

    state_messages() -> list[dict]   the *current* scene / info / board / charts. Checked after every frame; each
                                     page is sent a message only when it differs from the last one of that type
                                     the page got (so a newly opened page gets everything at once).
    metric_points: list[dict]        an append-only list of chart points. Each page is sent the ones it hasn't
                                     seen yet, as one "metrics" message.

Message types the page understands (all coordinates in metres, world frame):

    {"type": "scene", "targets": [[x, y, z], ...],                      small spheres (e.g. waypoint targets)
                      "gates": [{"c": [x, y, z], "yaw": ψ, "size": s, "state": "next" | "later" | "done"}, ...],
                      "pillars": [[x, y, radius], ...],                 floor-to-ceiling cylinders
                      "spheres": [[x, y, z, r, "#rrggbb"], ...],          general shapes for any task
                      "cylinders": [[x, y, z_bottom, height, r, "#rrggbb"], ...],
                      "lines": [[x1, y1, z1, x2, y2, z2, "#rrggbb"], ...]}
        Replaces the whole previous scene. Any key may be missing.
    {"type": "info", "lines": ["...", ...]}                             text lines under the HUD
    {"type": "charts", "series": [{"key": "...", "label": "..."}, ...]} which metrics to plot
    {"type": "metrics", "points": [{"iteration": 1, "<key>": value, ...}, ...]}   new points for the charts
    {"type": "board", "title": "...", "columns": [...], "rows": [[...], ...]}     a table (the race leaderboard)
"""

import asyncio
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mindflight.physics.params import CRAZYFLIE

STATIC_DIR = Path(__file__).parent / "static"
FRAME_RATE_HZ = 30.0
VALUES_PER_DRONE = 7  # x, y, z, qw, qx, qy, qz


class FrameSource(Protocol):
    """Anything that can make frames: the fake swarm now, the real simulator later."""

    n_drones: int

    def frame(self, time_s: float) -> np.ndarray:
        """Poses of all drones at `time_s` seconds, shape (N, 7), float32."""
        ...


class FakeSwarm:
    """N drones in a square grid, bobbing up and down and slowly yawing. No physics, just for the viewer.

    Drone i sits at its grid spot (spacing_m apart, centred on the origin) at height

        z = hover_height_m + bob_amplitude_m · sin(2π · bob_hz · t + phase_i)

    and turns about the world z axis at yaw_rate_rad_s, so its quaternion is (cos(ψ/2), 0, 0, sin(ψ/2)) with
    ψ = yaw_rate_rad_s · t + phase_i. Each drone gets its own phase so the grid ripples.
    """

    def __init__(
        self,
        n_drones: int,
        spacing_m: float = 0.5,
        hover_height_m: float = 1.0,
        bob_amplitude_m: float = 0.2,
        bob_hz: float = 0.5,
        yaw_rate_rad_s: float = 0.5,
    ):
        self.n_drones = n_drones
        self.hover_height_m = hover_height_m
        self.bob_amplitude_m = bob_amplitude_m
        self.bob_hz = bob_hz
        self.yaw_rate_rad_s = yaw_rate_rad_s

        side = math.ceil(math.sqrt(n_drones))
        index = np.arange(n_drones)
        row, col = index // side, index % side
        centre = (side - 1) / 2
        self.grid_x_m = ((row - centre) * spacing_m).astype(np.float32)
        self.grid_y_m = ((col - centre) * spacing_m).astype(np.float32)
        # Phase grows along the diagonal, so the bobbing looks like a wave crossing the grid.
        self.phase_rad = ((row + col) * 0.4).astype(np.float32)

    def frame(self, time_s: float) -> np.ndarray:
        """Poses at `time_s` seconds, shape (N, 7), float32: x, y, z, qw, qx, qy, qz."""
        poses = np.zeros((self.n_drones, VALUES_PER_DRONE), dtype=np.float32)
        poses[:, 0] = self.grid_x_m
        poses[:, 1] = self.grid_y_m
        poses[:, 2] = self.hover_height_m + self.bob_amplitude_m * np.sin(
            2 * math.pi * self.bob_hz * time_s + self.phase_rad
        )
        yaw_rad = self.yaw_rate_rad_s * time_s + self.phase_rad
        poses[:, 3] = np.cos(yaw_rad / 2)  # qw
        poses[:, 6] = np.sin(yaw_rad / 2)  # qz (qx = qy = 0: rotation about world z only)
        return poses


def encode_frame(poses: np.ndarray) -> bytes:
    """Turn an (N, 7) pose array into one binary WebSocket message: little-endian float32, row by row."""
    return np.ascontiguousarray(poses, dtype="<f4").tobytes()


def create_app(
    source: FrameSource,
    extra_config: dict[str, Any] | None = None,
    on_reset: Callable[[], None] | None = None,
) -> FastAPI:
    """Build the viewer app that streams frames from `source`.

    extra_config: more keys for `GET /config` (for example the demo's `groups`, which the page turns into a legend).
    on_reset: called by `POST /reset` (the page's R key). Without it there is no /reset and the page hides the key.
    """
    app = FastAPI(title="MindFlight viewer")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    # Draw the drone the source flies (a PolicySource's env knows it), else the default Crazyflie.
    drone = getattr(getattr(source, "env", None), "params", CRAZYFLIE)

    @app.get("/config")
    def config() -> dict:
        """Numbers the page needs to draw a drone at real scale, from physics/params.py."""
        return {
            "n_drones": source.n_drones,
            "frame_rate_hz": FRAME_RATE_HZ,
            "arm_m": drone.arm_m,
            "prop_radius_m": drone.prop_radius_m,
            "resettable": on_reset is not None,
            **(extra_config or {}),
        }

    if on_reset is not None:

        @app.post("/reset")
        async def reset() -> dict:
            """Start the sim over. `async` so it runs on the same thread as the frame sender, never mid-frame."""
            on_reset()
            return {"ok": True}

    @app.websocket("/ws")
    async def stream(websocket: WebSocket) -> None:
        """Send one frame every 1/30 s until the browser goes away."""
        await websocket.accept()
        start_s = time.perf_counter()

        last_sent: dict[str, str] = {}  # message type -> the JSON this page last got, for state messages
        points_sent = 0  # how many of source.metric_points this page has

        async def send_text_messages() -> None:
            nonlocal points_sent
            for message in source.state_messages() if hasattr(source, "state_messages") else []:
                text = json.dumps(message)
                if last_sent.get(message["type"]) != text:
                    await websocket.send_text(text)
                    last_sent[message["type"]] = text
            points = getattr(source, "metric_points", None)
            if points is not None and len(points) > points_sent:
                new_points = list(points[points_sent:])
                points_sent += len(new_points)
                await websocket.send_text(json.dumps({"type": "metrics", "points": new_points}))

        async def send_frames() -> None:
            next_send_s = time.perf_counter()
            while True:
                await websocket.send_bytes(encode_frame(source.frame(time.perf_counter() - start_s)))
                await send_text_messages()
                next_send_s += 1 / FRAME_RATE_HZ
                await asyncio.sleep(max(0.0, next_send_s - time.perf_counter()))

        sender = asyncio.create_task(send_frames())
        try:
            # The page never sends anything; wait here so we notice when it disconnects.
            while (await websocket.receive())["type"] != "websocket.disconnect":
                pass
        finally:
            sender.cancel()
            try:
                await sender
            except (asyncio.CancelledError, Exception):
                pass  # the socket is gone, so a failed last send doesn't matter

    return app
