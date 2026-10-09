"""Run a real simulation in real time and hand its frames to the viewer.

`SimSource` fits the `FrameSource` protocol in `server.py` (`n_drones` + `frame(time_s)`), so
`create_app(SimSource(...))` streams a live sim. It knows nothing about any one task: you give it two functions,

    start_state()  -> (N, 17) state tensor     where every drone starts (called again on reset)
    step(state)    -> (N, 17) state tensor     one physics step of dt_s seconds (motors, forces, floor, ...)

`mindflight.demo` uses it for the four test groups. Training and play code can use it the same way later: their
`step` would run the policy and then `mindflight.physics.dynamics.step`.

How real time works. The viewer asks for a frame about 30 times a second. Each time, `SimSource` looks at its own
clock, sees how much wall time passed since the last frame, and runs that many physics steps:

    steps this frame = (wall time since last frame × speed) / dt_s      (≈ 16.7 at 30 frames/s and 500 Hz)

Leftover time smaller than one step is carried over to the next frame, so nothing is lost. The catch-up is capped
at `max_catch_up_s` of sim time per frame: if the computer stalls for 5 s, the sim moves on by at most 0.1 s
instead of freezing while it grinds through 2,500 steps. With nobody watching, `frame` isn't called, so the sim
waits. The sim keeps its own clock (it ignores the viewer's `time_s`), so two browser tabs see the same swarm.
"""

import threading
import time
from collections.abc import Callable

import numpy as np
import torch

from mindflight.physics.dynamics import PHYSICS_DT_S

POSE_COLUMNS = slice(0, 7)  # x, y, z, qw, qx, qy, qz: the first 7 state columns are exactly the viewer's frame


class SimSource:
    """A simulation that advances with wall time and gives the viewer (N, 7) float32 poses."""

    def __init__(
        self,
        start_state: Callable[[], torch.Tensor],
        step: Callable[[torch.Tensor], torch.Tensor],
        dt_s: float = PHYSICS_DT_S,
        speed: float = 1.0,
        max_catch_up_s: float = 0.1,
        restart_after_s: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        """
        start_state: makes the starting (N, 17) state. step: advances a state by one physics step of dt_s.
        dt_s: physics step in s (500 Hz by default). speed: sim seconds per wall second (0.25 = slow motion).
        max_catch_up_s: most sim time run in one frame, in s. restart_after_s: start over after this much sim
        time (None = never). clock: wall clock in s; tests pass a fake one.
        """
        self._make_start_state = start_state
        self._step = step
        self.dt_s = dt_s
        self.speed = speed
        self.max_catch_up_s = max_catch_up_s
        self.restart_after_s = restart_after_s
        self._clock = clock
        self._lock = threading.Lock()  # reset can come from another thread than frame
        self.reset()
        self.n_drones = self.state.shape[0]

    def reset(self) -> None:
        """Put every drone back at its start and set sim time to 0."""
        with self._lock:
            self._reset_unlocked()

    def _reset_unlocked(self) -> None:
        self.state = self._make_start_state()
        self.sim_time_s = 0.0
        self._unspent_s = 0.0  # sim time owed but smaller than one step
        self._last_clock_s = self._clock()

    def advance(self, sim_seconds: float) -> None:
        """Run round(sim_seconds / dt_s) physics steps right now, ignoring the clock."""
        with self._lock:
            self._run_steps(round(sim_seconds / self.dt_s))

    def _run_steps(self, n_steps: int) -> None:
        with torch.no_grad():
            for _ in range(n_steps):
                self.state = self._step(self.state)
                self.sim_time_s += self.dt_s
                if self.restart_after_s is not None and self.sim_time_s >= self.restart_after_s:
                    self._reset_unlocked()

    def poses(self) -> np.ndarray:
        """Current poses, shape (N, 7), float32: x, y, z (m, world) and the quaternion qw, qx, qy, qz."""
        with self._lock:
            return self.state[:, POSE_COLUMNS].detach().to("cpu", torch.float32).numpy().copy()

    def frame(self, time_s: float) -> np.ndarray:
        """Catch the sim up with the wall clock (capped), then return the poses, (N, 7) float32.

        `time_s` (the viewer's time since it connected) is not used: see the note at the top of this file.
        """
        with self._lock:
            now_s = self._clock()
            owed_s = (now_s - self._last_clock_s) * self.speed
            self._last_clock_s = now_s
            self._unspent_s += min(max(owed_s, 0.0), self.max_catch_up_s)
            n_steps = int(self._unspent_s / self.dt_s)
            self._unspent_s -= n_steps * self.dt_s
            self._run_steps(n_steps)
        return self.poses()
