"""Fast physics for training: record many physics steps once as a CUDA graph, then replay them with one call.

Why is the plain `step` slow on a GPU for small swarms?
    A GPU is a huge number of small workers. To give them a job, the CPU "launches a kernel": it sends one small
    program (say, "multiply these two tensors") to the GPU. Every PyTorch operation is at least one kernel, and one
    `step` is about 200 of them (the RK4 rotation update alone is four rounds of quaternion math). Launching a kernel
    costs the CPU roughly 5-20 µs, whatever the tensor size. Running one of our kernels on, say, 4,096 drones takes
    the GPU only about 2 µs. So the GPU spends most of each step waiting for the CPU to hand it the next job:
    `docs/benchmark.md` measured the GPU busy only 12 % of the time. That is "launch-bound".

What a CUDA graph does
    Capture: we run the steps once while CUDA *records* every kernel and its arguments (which memory to read and
    write) instead of running them. The result is a graph: the whole list of kernels, ready to go.
    Replay: `graph.replay()` sends the whole list to the GPU in one launch. The GPU runs the same kernels on the
    same memory, back to back, with no CPU in between. 10 physics steps × ~200 kernels become one launch.

The static-buffer rule (the one thing to remember)
    A graph remembers *memory addresses*, not Python variables. Replay always reads from the exact tensors that
    were used during capture and writes into the exact tensors it produced then. So:
    - New inputs must be *copied into* the captured input tensors (`static_state.copy_(state)`), never passed as
      new tensors. `run()` does this for you.
    - The output tensor is overwritten by the next replay. `run()` returns a copy (`clone`) by default, so what
      you get back is yours to keep. Pass `copy_output=False` to get the graph's own output tensor instead
      (one kernel cheaper), but then read or copy it before the next `run()`.
    - Nothing inside the captured code may talk to the CPU: no `.item()`, no `print(tensor)`, no CPU -> GPU copies
      (that is why `motors`, `dynamics` and `rotations` keep their constants on the GPU, see `device_cache.py`).
    - Shapes are fixed: a stepper is built for exactly N drones.

Going further: `compile=True`
    A graph removes the CPU's launch cost, but the GPU still runs every tiny kernel one after another. With
    `compile=True`, `torch.compile` first fuses the ~200 kernels of a step into a few, and then the graph records
    those. See `CudaGraphStepper` for when to use it.

On the CPU there are no kernels to launch and no graphs, so `CudaGraphStepper` simply calls `step` in a loop.

Usage (the env calls this once per policy action):

    stepper = CudaGraphStepper(n_drones, "cuda", substeps=10, compile=True)
    state = make_state(n_drones, "cuda", pos=[0.0, 0.0, 1.0])
    state = stepper.run(state, rpm_cmd)   # 10 physics steps = 20 ms of flight
"""

import warnings

import torch

from mindflight.physics.aero import DroneVariation
from mindflight.physics.controller import ControllerGains, controller_step
from mindflight.physics.dynamics import PHYSICS_DT_S, STATE_SIZE, make_state, step
from mindflight.physics.params import CRAZYFLIE, CrazyflieParams

POLICY_SUBSTEPS = 10  # physics steps per policy action: 500 Hz physics / 50 Hz policy (PLAN.md -> Timing)
WARMUP_RUNS = 3  # eager runs on a side stream before capture, as the PyTorch CUDA graph docs ask


VARIATION_TENSORS = ("mass_scale", "thrust_scale", "motor_tau_scale", "wind_mps", "world_id", "com_offset_m")


class CudaGraphStepper:
    """Advance N drones by `substeps` physics steps per call, holding the motor command fixed over those steps.

    One `run(state, rpm_cmd)` is exactly

        for _ in range(substeps):
            state = step(state, rpm_cmd, dt, params)

    (the policy picks rpm_cmd at 50 Hz and the physics runs 10 times in between at 500 Hz). On CUDA all of it is
    one CUDA graph replay; on the CPU it is that loop. The numbers are the same either way.

    n: number of drones (fixed: the graph is recorded for this N).
    device: "cuda", "cuda:0", "cpu", ... Defaults to "cuda" when a GPU is available.
    substeps: physics steps per `run()` call. dt: physics step in s. params: the drone.
    compile: CUDA only. Also run `torch.compile` on the step before recording the graph. The graph removes the
        CPU's launch cost, but the GPU still runs ~200 tiny kernels per step, each with a few µs of start-up and
        its own trip through GPU memory. The compiler fuses them into a handful of bigger kernels, which made the
        stepper about 10× faster again on an RTX 5090 (docs/benchmark.md). Needs the `triton` package (Windows:
        `pip install "triton-windows<3.8"` for torch 2.11) and ~10 s of compiling the first time (later runs reuse
        an on-disk cache). Results match plain `step` to float32 rounding (~1e-8), not bit for bit, because the
        fused kernels add things up in a different order. If compiling fails (no triton), the stepper warns and
        uses plain `step`; `self.compiled` tells you which one you got.
    """

    def __init__(
        self,
        n: int,
        device: torch.device | str | None = None,
        substeps: int = POLICY_SUBSTEPS,
        dt: float = PHYSICS_DT_S,
        params: CrazyflieParams = CRAZYFLIE,
        compile: bool = False,
        variation: DroneVariation | None = None,
        controller: ControllerGains | None = None,
    ):
        """variation (Level 5): a template saying which per-drone extras the physics uses. Its flags (drag,
        ground_effect, downwash) are fixed when the graph is recorded; its tensors get static buffers, and `run`
        copies new values into them each call (so wind and mass can change every step without re-recording).

        controller (Phase 2): run the onboard controller (physics/controller.py) before every physics step. Then
        use `run_controlled(state, setpoint, integral)`: the command is an attitude setpoint instead of motor
        speeds, and the controller's integral is carried in and out."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n = n
        self.device = torch.device(device)
        self.substeps = substeps
        self.dt = dt
        self.params = params
        self.uses_cuda_graph = self.device.type == "cuda"
        self.compiled = compile and self.uses_cuda_graph
        self.step_fn = torch.compile(step, dynamic=False) if self.compiled else step
        self.graph = None
        self.variation = variation
        self.static_variation = None
        if variation is not None:
            # Buffers for the variation's tensors (or None where the template has None). Same values as given.
            fields = {}
            for name in VARIATION_TENSORS:
                value = getattr(variation, name)
                fields[name] = None if value is None else value.to(self.device).clone()
            self.static_variation = DroneVariation(
                **fields,
                drag=variation.drag,
                ground_effect=variation.ground_effect,
                downwash=variation.downwash,
                group_size=variation.group_size,  # a fixed shape, so it is recorded into the graph like the flags
            )
        self.controller = controller
        if self.uses_cuda_graph:
            self._capture()

    def _advance(self, state: torch.Tensor, command: torch.Tensor, integral: torch.Tensor | None = None):
        """`substeps` physics steps with the same command. This is what the graph records.

        Without a controller the command is motor RPM and this returns the new state. With one, the command is an
        attitude setpoint, the controller picks motor speeds before every physics step, and this returns
        (new state, new integral)."""
        for _ in range(self.substeps):
            if self.controller is not None:
                rpm_cmd, integral = controller_step(state, command, integral, self.dt, self.params, self.controller)
            else:
                rpm_cmd = command
            state = self.step_fn(state, rpm_cmd, self.dt, self.params, self.static_variation)
        return state if self.controller is None else (state, integral)

    def _warm_up(self) -> None:
        for _ in range(WARMUP_RUNS):
            self._advance(self.static_state, self.static_rpm_cmd, self.static_integral)

    @torch.no_grad()
    def _capture(self) -> None:
        """Make the static buffers, warm up, then record `_advance` into one CUDA graph."""
        # Static input buffers: replay always reads these. Any valid starting values do for the recording.
        self.static_state = make_state(self.n, self.device, pos=[0.0, 0.0, 1.0], rpm=[self.params.hover_rpm] * 4)
        self.static_rpm_cmd = torch.full((self.n, 4), self.params.hover_rpm, device=self.device)
        self.static_integral = None
        if self.controller is not None:  # the command buffer holds a setpoint: hover thrust, level, no yaw
            self.static_rpm_cmd = torch.zeros(self.n, 4, device=self.device)
            self.static_rpm_cmd[:, 0] = self.params.weight_n
            self.static_integral = torch.zeros(self.n, 3, device=self.device)

        # Warm-up on a side stream: the first calls allocate memory, fill the constant caches (device_cache.py),
        # pick kernels and (with compile=True) compile. None of that may happen while recording.
        side_stream = torch.cuda.Stream(self.device)
        side_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side_stream):
            try:
                self._warm_up()
            except Exception as error:  # torch.compile without triton, or without a working compiler
                if not self.compiled:
                    raise
                warnings.warn(f"torch.compile failed, using plain step in the CUDA graph instead: {error}")
                self.compiled, self.step_fn = False, step
                self._warm_up()
        torch.cuda.current_stream(self.device).wait_stream(side_stream)

        # Record. Inside this block the kernels are written down, not run. `static_output` is the graph's own
        # output tensor: every replay writes the new state into this same memory.
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self._advance(self.static_state, self.static_rpm_cmd, self.static_integral)

    @torch.no_grad()  # the simulator never needs gradients, and a graph can't record autograd anyway
    def run(
        self,
        state: torch.Tensor,
        rpm_cmd: torch.Tensor,
        copy_output: bool = True,
        variation: DroneVariation | None = None,
    ) -> torch.Tensor:
        """Advance every drone by `substeps` physics steps and return the new (N, 17) state.

        state: (N, 17) float32, the layout in `dynamics.py`. rpm_cmd: (N, 4) commanded RPM, held for all substeps.
        Neither input is changed. Both should already be on this stepper's device (a copy from elsewhere works,
        but it is slow).

        copy_output=True (default): the result is a new tensor, yours to keep and change.
        copy_output=False (CUDA only): the result is the graph's output buffer itself. It is overwritten by the
            next `run()`, so use it (or copy what you need) before calling `run()` again, and don't write into it.
        """
        if state.shape != (self.n, STATE_SIZE) or rpm_cmd.shape != (self.n, 4):
            raise ValueError(
                f"stepper is built for N = {self.n}: expected state ({self.n}, {STATE_SIZE}) and rpm_cmd ({self.n}, 4), "
                f"got {tuple(state.shape)} and {tuple(rpm_cmd.shape)}"
            )
        if variation is not None:  # new per-drone values for the variation's buffers
            for name in VARIATION_TENSORS:
                value, buffer = getattr(variation, name), getattr(self.static_variation, name)
                if value is not None:
                    buffer.copy_(value)
        if not self.uses_cuda_graph:
            return self._advance(state, rpm_cmd)

        self.static_state.copy_(state)  # the static-buffer rule: copy new inputs into the recorded memory
        self.static_rpm_cmd.copy_(rpm_cmd)
        self.graph.replay()  # all substeps × ~200 kernels, one launch
        return self.static_output.clone() if copy_output else self.static_output

    @torch.no_grad()
    def run_controlled(
        self,
        state: torch.Tensor,
        setpoint: torch.Tensor,
        integral: torch.Tensor,
        variation: DroneVariation | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Like `run`, for a stepper built with a controller: setpoint (N, 4) = thrust (N), roll (rad), pitch (rad),
        yaw rate (rad/s), held for all substeps; integral (N, 3) is the controller's memory. Returns new copies of
        (state, integral)."""
        if self.controller is None:
            raise ValueError("this stepper has no controller: build it with controller=ControllerGains()")
        if variation is not None:
            for name in VARIATION_TENSORS:
                value, buffer = getattr(variation, name), getattr(self.static_variation, name)
                if value is not None:
                    buffer.copy_(value)
        if not self.uses_cuda_graph:
            return self._advance(state, setpoint, integral)
        self.static_state.copy_(state)
        self.static_rpm_cmd.copy_(setpoint)
        self.static_integral.copy_(integral)
        self.graph.replay()
        new_state, new_integral = self.static_output
        return new_state.clone(), new_integral.clone()
