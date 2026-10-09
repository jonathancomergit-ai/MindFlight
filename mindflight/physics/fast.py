"""Fast physics for training: record many physics steps once as a CUDA graph, then replay them with one call.

Why is the plain `step` slow on a GPU for small swarms?
    A GPU is a huge number of small workers. To give them a job, the CPU "launches a kernel": it sends one small
    program (say, "multiply these two tensors") to the GPU. Every PyTorch operation is at least one kernel, and one
    `step` is about 80 of them (the RK4 rotation update alone is four rounds of quaternion math). Launching a kernel
    costs the CPU roughly 5-20 µs, whatever the tensor size. Running one of our kernels on, say, 4,096 drones takes
    the GPU only about 2 µs. So the GPU spends most of each step waiting for the CPU to hand it the next job:
    `docs/benchmark.md` measured the GPU busy only 12 % of the time. That is "launch-bound".

What a CUDA graph does
    Capture: we run the steps once while CUDA *records* every kernel and its arguments (which memory to read and
    write) instead of running them. The result is a graph: the whole list of kernels, ready to go.
    Replay: `graph.replay()` sends the whole list to the GPU in one launch. The GPU runs the same kernels on the
    same memory, back to back, with no CPU in between. 10 physics steps × ~80 kernels become one launch.

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
    `compile=True`, `torch.compile` first fuses the ~80 kernels of a step into a few, and then the graph records
    those. See `CudaGraphStepper` for when to use it.

On the CPU there are no kernels to launch and no graphs, so `CudaGraphStepper` simply calls `step` in a loop.

Usage (the env calls this once per policy action):

    stepper = CudaGraphStepper(n_drones, "cuda", substeps=10, compile=True)
    state = make_state(n_drones, "cuda", pos=[0.0, 0.0, 1.0])
    state = stepper.run(state, rpm_cmd)   # 10 physics steps = 20 ms of flight
"""

import warnings

import torch

from mindflight.physics.dynamics import PHYSICS_DT_S, STATE_SIZE, make_state, step
from mindflight.physics.params import CRAZYFLIE, CrazyflieParams

POLICY_SUBSTEPS = 10  # physics steps per policy action: 500 Hz physics / 50 Hz policy (PLAN.md -> Timing)
WARMUP_RUNS = 3  # eager runs on a side stream before capture, as the PyTorch CUDA graph docs ask


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
        CPU's launch cost, but the GPU still runs ~80 tiny kernels per step, each with a few µs of start-up and
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
    ):
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
        if self.uses_cuda_graph:
            self._capture()

    def _advance(self, state: torch.Tensor, rpm_cmd: torch.Tensor) -> torch.Tensor:
        """`substeps` physics steps with the same command. This is what the graph records."""
        for _ in range(self.substeps):
            state = self.step_fn(state, rpm_cmd, self.dt, self.params)
        return state

    def _warm_up(self) -> None:
        for _ in range(WARMUP_RUNS):
            self._advance(self.static_state, self.static_rpm_cmd)

    @torch.no_grad()
    def _capture(self) -> None:
        """Make the static buffers, warm up, then record `_advance` into one CUDA graph."""
        # Static input buffers: replay always reads these. Any valid starting values do for the recording.
        self.static_state = make_state(self.n, self.device, pos=[0.0, 0.0, 1.0], rpm=[self.params.hover_rpm] * 4)
        self.static_rpm_cmd = torch.full((self.n, 4), self.params.hover_rpm, device=self.device)

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
            self.static_output = self._advance(self.static_state, self.static_rpm_cmd)

    @torch.no_grad()  # the simulator never needs gradients, and a graph can't record autograd anyway
    def run(self, state: torch.Tensor, rpm_cmd: torch.Tensor, copy_output: bool = True) -> torch.Tensor:
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
        if not self.uses_cuda_graph:
            return self._advance(state, rpm_cmd)

        self.static_state.copy_(state)  # the static-buffer rule: copy new inputs into the recorded memory
        self.static_rpm_cmd.copy_(rpm_cmd)
        self.graph.replay()  # all substeps × ~80 kernels, one launch
        return self.static_output.clone() if copy_output else self.static_output
