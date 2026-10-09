"""The fast stepper (CUDA graph replay) must give the same physics as the plain `step`, call after call."""

import importlib.util

import pytest
import torch

from mindflight.physics.dynamics import OMEGA, QUAT, RPM, make_state, step
from mindflight.physics.fast import CudaGraphStepper
from mindflight.physics.params import CRAZYFLIE

SUBSTEPS = 10  # 50 Hz policy, 500 Hz physics
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
needs_triton = pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="torch.compile needs triton")


def tumbling_swarm(n: int, device: str, seed: int = 0) -> torch.Tensor:
    """N drones high up (no floor contact), each with a random orientation, spin and motor speeds."""
    gen = torch.Generator().manual_seed(seed)
    state = make_state(n, "cpu", pos=[0.0, 0.0, 100.0])
    state[:, QUAT] = torch.nn.functional.normalize(torch.randn(n, 4, generator=gen), dim=1)
    state[:, OMEGA] = 20.0 * torch.randn(n, 3, generator=gen)  # rad/s, a hard tumble on every axis
    state[:, RPM] = CRAZYFLIE.max_rpm * torch.rand(n, 4, generator=gen)
    return state.to(device)


def random_commands(n: int, device: str, count: int, seed: int = 1) -> list[torch.Tensor]:
    """`count` random motor commands in [0, max_rpm], one per policy action."""
    gen = torch.Generator().manual_seed(seed)
    return [(CRAZYFLIE.max_rpm * torch.rand(n, 4, generator=gen)).to(device) for _ in range(count)]


def eager_run(state: torch.Tensor, rpm_cmd: torch.Tensor) -> torch.Tensor:
    """What one `stepper.run` must equal: SUBSTEPS plain steps with the same command."""
    for _ in range(SUBSTEPS):
        state = step(state, rpm_cmd)
    return state


@cuda_only
@pytest.mark.parametrize("n", [1, 4096])
def test_graph_matches_eager_over_1000_steps(n):
    # 100 replays × 10 substeps = 1,000 physics steps (2 s) of tumbling drones with a new random command each replay.
    stepper = CudaGraphStepper(n, "cuda", substeps=SUBSTEPS)
    assert stepper.uses_cuda_graph and not stepper.compiled
    eager = fast = tumbling_swarm(n, "cuda")
    for rpm_cmd in random_commands(n, "cuda", count=100):
        eager = eager_run(eager, rpm_cmd)
        fast = stepper.run(fast, rpm_cmd)
    assert torch.isfinite(fast).all()
    assert fast[:, 2].min() > 0.0  # still flying, so the floor never hid a difference
    # The graph replays the very same kernels on the same numbers, so this agrees to float32 rounding
    # (in practice it is bit-identical).
    torch.testing.assert_close(fast, eager, rtol=1e-5, atol=1e-5)


@cuda_only
@needs_triton
@pytest.mark.parametrize("n", [1, 4096])
def test_compiled_graph_is_as_accurate_as_eager_over_1000_steps(n):
    # torch.compile fuses kernels and so rounds in a slightly different order (~1e-8 per step). Over 1,000 steps of
    # a hard tumble, tiny rounding differences grow for a few drones (the same happens between any two float32
    # codes), so comparing compiled to eager directly would only measure that growth. Instead, run a float64
    # reference and check the compiled float32 run is no further from it than the eager float32 run is.
    stepper = CudaGraphStepper(n, "cuda", substeps=SUBSTEPS, compile=True)
    assert stepper.uses_cuda_graph and stepper.compiled
    eager = fast = tumbling_swarm(n, "cuda")
    reference = eager.double()
    for rpm_cmd in random_commands(n, "cuda", count=100):
        eager = eager_run(eager, rpm_cmd)
        fast = stepper.run(fast, rpm_cmd)
        reference = eager_run(reference, rpm_cmd.double())
    assert torch.isfinite(fast).all()
    eager_error = (eager.double() - reference).abs().amax(dim=0)  # worst error per state column
    fast_error = (fast.double() - reference).abs().amax(dim=0)
    assert (fast_error <= 2.0 * eager_error + 1e-6).all(), (fast_error, eager_error)
    # And for the typical drone the two float32 runs agree closely.
    assert (fast - eager).abs().median() < 1e-5


@cuda_only
def test_replay_uses_new_inputs():
    # If run() forgot to copy into the static buffers, the graph would keep replaying the recorded inputs.
    n = 64
    stepper = CudaGraphStepper(n, "cuda", substeps=SUBSTEPS)
    state = tumbling_swarm(n, "cuda")
    low_cmd, high_cmd = random_commands(n, "cuda", count=2)

    out_low = stepper.run(state, low_cmd)
    out_high = stepper.run(state, high_cmd)  # same state, new command
    torch.testing.assert_close(out_low, eager_run(state, low_cmd), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_high, eager_run(state, high_cmd), rtol=1e-5, atol=1e-5)
    assert not torch.equal(out_low, out_high)

    other_state = tumbling_swarm(n, "cuda", seed=7)  # new state, same command
    out_other = stepper.run(other_state, high_cmd)
    torch.testing.assert_close(out_other, eager_run(other_state, high_cmd), rtol=1e-5, atol=1e-5)
    assert not torch.equal(out_other, out_high)


@cuda_only
def test_outputs_are_copies_unless_asked():
    n = 8
    stepper = CudaGraphStepper(n, "cuda", substeps=SUBSTEPS)
    state = tumbling_swarm(n, "cuda")
    commands = random_commands(n, "cuda", count=2)
    first = stepper.run(state, commands[0])
    kept = first.clone()
    stepper.run(state, commands[1])
    assert torch.equal(first, kept)  # default: the next replay doesn't overwrite what we got back

    view = stepper.run(state, commands[0], copy_output=False)
    assert view.data_ptr() == stepper.static_output.data_ptr()  # the graph's own buffer, as documented


def test_cpu_fallback_matches_exactly():
    n = 32
    stepper = CudaGraphStepper(n, "cpu", substeps=SUBSTEPS)
    assert not stepper.uses_cuda_graph
    eager = fast = tumbling_swarm(n, "cpu")
    for rpm_cmd in random_commands(n, "cpu", count=10):
        eager = eager_run(eager, rpm_cmd)
        fast = stepper.run(fast, rpm_cmd)
    assert torch.equal(fast, eager)  # on the CPU it is literally the same loop


def test_wrong_shape_is_rejected():
    stepper = CudaGraphStepper(4, "cpu")
    with pytest.raises(ValueError, match="N = 4"):
        stepper.run(make_state(5, "cpu"), torch.zeros(5, 4))


@cuda_only
@pytest.mark.filterwarnings("ignore:Warning. Profiler clears events")
def test_step_makes_no_cpu_to_gpu_copies():
    # The rule that lets a step be recorded: after the first call has filled the caches, `step` must not copy
    # anything from the CPU. The profiler lists such a copy as a "Memcpy HtoD" kernel.
    from torch.profiler import ProfilerActivity, profile

    state = tumbling_swarm(16, "cuda")
    rpm_cmd = random_commands(16, "cuda", count=1)[0]
    step(state, rpm_cmd)  # fills the device caches
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            state = step(state, rpm_cmd)
        torch.cuda.synchronize()
    copies = [e.key for e in prof.key_averages() if "HtoD" in e.key]
    assert copies == []


@cuda_only
def test_compile_failure_falls_back_to_plain_step(monkeypatch):
    # On a machine without triton, compile=True must still give a working (graph-only) stepper, with a warning.
    def broken_compile(fn, **kwargs):
        def compiled(*args):
            raise RuntimeError("no triton here")

        return compiled

    monkeypatch.setattr(torch, "compile", broken_compile)
    with pytest.warns(UserWarning, match="torch.compile failed"):
        stepper = CudaGraphStepper(4, "cuda", substeps=SUBSTEPS, compile=True)
    assert stepper.uses_cuda_graph and not stepper.compiled
    state = tumbling_swarm(4, "cuda")
    rpm_cmd = random_commands(4, "cuda", count=1)[0]
    torch.testing.assert_close(stepper.run(state, rpm_cmd), eager_run(state, rpm_cmd), rtol=1e-5, atol=1e-5)
