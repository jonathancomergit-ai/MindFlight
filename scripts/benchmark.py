"""How many drones can we fly? Time the physics `step()` for many swarm sizes on CPU and GPU.

    python scripts/benchmark.py            # full run, prints the table and writes docs/benchmark.md
    python scripts/benchmark.py --quick    # short timings, for a smoke test

For every swarm size N, device and way of running `step` we report:
- steps/s:        how many times per second we can advance the *whole* swarm by one physics step (dt = 2 ms)
- drone-steps/s:  N × steps/s, the total amount of simulated drone work per second. This is the number that tells
                  you how much flying the hardware can do; steps/s alone always looks best at N = 1.
- real-time ×:    steps/s × dt. At 1.0 the swarm flies exactly as fast as real time; above 1 it is faster.

Ways of running `step`:
- eager:                plain PyTorch, one GPU kernel (or CPU loop) per operation
- compile (inductor):   `torch.compile(step)`, the default backend, which fuses operations into a few kernels.
                        It needs Triton for CUDA and a C++ compiler for CPU; if they are missing the row says so.
- compile (cudagraphs): `torch.compile(step, backend="cudagraphs")`, CUDA only. Records the step's kernels once and
                        replays them with one launch, which removes the per-kernel launch cost but fuses nothing.

GPU timings call `torch.cuda.synchronize()` before and after, because CUDA calls return before the GPU finishes.
"""

import argparse
import logging
import platform
import statistics
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so it runs without `pip install -e .`

from mindflight.physics.dynamics import PHYSICS_DT_S, make_state, step  # noqa: E402
from mindflight.physics.params import CRAZYFLIE  # noqa: E402

SWARM_SIZES = [1, 16, 256, 1024, 4096, 16384]
TRIALS = 3  # timed runs per row; the median is reported
PROFILE_N = 4096  # swarm size for the "where a GPU step's time goes" profile
DOC_PATH = Path(__file__).resolve().parents[1] / "docs" / "benchmark.md"


@dataclass
class Result:
    n_drones: int
    device: str
    mode: str
    steps_per_s: float | None  # None when this mode could not run
    note: str = ""

    @property
    def drone_steps_per_s(self) -> float | None:
        return None if self.steps_per_s is None else self.n_drones * self.steps_per_s


class CudagraphSkipCatcher(logging.Handler):
    """Collects torch's "skipping cudagraphs due to ..." warnings, so a cudagraphs row can say it fell back to eager."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.reasons: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "skipping cudagraphs" in message:
            reason = message.split("skipping cudagraphs due to")[-1].split("Found from")[0].strip(" .:")
            self.reasons.append(reason)


CUDAGRAPH_LOGGER = logging.getLogger("torch._inductor.cudagraph_utils")
ALL_SIZES = 0  # Result.n_drones for a row that covers every N (a mode that could not run at all)


def cudagraph_skip_note(reason: str) -> str:
    """Plain-words version of torch's reason for not capturing a CUDA graph."""
    if "cpu device" in reason:
        return "graphs not captured, ran eager: step() copies a CPU tensor to the GPU each call"
    return f"graphs not captured, ran eager: {reason}"


def sync(device: str) -> None:
    """Wait for the GPU to finish all queued work, so the clock measures real work. No-op on CPU."""
    if device == "cuda":
        torch.cuda.synchronize()


def time_steps(step_fn, n_drones: int, device: str, min_seconds: float, warmup_steps: int) -> float:
    """Steps per second for `step_fn` on a hovering swarm of `n_drones` on `device`.

    Warm up first (first calls pay for memory allocation, kernel loading and compiling). Then do TRIALS timed runs,
    each repeating batches of steps until at least `min_seconds` have passed, and report the median run, so one
    hiccup from Windows or another program doesn't decide the number.
    """
    state = make_state(n_drones, device, pos=[0.0, 0.0, 1.0], rpm=[CRAZYFLIE.hover_rpm] * 4)
    rpm_cmd = torch.full((n_drones, 4), CRAZYFLIE.hover_rpm, device=device)
    for _ in range(warmup_steps):
        state = step_fn(state, rpm_cmd)
    sync(device)

    rates = []
    for _ in range(TRIALS):
        steps_done, batch = 0, 10
        start_s = time.perf_counter()
        while True:
            for _ in range(batch):
                state = step_fn(state, rpm_cmd)
            steps_done += batch
            sync(device)
            elapsed_s = time.perf_counter() - start_s
            if elapsed_s >= min_seconds:
                break
            batch = min(batch * 2, 10_000)
        rates.append(steps_done / elapsed_s)
    return statistics.median(rates)


def check_matches_eager(step_fn, device: str) -> None:
    """A compiled step must give the same physics as the plain one: compare 100 steps of a tumbling drone."""
    state = make_state(4, device, pos=[0.0, 0.0, 1.0], rpm=[CRAZYFLIE.hover_rpm] * 4)
    rpm_cmd = torch.tensor([[14000.0, 15000.0, 14500.0, 15500.0]] * 4, device=device)
    eager, compiled = state, state.clone()
    for _ in range(100):
        eager, compiled = step(eager, rpm_cmd), step_fn(compiled, rpm_cmd)
    if not torch.allclose(eager, compiled, rtol=1e-4, atol=1e-5):
        raise RuntimeError("compiled step does not match eager step")


def short_error(error: Exception) -> str:
    """First line of an exception, trimmed so it fits in a table cell."""
    line = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
    first_sentence = line.split(". ")[0].rstrip(".") + "."
    return f"{type(error).__name__}: {first_sentence}"[:120].replace("|", "/")


def benchmark(devices: list[str], min_seconds: float) -> list[Result]:
    """Run every (device, mode, N) combination and return the results, printing each as it finishes."""
    modes = [("eager", None), ("compile (inductor)", "inductor")]
    results = []
    skip_catcher = CudagraphSkipCatcher()
    CUDAGRAPH_LOGGER.addHandler(skip_catcher)
    CUDAGRAPH_LOGGER.propagate = False  # we report the reason in the table instead of a page of stack traces
    with torch.inference_mode():  # no autograd bookkeeping: the sim never needs gradients
        for device in devices:
            device_modes = modes + ([("compile (cudagraphs)", "cudagraphs")] if device == "cuda" else [])
            for mode, backend in device_modes:
                torch._dynamo.reset()  # each mode starts with no cached compiled code
                step_fn = step if backend is None else torch.compile(step, backend=backend, dynamic=False)
                warmup = 20 if backend is None else 50
                failed_note = ""
                if backend is not None:
                    try:
                        check_matches_eager(step_fn, device)
                    except Exception as error:  # missing compiler, unsupported op, wrong numbers, ...
                        failed_note = short_error(error)
                if failed_note:  # one row saying why, instead of the same error for every N
                    results.append(Result(ALL_SIZES, device, mode, None, failed_note))
                    print(format_row(results[-1], results), flush=True)
                    continue
                for n_drones in SWARM_SIZES:
                    skip_catcher.reasons.clear()
                    result = Result(n_drones, device, mode, time_steps(step_fn, n_drones, device, min_seconds, warmup))
                    if skip_catcher.reasons:
                        result.note = cudagraph_skip_note(skip_catcher.reasons[0])
                    results.append(result)
                    print(format_row(result, results), flush=True)
    return results


@dataclass
class GpuProfile:
    n_drones: int
    wall_ms_per_step: float  # time the CPU takes to get through one step
    gpu_busy_ms_per_step: float  # time the GPU spends actually running kernels for one step
    kernel_launches_per_step: float


def profile_gpu_step(n_drones: int, steps: int = 100) -> GpuProfile:
    """How busy the GPU really is during an eager step: kernel time vs wall-clock time, from torch.profiler."""
    from torch.profiler import ProfilerActivity, profile

    state = make_state(n_drones, "cuda", pos=[0.0, 0.0, 1.0], rpm=[CRAZYFLIE.hover_rpm] * 4)
    rpm_cmd = torch.full((n_drones, 4), CRAZYFLIE.hover_rpm, device="cuda")
    with torch.inference_mode():
        for _ in range(20):
            state = step(state, rpm_cmd)
        sync("cuda")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            start_s = time.perf_counter()
            for _ in range(steps):
                state = step(state, rpm_cmd)
            sync("cuda")
            wall_s = time.perf_counter() - start_s
    events = prof.key_averages()
    gpu_busy_us = sum(e.self_device_time_total for e in events)
    launches = sum(e.count for e in events if e.key == "cudaLaunchKernel")
    return GpuProfile(n_drones, 1000 * wall_s / steps, gpu_busy_us / 1000 / steps, launches / steps)


def eager_baseline(result: Result, results: list[Result]) -> Result | None:
    """The eager result for the same N and device, to compare a compiled mode against."""
    return next((r for r in results if r.n_drones == result.n_drones and r.device == result.device and r.mode == "eager"), None)


def format_number(value: float) -> str:
    """Readable numbers: 4,135 / 773 k / 2.42 M / 1.05 G."""
    if value >= 1e9:
        return f"{value / 1e9:.2f} G"
    if value >= 1e6:
        return f"{value / 1e6:.2f} M"
    if value >= 1e4:
        return f"{value / 1e3:.0f} k"
    return f"{value:,.0f}"


def format_row(result: Result, results: list[Result]) -> str:
    """One Markdown table row: N | device | mode | steps/s | drone-steps/s | real-time × | vs eager."""
    if result.steps_per_s is None:
        n_label = "all" if result.n_drones == ALL_SIZES else f"{result.n_drones:,}"
        return f"| {n_label} | {result.device} | {result.mode} | – | – | – | not available: {result.note} |"
    baseline = eager_baseline(result, results)
    speedup = "" if result.mode == "eager" or baseline is None else f"{result.steps_per_s / baseline.steps_per_s:.2f}×"
    if result.note:
        speedup = result.note  # it fell back to eager, so a speed-up ratio would only measure noise
    return (
        f"| {result.n_drones:,} | {result.device} | {result.mode} | {format_number(result.steps_per_s)} | "
        f"{format_number(result.drone_steps_per_s)} | {result.steps_per_s * PHYSICS_DT_S:,.1f} | {speedup} |"
    )


TABLE_HEADER = "| N drones | device | mode | steps/s | drone-steps/s | real-time × | vs eager |\n|---:|---|---|---:|---:|---:|---:|"


def cpu_name() -> str:
    """The CPU's marketing name (platform.processor() only gives the family code on Windows)."""
    if sys.platform == "win32":
        import winreg

        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def best(results: list[Result], n_drones: int, device: str) -> Result | None:
    """Fastest mode that really ran for this N and device (not one that silently fell back to eager)."""
    ran = [
        r for r in results
        if r.n_drones == n_drones and r.device == device and r.steps_per_s is not None and not r.note
    ]
    return max(ran, key=lambda r: r.steps_per_s, default=None)


def summary(results: list[Result], gpu_profile: GpuProfile | None) -> str:
    """2–3 sentences on where the GPU beats the CPU and why, worked out from these numbers."""
    pairs = [(n, best(results, n, "cpu"), best(results, n, "cuda")) for n in SWARM_SIZES]
    pairs = [(n, cpu, gpu) for n, cpu, gpu in pairs if cpu and gpu]
    if not pairs:
        return "Only one device ran, so there is no CPU vs GPU comparison."
    gpu_wins = [n for n, cpu, gpu in pairs if gpu.steps_per_s > cpu.steps_per_s]
    n_big, cpu_big, gpu_big = pairs[-1]
    n_small, cpu_small, gpu_small = pairs[0]
    crossover = (
        f"The GPU pulls ahead from about **N = {gpu_wins[0]:,}**"
        if gpu_wins and all(n in gpu_wins for n, _, _ in pairs if n >= gpu_wins[0])
        else "Neither device wins at every size above some N"
    )
    why = (
        "because every step is a few dozen tiny operations and the GPU pays a fixed cost to launch each one, "
        "whatever N is"
    )
    if gpu_profile:
        busy_percent = 100 * gpu_profile.gpu_busy_ms_per_step / gpu_profile.wall_ms_per_step
        why = (
            f"because a step is about {gpu_profile.kernel_launches_per_step:.0f} tiny GPU kernels and the cost is "
            f"launching them, not running them: at N = {gpu_profile.n_drones:,} a step takes "
            f"{gpu_profile.wall_ms_per_step:.2f} ms but the GPU is busy for only {gpu_profile.gpu_busy_ms_per_step:.3f} ms "
            f"({busy_percent:.0f}%)"
        )
    return (
        f"{crossover}: at N = {n_big:,} it does {format_number(gpu_big.drone_steps_per_s)} drone-steps/s against "
        f"the CPU's {format_number(cpu_big.drone_steps_per_s)} ({gpu_big.steps_per_s / cpu_big.steps_per_s:.1f}× faster). "
        f"Below that the CPU wins (N = {n_small:,}: CPU {format_number(cpu_small.steps_per_s)} vs "
        f"GPU {format_number(gpu_small.steps_per_s)} steps/s), {why}. "
        f"That launch cost is the same for 1 drone or 16,384, so the GPU's steps/s stays almost flat as N grows while "
        f"the CPU's time grows with N, and fusing the step into fewer kernels (torch.compile with Triton, or CUDA graphs) "
        f"is the next big speed-up."
    )


def write_doc(results: list[Result], devices: list[str], min_seconds: float, gpu_profile: GpuProfile | None) -> str:
    """Render docs/benchmark.md and return its text."""
    gpu = torch.cuda.get_device_name(0) if "cuda" in devices else "none"
    lines = [
        "# Physics benchmark: how many drones can we fly?",
        "",
        "Generated by `python scripts/benchmark.py`. Times one call of `mindflight.physics.dynamics.step` for the",
        f"whole swarm (dt = {PHYSICS_DT_S * 1000:g} ms), hovering, with no autograd. **drone-steps/s** = N × steps/s.",
        "**real-time ×** = steps/s × dt: at 1.0 the swarm flies as fast as real time. **vs eager** compares a",
        "compiled step with the plain one on the same device and N.",
        "",
        "## Machine",
        "",
        f"- GPU: {gpu}",
        f"- CPU: {cpu_name()} ({torch.get_num_threads()} torch threads)",
        f"- torch {torch.__version__}, CUDA {torch.version.cuda or 'n/a'}, Python {platform.python_version()}, {platform.system()} {platform.release()}",
        f"- Each row: warm-up, then the median of {TRIALS} runs of at least {min_seconds:g} s",
        "",
        "## GPU vs CPU",
        "",
        summary(results, gpu_profile),
        "",
        *(
            [
                "## Where a GPU step's time goes",
                "",
                f"Eager steps at N = {gpu_profile.n_drones:,}. Wall time is from the Results table below; GPU busy time and",
                "kernel launches are from `torch.profiler` over 100 steps.",
                "",
                "| wall time per step | GPU busy per step | GPU busy | kernel launches per step |",
                "|---:|---:|---:|---:|",
                f"| {gpu_profile.wall_ms_per_step:.3f} ms | {gpu_profile.gpu_busy_ms_per_step:.3f} ms | "
                f"{100 * gpu_profile.gpu_busy_ms_per_step / gpu_profile.wall_ms_per_step:.0f}% | "
                f"{gpu_profile.kernel_launches_per_step:.0f} |",
                "",
            ]
            if gpu_profile
            else []
        ),
        "## Results",
        "",
        TABLE_HEADER,
        *(format_row(r, results) for r in results),
        "",
        "## Why `torch.compile` (inductor) may say \"not available\"",
        "",
        "Inductor writes new fused kernels at run time. For CUDA it needs the `triton` package (on Windows:",
        "`pip install triton-windows`), and for CPU it needs a C++ compiler (on Windows: MSVC from the Visual Studio",
        "Build Tools, run from a Developer prompt). Without them the row above records the error instead of a time.",
        "The `cudagraphs` backend needs neither: it records the step's kernels once and replays them with one launch.",
        "It can only do that if `step()` creates no CPU tensors; if the rows above say \"graphs not captured\", that is",
        "why, and those rows are plain eager speed.",
        "",
    ]
    text = "\n".join(lines)
    DOC_PATH.parent.mkdir(exist_ok=True)
    DOC_PATH.write_text(text, encoding="utf-8", newline="\n")
    return text


def main() -> None:
    warnings.filterwarnings("ignore", message="Warning: Profiler clears events")  # harmless, we profile once
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)  # "triton not found": we say so in the table
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="time each row for 0.1 s instead of 1 s")
    parser.add_argument("--cpu-only", action="store_true", help="skip the GPU even if one is available")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # the table uses × and –, which the Windows console code page lacks
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() and not args.cpu_only else [])
    min_seconds = 0.1 if args.quick else 1.0
    print(TABLE_HEADER)
    results = benchmark(devices, min_seconds)
    gpu_profile = profile_gpu_step(PROFILE_N) if "cuda" in devices else None
    timed = next((r for r in results if r.n_drones == PROFILE_N and r.device == "cuda" and r.mode == "eager"), None)
    if gpu_profile and timed:
        # The profiler itself slows the CPU side down, so use the clean timed run for the wall-clock time.
        gpu_profile.wall_ms_per_step = 1000 / timed.steps_per_s
    write_doc(results, devices, min_seconds, gpu_profile)
    print()
    print(summary(results, gpu_profile))
    print(f"\nWrote {DOC_PATH}")


if __name__ == "__main__":
    main()
