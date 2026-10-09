"""Draw a run's learning curves to a PNG: `python scripts/plot_run.py runs/hover/<run> docs/img/level1_curves.png`.

Reads metrics.jsonl and plots episode return, success rate (training and test) and episode length against the
number of samples (drone-steps of experience).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]


def plot(runs: dict[str, Path], out: Path, title: str = "") -> None:
    keys = [
        ("episode_return", "episode return"),
        ("episode_success", "success rate (training, with exploration noise)"),
        ("eval_success", "success rate (test: 1,000 fixed starts)"),
        ("episode_length_s", "episode length (s)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for (key, label), ax in zip(keys, axes.flat):
        for name, run_dir in runs.items():
            rows = [r for r in load(run_dir) if key in r]
            if rows:
                ax.plot([r["samples"] / 1e6 for r in rows], [r[key] for r in rows], label=name, marker="o" if key.startswith("eval") else None, markersize=3)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("million samples")
        ax.grid(alpha=0.3)
        if len(runs) > 1:
            ax.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    *run_args, out_path = sys.argv[1:]
    # Each run argument is a folder, or name=folder to label it in the legend.
    runs = {}
    for arg in run_args:
        name, _, folder = arg.rpartition("=")
        runs[name or Path(folder).name] = Path(folder)
    plot(runs, Path(out_path))
