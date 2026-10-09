"""Export a trained brain to ONNX: `python -m mindflight.deploy.export policies/waypoint_attitude.pt`.

ONNX is a standard file format for neural networks. Any language can run it (onnxruntime in Python, C++, even a
phone), without PyTorch. The file holds only what flying needs, the *deterministic* action:

    action = actor( clamp((obs − mean) / sqrt(var + 1e-8), −5, 5) )        (RunningNorm, then the actor MLP)

so the observation scaling learned in training travels inside the file. No value network, no random spread.
The task and its options are stored in the file's metadata, so the runner can rebuild the observation from it.

After writing, the export checks itself: onnxruntime and PyTorch must agree to within 1e-5 on random observations.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mindflight.rl.ppo import ActorCritic
from mindflight.rl.train import load_policy

ONNX_OPSET = 17
MAX_ABS_DIFF = 1e-5


class DeterministicActor(nn.Module):
    """obs (B, obs_size) -> mean action (B, 4): the observation scaling plus the actor, nothing else."""

    def __init__(self, model: ActorCritic):
        super().__init__()
        self.obs_norm = model.obs_norm
        self.actor = model.actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.obs_norm(obs))


def random_observations(model: ActorCritic, n: int = 256, seed: int = 0) -> torch.Tensor:
    """n observations spread like real ones (mean ± 3 std of the training data), (n, obs_size), for checking."""
    generator = torch.Generator().manual_seed(seed)
    mean, std = model.obs_norm.mean.cpu(), model.obs_norm.var.cpu().sqrt()
    return mean + 3.0 * std * torch.randn(n, mean.shape[0], generator=generator)


def export_onnx(checkpoint: str | Path, out: str | Path | None = None) -> tuple[Path, float]:
    """Write `checkpoint` (.pt) as ONNX and check it. Returns (path, max |onnx − torch| on 256 random observations).

    out: where to write; default next to the checkpoint with .onnx. Raises if the check fails (> 1e-5).
    """
    import onnx
    import onnxruntime

    checkpoint = Path(checkpoint)
    out = Path(out) if out is not None else checkpoint.with_suffix(".onnx")
    model, meta = load_policy(checkpoint, "cpu")
    actor = DeterministicActor(model).eval()
    obs_size = model.obs_norm.mean.shape[0]
    out.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the TorchScript exporter is "legacy" but needs no extra packages
        torch.onnx.export(
            actor,
            torch.zeros(1, obs_size),
            str(out),
            input_names=["obs"],
            output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=ONNX_OPSET,
            dynamo=False,
        )

    # Remember what this brain was trained on, so the runner can rebuild its observation.
    proto = onnx.load(str(out))
    for key, value in {
        "mindflight.task": meta.get("task", ""),
        "mindflight.env_kwargs": json.dumps(meta.get("env_kwargs", {})),
        "mindflight.source": checkpoint.name,
    }.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(proto, str(out))

    obs = random_observations(model)
    session = onnxruntime.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_action = session.run(None, {"obs": obs.numpy().astype(np.float32)})[0]
    torch_action = model.act(obs, deterministic=True).numpy()
    diff = float(np.abs(onnx_action - torch_action).max())
    if diff > MAX_ABS_DIFF:
        raise RuntimeError(f"ONNX and PyTorch disagree by {diff:.2e} (> {MAX_ABS_DIFF:.0e})")
    return out, diff


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m mindflight.deploy.export", description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", help="a trained policy, e.g. policies/waypoint_attitude.pt")
    parser.add_argument("-o", "--out", default=None, help="output .onnx (default: next to the checkpoint)")
    args = parser.parse_args()
    out, diff = export_onnx(args.checkpoint, args.out)
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print(f"check: ONNX vs PyTorch max difference {diff:.1e} (limit {MAX_ABS_DIFF:.0e})  OK")


if __name__ == "__main__":
    main()
