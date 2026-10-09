"""The brain that flies: a PyTorch checkpoint (.pt) or an exported ONNX file (.onnx), behind one small interface.

Both give the *deterministic* action (the mean, no exploration noise) for an observation, and both know which task
and task options they were trained with, so the runner can build the observation exactly like training did.
"""

import json
from pathlib import Path

import numpy as np
import torch

from mindflight.rl.train import load_policy


class FlightPolicy:
    """obs (1, obs_size) tensor -> action (1, 4) tensor in [−1, 1] (clamped here, as the env does)."""

    task: str
    env_kwargs: dict
    source: str

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TorchPolicy(FlightPolicy):
    def __init__(self, path: str | Path):
        self.model, checkpoint = load_policy(path, "cpu")
        self.task = checkpoint["task"]
        self.env_kwargs = dict(checkpoint.get("env_kwargs", {}))
        self.source = f"{Path(path).name} (PyTorch)"

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model.act(obs, deterministic=True).clamp(-1.0, 1.0)


class OnnxPolicy(FlightPolicy):
    def __init__(self, path: str | Path):
        import onnxruntime

        self.session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        if "mindflight.task" not in meta:
            raise ValueError(f"{path} has no MindFlight metadata: export it with python -m mindflight.deploy.export")
        self.task = meta["mindflight.task"]
        self.env_kwargs = json.loads(meta.get("mindflight.env_kwargs", "{}"))
        self.source = f"{Path(path).name} (ONNX)"

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        action = self.session.run(None, {"obs": obs.numpy().astype(np.float32)})[0]
        return torch.from_numpy(action).clamp(-1.0, 1.0)


def load_flight_policy(path: str | Path, use_onnx: bool = False) -> FlightPolicy:
    """A .onnx file -> OnnxPolicy. A .pt checkpoint -> TorchPolicy, or with use_onnx=True: export it first (to
    runs/onnx/, checked to 1e-5) and fly the ONNX file, exactly as a robot without PyTorch would."""
    path = Path(path)
    if path.suffix == ".onnx":
        return OnnxPolicy(path)
    if use_onnx:
        from mindflight.deploy.export import export_onnx

        onnx_path, _ = export_onnx(path, Path("runs") / "onnx" / f"{path.stem}.onnx")
        return OnnxPolicy(onnx_path)
    return TorchPolicy(path)
