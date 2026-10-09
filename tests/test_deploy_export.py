"""Deploy: the ONNX file gives the same action as the PyTorch brain."""

import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from mindflight.deploy.export import MAX_ABS_DIFF, export_onnx, random_observations  # noqa: E402
from mindflight.deploy.policy import OnnxPolicy, TorchPolicy  # noqa: E402


@pytest.mark.parametrize("name", ["hover_attitude", "waypoint_attitude"])
def test_onnx_matches_pytorch_to_1e_5(tmp_path, name):
    path, diff = export_onnx(f"policies/{name}.pt", tmp_path / f"{name}.onnx")
    assert path.exists() and diff < MAX_ABS_DIFF

    torch_policy, onnx_policy = TorchPolicy(f"policies/{name}.pt"), OnnxPolicy(path)
    assert onnx_policy.task == torch_policy.task  # metadata travels inside the file
    assert onnx_policy.env_kwargs == torch_policy.env_kwargs == {"action_mode": "attitude"}
    obs = random_observations(torch_policy.model, n=64, seed=7)
    for row in obs.split(1):  # the runner calls it one observation at a time
        assert torch.allclose(onnx_policy(row), torch_policy(row), atol=MAX_ABS_DIFF)


def test_onnx_without_metadata_is_refused(tmp_path):
    import onnx

    path, _ = export_onnx("policies/hover_attitude.pt", tmp_path / "x.onnx")
    proto = onnx.load(str(path))
    del proto.metadata_props[:]
    onnx.save(proto, str(path))
    with pytest.raises(ValueError, match="no MindFlight metadata"):
        OnnxPolicy(path)
