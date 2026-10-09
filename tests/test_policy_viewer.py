import json

import numpy as np
import torch
from fastapi.testclient import TestClient

from mindflight.envs import make_env
from mindflight.rl.ppo import ActorCritic
from mindflight.viewer.policy_source import PolicySource, grid_offsets
from mindflight.viewer.server import create_app


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make_source(task="hover", n=4, layout="grid", **env_kwargs):
    env = make_env(task, n, device="cpu", seed=0, **env_kwargs)
    model = ActorCritic(env.obs_size, env.action_size)
    clock = FakeClock()
    return PolicySource(env, model, layout=layout, spacing_m=3.0, clock=clock), clock


def test_grid_offsets_are_centred_and_spaced():
    offsets = grid_offsets(4, 3.0)
    assert offsets[:, :2].tolist() == [[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]]
    assert torch.all(offsets[:, 2] == 0)


def test_frame_runs_real_time_policy_steps_and_offsets_poses():
    source, clock = make_source()
    clock.now = 0.1  # 0.1 s of wall time = 5 policy steps of 0.02 s
    poses = source.frame(0.0)
    assert source.env.step_count.tolist() == [5, 5, 5, 5]
    assert poses.shape == (4, 7) and poses.dtype == np.float32
    world = source.env.poses()[:, :3].numpy()
    assert np.allclose(poses[:, :3], world + source.offsets.numpy())


def test_catch_up_is_capped():
    source, clock = make_source()
    clock.now = 10.0  # a 10 s stall
    source.frame(0.0)
    assert source.env.step_count.max().item() <= 5  # at most 0.1 s of catch-up


def test_scene_has_targets_for_hover_and_gates_for_gates():
    source, _ = make_source("hover")
    scene = source.scene()
    assert len(scene["targets"]) == 4
    target0 = (source.env.target[0] + source.offsets[0]).tolist()
    assert scene["targets"][0] == [round(v, 3) for v in target0]

    source, _ = make_source("gates", n=3, layout="shared", difficulty=1.0, shared_course=True)
    scene = source.scene()
    assert len(scene["gates"]) == 6  # one shared course, drawn once
    assert scene["gates"][0]["state"] == "next" and scene["gates"][1]["state"] == "later"
    assert scene["gates"][0]["size"] == 0.5
    # All 3 drones really do have the same course.
    assert torch.equal(source.env.gate_centre[0], source.env.gate_centre[2])


def test_set_policy_swaps_weights():
    source, _ = make_source()
    other = ActorCritic(source.env.obs_size, source.env.action_size)
    source.set_policy(other.state_dict())
    for a, b in zip(source.models[0].parameters(), other.parameters()):
        assert torch.equal(a, b)


def test_websocket_sends_scene_info_charts_and_metrics_once():
    source, _ = make_source()
    source.info_lines = ["hello"]
    source.chart_series = [{"key": "episode_return", "label": "return"}]
    source.metric_points.append({"iteration": 1, "episode_return": 2.0})
    with TestClient(create_app(source)) as client, client.websocket_connect("/ws") as ws:
        texts = []
        binaries = 0
        while binaries < 3:  # three frames' worth of messages
            message = ws.receive()
            if message.get("bytes") is not None:
                binaries += 1
            else:
                texts.append(json.loads(message["text"]))
    types = [t["type"] for t in texts]
    assert types.count("info") == 1 and types.count("charts") == 1 and types.count("metrics") == 1
    assert "scene" in types
    metrics = next(t for t in texts if t["type"] == "metrics")
    assert metrics["points"] == [{"iteration": 1, "episode_return": 2.0}]


def test_general_shapes_are_sent_with_offsets_and_colours():
    source, _ = make_source("hover", n=2)
    env = source.env
    spheres = torch.zeros(2, 2, 4)
    spheres[:, 0] = torch.tensor([1.0, 2.0, 3.0, 0.1])  # slot 0 drawn, slot 1 radius 0 = skipped
    lines = torch.zeros(2, 1, 6)
    lines[:, 0] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    env.display_objects = lambda: {"spheres": spheres, "spheres_colors": ["#ff0000", "#00ff00"], "lines": lines}
    scene = source.scene()
    assert len(scene["spheres"]) == 2 and len(scene["lines"]) == 2
    off = source.offsets[1].tolist()
    assert scene["spheres"][1][:3] == [round(1.0 + off[0], 3), round(2.0 + off[1], 3), round(3.0 + off[2], 3)]
    assert scene["spheres"][0][4] == "#ff0000"
    assert scene["lines"][0][6] == "#c8d0da"  # default colour
