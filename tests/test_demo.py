"""The live demo (python -m mindflight.demo): the 4 groups do what the legend says, and the viewer plumbing works."""

import sys

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from mindflight.demo import GROUPS, FourGroupDemo
from mindflight.physics.dynamics import PHYSICS_DT_S
from mindflight.viewer import __main__ as viewer_main
from mindflight.viewer.server import create_app
from mindflight.viewer.sim_source import SimSource

HOVER, OFF, TIP, SPIN = range(4)


class FakeClock:
    """A wall clock the test moves by hand, in seconds."""

    def __init__(self):
        self.now_s = 0.0

    def __call__(self) -> float:
        return self.now_s


def make_demo(n=64, clock=None):
    demo = FourGroupDemo(n)
    source = SimSource(demo.start_state, demo.step, clock=clock or FakeClock())
    return demo, source


@pytest.fixture(scope="module")
def after_one_second():
    """Poses at the start and after 1 s of sim time (500 steps), plus which group each drone is in."""
    demo, source = make_demo()
    start = source.poses()
    source.advance(1.0)
    return start, source.poses(), demo.group_of_drone


def test_groups_split_evenly_and_sit_apart():
    demo = FourGroupDemo(64)
    assert demo.group_count == [16, 16, 16, 16]
    start = demo.start_pos_m.numpy()
    assert np.all(start[:, 2] == 1.0)  # everyone starts at 1 m
    # Each group sits in its own corner: the signs of x and y match the group's corner.
    for g, group in enumerate(GROUPS):
        in_group = demo.group_of_drone == g
        assert np.all(np.sign(start[in_group, 0]) == group.corner[0])
        assert np.all(np.sign(start[in_group, 1]) == group.corner[1])


def test_hover_group_stays_within_1_cm(after_one_second):
    start, now, group = after_one_second
    moved_m = np.linalg.norm(now[group == HOVER, :3] - start[group == HOVER, :3], axis=1)
    assert moved_m.max() < 0.01


def test_off_group_falls_onto_the_floor(after_one_second):
    start, now, group = after_one_second
    # z = 1 − ½ g t² reaches 0 at t = 0.45 s, so after 1 s they are all lying on the floor, where they fell.
    assert np.all(now[group == OFF, 2] == 0.0)
    assert np.allclose(now[group == OFF, :2], start[group == OFF, :2], atol=1e-4)  # straight down
    assert np.allclose(now[group == OFF, 3:7], [1, 0, 0, 0], atol=1e-5)  # still level


def test_tip_group_rolls_and_drifts_toward_minus_y(after_one_second):
    start, now, group = after_one_second
    tip = group == TIP
    drift_m = now[tip, :3] - start[tip, :3]
    assert np.all(drift_m[:, 1] < -0.5)  # well toward −y
    assert np.allclose(drift_m[:, 0], 0.0, atol=1e-3)  # no pitch, so no drift along x
    # It rolled: the quaternion turned about the body x axis only (qx big, qy = qz = 0).
    quat = now[tip, 3:7]
    assert np.all(np.abs(quat[:, 1]) > 0.5)
    assert np.allclose(quat[:, 2:4], 0.0, atol=1e-4)


def test_spin_group_yaws_in_place(after_one_second):
    start, now, group = after_one_second
    spin = group == SPIN
    assert np.allclose(now[spin, :3], start[spin, :3], atol=1e-3)  # no drift in x, y or z
    # Pure yaw: quaternion (cos(ψ/2), 0, 0, sin(ψ/2)), and ψ has grown a lot (about 6 rad after 1 s).
    quat = now[spin, 3:7]
    assert np.allclose(quat[:, 1:3], 0.0, atol=1e-4)
    yaw_rad = 2 * np.arctan2(quat[:, 3], quat[:, 0])
    assert np.all(np.abs(yaw_rad) > 0.1)


def test_spin_group_turns_counter_clockwise_at_first():
    # Faster CW props push the body CCW (positive yaw about +z). Check early, before ψ wraps past π.
    demo, source = make_demo()
    source.advance(0.2)
    quat = source.poses()[demo.group_of_drone == SPIN, 3:7]
    assert np.all(quat[:, 3] * quat[:, 0] > 0)  # sin(ψ/2)·cos(ψ/2) > 0 means ψ > 0


def test_crashed_drones_lie_still():
    demo, source = make_demo()
    source.advance(1.0)
    landed = source.poses()
    source.advance(0.5)
    tip_and_off = np.isin(demo.group_of_drone, [OFF, TIP])
    assert np.allclose(source.poses()[tip_and_off], landed[tip_and_off])
    assert torch.all(source.state[torch.as_tensor(tip_and_off), 10:17] == 0)  # no spin, props stopped


def test_frame_follows_the_wall_clock_and_caps_catch_up():
    clock = FakeClock()
    _, source = make_demo(clock=clock)
    clock.now_s += 1 / 30  # one viewer frame of wall time
    source.frame(0.0)
    assert source.sim_time_s == pytest.approx(1 / 30, abs=PHYSICS_DT_S)
    clock.now_s += 10.0  # a 10 s stall: only max_catch_up_s (0.1 s) of sim time is run
    source.frame(0.0)
    assert source.sim_time_s == pytest.approx(1 / 30 + source.max_catch_up_s, abs=PHYSICS_DT_S)


def test_restart_after_s_starts_over():
    demo = FourGroupDemo(8)
    source = SimSource(demo.start_state, demo.step, restart_after_s=0.5, clock=FakeClock())
    start = source.poses()
    source.advance(0.5)
    assert source.sim_time_s == 0.0
    assert np.array_equal(source.poses(), start)


def test_websocket_frames_are_n_by_7_float32_unit_quaternions():
    _, source = make_demo(n=64, clock=None)
    source.advance(1.0)  # mid-flight: things have fallen, rolled and spun
    with TestClient(create_app(source)) as client, client.websocket_connect("/ws") as websocket:
        message = websocket.receive_bytes()
    assert len(message) == 64 * 7 * 4
    poses = np.frombuffer(message, dtype="<f4").reshape(64, 7)
    assert np.all(np.isfinite(poses))
    assert np.allclose(np.linalg.norm(poses[:, 3:7], axis=1), 1.0, atol=1e-5)


def test_reset_endpoint_puts_drones_back_and_config_lists_groups():
    demo, source = make_demo()
    start = source.poses()
    client = TestClient(create_app(source, extra_config=demo.config(), on_reset=source.reset))

    config = client.get("/config").json()
    assert config["resettable"] is True
    assert [g["name"] for g in config["groups"]] == [g.name for g in GROUPS]
    assert sum(g["count"] for g in config["groups"]) == config["n_drones"] == 64

    source.advance(1.0)
    assert not np.array_equal(source.poses(), start)
    assert client.post("/reset").json() == {"ok": True}
    assert source.sim_time_s == 0.0
    assert np.array_equal(source.poses(), start)


def test_no_reset_endpoint_without_a_reset_function():
    _, source = make_demo(n=4)
    client = TestClient(create_app(source))
    assert client.get("/config").json()["resettable"] is False
    assert client.post("/reset").status_code in (404, 405)


def test_viewer_without_fake_points_to_the_demo(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["python -m mindflight.viewer"])
    with pytest.raises(SystemExit):
        viewer_main.main()
    assert "python -m mindflight.demo" in capsys.readouterr().err
