"""Deploy: the whole flight loop against the fake Crazyflie. Missions, every safety stop, calibration, the shadow env.

These fly the exact runner code that will fly the real drone; only the link is fake.
"""

import json

import pytest
import torch

from mindflight.deploy import runner as runner_module
from mindflight.deploy.interface import Estimate
from mindflight.deploy.policy import OnnxPolicy, TorchPolicy
from mindflight.deploy.runner import Fence, FlightRunner, Mission, Safety, ShadowEnv, calibrate_hover
from mindflight.deploy.sitl import SitlConfig, SitlCrazyflie
from mindflight.envs import make_env
from mindflight.physics.params import CRAZYFLIE

WAYPOINTS = [(0.5, 0.0, 0.5), (0.5, 0.5, 0.8), (0.0, 0.0, 0.6)]


def sitl(**config) -> SitlCrazyflie:
    link = SitlCrazyflie(config=SitlConfig(**config))
    link.connect()
    return link


def fly(link, policy, tmp_path, mission=None, **kwargs):
    runner = FlightRunner(link, policy, mission or Mission(hover_s=1.0), log_dir=tmp_path, print_fn=lambda s: None, **kwargs)
    return runner.fly()


def assert_landed_and_off(link, result):
    link.advance(0.3)
    assert link.position_m[2] < 0.02  # on the floor
    assert not link.motors_on
    assert result.final_position_m[2] < 0.08


@pytest.fixture(scope="module")
def hover_policy():
    return TorchPolicy("policies/hover_attitude.pt")


def test_full_mission_with_the_hover_brain(tmp_path, hover_policy):
    link = sitl(seed=1)
    result = fly(link, hover_policy, tmp_path, Mission(hover_s=1.0, waypoints=WAYPOINTS))
    assert result.outcome == "landed" and result.reason == "mission complete"
    assert result.waypoints_reached == [True, True, True]
    assert max(result.waypoint_closest_m) < 0.15
    assert result.max_tilt_deg < 35
    assert_landed_and_off(link, result)
    lines = [json.loads(line) for line in open(result.log_path, encoding="utf-8")]
    assert lines[0]["type"] == "start" and lines[-1]["type"] == "end"
    ticks = [line for line in lines if line["type"] == "tick"]
    assert len(ticks) > 200 and {"pos", "target", "action", "command"} <= set(ticks[0])


def test_full_mission_with_the_waypoint_brain_as_onnx(tmp_path):
    pytest.importorskip("onnxruntime")
    from mindflight.deploy.export import export_onnx

    onnx_path, _ = export_onnx("policies/waypoint_attitude.pt", tmp_path / "waypoint.onnx")
    link = sitl(seed=2)
    result = fly(link, OnnxPolicy(onnx_path), tmp_path, Mission(hover_s=1.0, waypoints=WAYPOINTS))
    assert result.outcome == "landed" and result.waypoints_reached == [True, True, True]
    assert max(result.waypoint_closest_m) < 0.15
    assert_landed_and_off(link, result)


def test_leaving_the_geofence_lands_inside_it(tmp_path, hover_policy):
    link = sitl(seed=3)
    fence = Fence(x_m=(-1.0, 0.4), y_m=(-1.0, 1.0), z_max_m=1.5)

    pushed = []

    def push(runner, t, estimate):
        if t > 2.5 and not pushed:
            pushed.append(t)
            link.kick((2.0, 0.0, 0.0))  # a gust shoves it toward +x, past the fence at 0.4 m

    result = fly(link, hover_policy, tmp_path, Mission(hover_s=4.0, fence=fence), on_tick=push)
    assert result.outcome == "landed" and result.reason.startswith("geofence")
    assert_landed_and_off(link, result)
    assert link.position_m[0] < 0.4  # came back inside before touching down


def test_kill_stops_the_motors_within_one_cycle(tmp_path, hover_policy):
    link = sitl(seed=4)
    killed_at = []

    def kill(runner, t, estimate):
        if t > 2.0 and not killed_at:
            killed_at.append(link.now())
            runner.request_kill("test")

    result = fly(link, hover_policy, tmp_path, on_tick=kill)
    assert result.outcome == "killed"
    # Motors off is sent the moment the kill is pulled. Allow one 50 Hz cycle plus the longest radio delay:
    deadline = killed_at[0] + 0.02 + link.config.uplink_delay_s[1]
    assert any(thrust == 0 for t, thrust in link.received if killed_at[0] < t <= deadline)
    assert all(thrust == 0 for t, thrust in link.received if t > deadline)  # nothing with thrust ever again
    link.advance(deadline - link.now())
    assert not link.motors_on


def test_link_loss_stops_the_runner_and_the_drone_cuts_its_motors_after_2_s(tmp_path, hover_policy):
    link = sitl(seed=5)
    cut_at = []

    def cut(runner, t, estimate):
        if t > 2.5 and not cut_at:
            cut_at.append(link.now())
            link.set_link(False)

    result = fly(link, hover_policy, tmp_path, Mission(hover_s=4.0), on_tick=cut)
    assert result.outcome == "link_lost"
    assert link.now() - cut_at[0] < 0.5  # the runner noticed fast
    last_packet_s = max(t for t, _ in link.received)
    link.advance(last_packet_s + 1.5 - link.now())
    assert link.motors_on and link.supervisor == "level_out"  # 0.5–2 s: level, keep thrust
    link.advance(0.6)
    assert not link.motors_on and link.supervisor == "locked"  # > 2 s: motors off, locked


def test_low_battery_lands(tmp_path, hover_policy):
    link = sitl(seed=6, battery_start_v=3.55, battery_sag_v=0.4)  # rests above 3.5 V; sags below 3.2 V in flight
    result = fly(link, hover_policy, tmp_path, Mission(hover_s=6.0))
    assert result.outcome == "landed" and result.reason.startswith("low battery")
    assert result.duration_s < 8.0
    assert_landed_and_off(link, result)


class Sabotaged:
    """Wraps a brain and spoils its action after `after_s` seconds of flight (the runner calls it at 50 Hz)."""

    def __init__(self, policy, after_s, spoil):
        self.policy, self.after_steps, self.spoil, self.steps = policy, round(after_s * 50), spoil, 0
        self.task, self.env_kwargs, self.source = policy.task, policy.env_kwargs, "sabotaged"

    def __call__(self, obs):
        self.steps += 1
        action = self.policy(obs)
        return self.spoil(action) if self.steps > self.after_steps else action


def test_nan_from_the_brain_stops_the_motors(tmp_path, hover_policy):
    link = sitl(seed=7)
    nan_brain = Sabotaged(hover_policy, 2.0, lambda a: torch.full_like(a, float("nan")))
    result = fly(link, nan_brain, tmp_path)
    assert result.outcome == "stopped" and "NaN" in result.reason
    link.advance(0.05)
    assert not link.motors_on


def test_too_much_tilt_stops_the_motors(tmp_path, hover_policy):
    link = sitl(seed=8)

    def full_roll(action):
        action = action.clone()
        action[:, 1] = 1.0  # 30° roll, held
        return action

    result = fly(link, Sabotaged(hover_policy, 2.0, full_roll), tmp_path, safety=Safety(max_tilt_deg=20))
    assert result.outcome == "stopped" and result.reason.startswith("tilt")
    link.advance(0.05)
    assert not link.motors_on


def test_refuses_to_take_off_from_the_air_or_with_a_flat_battery(tmp_path, hover_policy):
    assert fly(sitl(start_position_m=(0.0, 0.0, 1.0)), hover_policy, tmp_path).outcome == "refused"
    flat = fly(sitl(battery_start_v=3.3), hover_policy, tmp_path)
    assert flat.outcome == "refused" and "battery" in flat.reason


def test_mission_outside_its_fence_is_rejected(hover_policy):
    with pytest.raises(ValueError, match="fence"):
        Mission(waypoints=[(2.0, 0.0, 1.0)]).check()


def test_mission_file_is_read(tmp_path):
    path = tmp_path / "mission.json"
    path.write_text(json.dumps({"takeoff_height_m": 0.4, "waypoints": [[0.3, 0, 0.4]], "fence": {"x_m": [-1, 1]}}))
    mission = Mission.from_json(path)
    assert mission.takeoff_height_m == 0.4 and mission.waypoints == [(0.3, 0, 0.4)]
    assert mission.fence.x_m == (-1, 1) and mission.fence.y_m == (-1.5, 1.5)
    mission.check()


@pytest.mark.parametrize("scale", [0.85, 1.15])
def test_calibrate_hover_finds_the_true_hover_command(tmp_path, hover_policy, scale):
    link = sitl(seed=9, thrust_scale=scale)  # the real drone is 15 % weaker / stronger than the model
    hover, result = calibrate_hover(
        lambda m: FlightRunner(link, hover_policy, m, log_dir=tmp_path, print_fn=lambda s: None), hover_s=5.0
    )
    assert result.outcome == "landed"
    assert hover == pytest.approx(link.true_hover_command, rel=0.03)


def test_shadow_env_builds_the_training_observation_exactly():
    env = make_env("waypoint", 1, device="cpu", seed=3, action_mode="attitude")
    env.reset()
    env.step(torch.tensor([[0.1, -0.2, 0.3, 0.05]]))
    s = env.state[0].tolist()
    estimate = Estimate(0.0, 0.0, 0, tuple(s[0:3]), tuple(s[7:10]), tuple(s[3:7]), tuple(s[10:13]), 4.0)
    shadow = ShadowEnv("waypoint", {"action_mode": "attitude"}, CRAZYFLIE)
    obs = shadow.observe(estimate, tuple(env.target[0].tolist()), env.action)
    assert torch.allclose(obs, env.observe(), atol=1e-6)


def test_shadow_env_switches_off_simulated_sensor_noise_and_delay(monkeypatch):
    seen = {}

    def fake_make_env(task, n, device=None, seed=0, **kwargs):
        seen.update(kwargs)
        kwargs = {k: v for k, v in kwargs.items() if k not in ("sensors", "real_world")}  # not on main yet
        return make_env(task, n, device=device, seed=seed, **kwargs)

    monkeypatch.setattr(runner_module, "make_env", fake_make_env)
    ShadowEnv("hover", {"action_mode": "attitude", "sensors": "flowdeck", "real_world": True}, CRAZYFLIE)
    assert seen["sensors"] == "ideal" and seen["real_world"] is False


def test_motor_brains_are_refused():
    with pytest.raises(ValueError, match="attitude"):
        ShadowEnv("hover", {}, CRAZYFLIE)
