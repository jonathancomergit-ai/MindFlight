import math

import pytest
import torch

from mindflight.envs import make_env
from mindflight.physics.controller import (
    attitude_action_to_setpoint,
    controller_step,
    mix,
)
from mindflight.physics.dynamics import make_state, step
from mindflight.physics.motors import rotor_wrench
from mindflight.physics.params import CRAZYFLIE
from mindflight.physics.rotations import quat_to_euler

P = CRAZYFLIE


def test_mixing_gives_back_the_thrust_and_torques_asked_for():
    thrust = torch.tensor([P.weight_n, 1.3 * P.weight_n])
    torque = torch.tensor([[2e-4, -1e-4, 3e-5], [-1e-4, 2e-4, -2e-5]])
    rpm = mix(thrust, torque)
    thrust_body, torque_body = rotor_wrench(rpm)
    assert torch.allclose(thrust_body[:, 2], thrust, rtol=1e-5)
    assert torch.allclose(torque_body, torque, rtol=1e-4, atol=1e-9)


def test_mixing_signs_match_the_motor_table():
    rpm = mix(torch.tensor([P.weight_n] * 3), torch.tensor([[1e-4, 0, 0], [0, 1e-4, 0], [0, 0, 1e-5]]))
    m0, m1, m2, m3 = rpm.unbind(1)
    assert (m2[0] > m0[0]) and (m3[0] > m1[0])  # +roll: left motors (m2, m3) faster
    assert (m1[1] > m0[1]) and (m2[1] > m3[1])  # +pitch: back motors (m1, m2) faster
    assert (m1[2] > m0[2]) and (m3[2] > m2[2])  # +yaw: CW motors (m1, m3) faster


def step_response(axis: int, target: float, seconds: float = 0.8):
    state = make_state(1, "cpu", pos=[0, 0, 2], rpm=[P.hover_rpm] * 4)
    integral = torch.zeros(1, 3)
    setpoint = torch.tensor([[P.weight_n, 0.0, 0.0, 0.0]])
    setpoint[0, axis] = target
    trace = []
    for _ in range(round(seconds / 0.002)):
        rpm, integral = controller_step(state, setpoint, integral, 0.002)
        state = step(state, rpm)
        trace.append(quat_to_euler(state[:, 3:7])[0, axis - 1].item() if axis < 3 else state[0, 12].item())
    return torch.tensor(trace)


@pytest.mark.parametrize("axis", [1, 2])
def test_roll_and_pitch_steps_settle_quickly_without_much_overshoot(axis):
    target = math.radians(20)
    trace = step_response(axis, target)
    outside = (trace - target).abs() > 0.1 * target
    settle_s = (outside.nonzero()[-1].item() + 1) * 0.002
    assert settle_s < 0.3
    assert trace.max() < 1.2 * target
    assert trace[-1] == pytest.approx(target, rel=0.02)


def test_yaw_rate_is_tracked():
    trace = step_response(3, math.radians(90), seconds=0.6)
    assert trace[-1] == pytest.approx(math.radians(90), rel=0.05)


def test_action_scaling():
    sp = attitude_action_to_setpoint(torch.tensor([[0.0, 1.0, -1.0, 0.5], [1.0, 0, 0, 0], [-1.0, 0, 0, 0]]))
    assert sp[0, 0].item() == pytest.approx(P.weight_n)
    assert sp[0, 1:].tolist() == pytest.approx([math.radians(30), math.radians(-30), math.radians(90)])
    assert sp[1, 0].item() == pytest.approx(P.max_command_thrust_n)  # full: the most the radio can command
    assert P.max_command_thrust_n / P.weight_n == pytest.approx(1.66, abs=0.01)
    assert sp[2, 0].item() == 0.0  # clamped at zero


def place_level(env, height=1.0):
    env.reset()
    env.state[:] = 0.0
    env.state[:, 2] = height
    env.state[:, 3] = 1.0
    env.state[:, 13:17] = P.hover_rpm


def test_attitude_mode_commands_move_the_drone_the_right_way():
    env = make_env("hover", 3, device="cpu", action_mode="attitude")
    place_level(env)
    # drone 0: roll right (+roll tips the right side down) -> drifts to −y
    # drone 1: pitch nose-down (+pitch) -> goes forward, +x
    # drone 2: more thrust -> climbs
    action = torch.tensor([[0.0, 0.3, 0.0, 0.0], [0.0, 0.0, 0.3, 0.0], [0.3, 0.0, 0.0, 0.0]])
    for _ in range(25):  # 0.5 s
        env.step(action)
    assert env.state[0, 1] < -0.02 and env.state[1, 0] > 0.02 and env.state[2, 2] > 1.05


def test_zero_action_hovers_in_attitude_mode():
    env = make_env("hover", 2, device="cpu", action_mode="attitude")
    place_level(env)
    for _ in range(100):
        env.step(torch.zeros(2, 4))
    assert (env.state[:, 2] - 1.0).abs().max() < 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_graph_path_matches_the_cpu_loop():
    torch.manual_seed(0)
    action = 0.3 * torch.randn(64, 4)
    envs = {}
    for device in ("cpu", "cuda"):
        env = make_env("hover", 64, device=device, action_mode="attitude", seed=3)
        place_level(env)
        for _ in range(20):
            env.step(action.to(device))
        envs[device] = env
    assert torch.allclose(envs["cpu"].state, envs["cuda"].state.cpu(), atol=2e-4)
    assert torch.allclose(envs["cpu"].controller_integral, envs["cuda"].controller_integral.cpu(), atol=2e-4)


def test_unknown_action_mode_is_an_error():
    with pytest.raises(ValueError):
        make_env("hover", 1, device="cpu", action_mode="joystick")
