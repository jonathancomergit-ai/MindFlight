import math

import pytest
import torch

from mindflight.physics.motors import MOTOR_POSITIONS, MOTOR_SPIN, motor_step, rotor_wrench
from mindflight.physics.params import CRAZYFLIE

DT_S = 0.002  # physics step from PLAN.md (500 Hz)
DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")),
]


def hover_rpm(n: int, device: str = "cpu") -> torch.Tensor:
    return torch.full((n, 4), CRAZYFLIE.hover_rpm, dtype=torch.float32, device=device)


def test_layout_matches_plan_table():
    d = CRAZYFLIE.arm_xy_m
    expected = torch.tensor([[+d, -d], [-d, -d], [-d, +d], [+d, +d]])
    assert torch.allclose(MOTOR_POSITIONS, expected)
    assert MOTOR_SPIN.tolist() == [1.0, -1.0, 1.0, -1.0]  # m0 CCW, m1 CW, m2 CCW, m3 CW


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", [1, 4096])
def test_hover_rpm_thrust_equals_weight_and_no_torque(n, device):
    thrust, torque = rotor_wrench(hover_rpm(n, device))
    assert thrust.shape == (n, 3) and torque.shape == (n, 3)
    assert thrust.dtype == torch.float32 and thrust.device.type == device
    assert torch.all(thrust[:, :2] == 0)
    rel_error = (thrust[:, 2].double() - CRAZYFLIE.weight_n).abs() / CRAZYFLIE.weight_n
    assert rel_error.max().item() < 1e-6
    assert torch.allclose(torque, torch.zeros_like(torque), atol=1e-12)


def spin_up(motors: list[int], extra_rpm: float = 1000.0) -> tuple[torch.Tensor, torch.Tensor]:
    rpm = hover_rpm(1)
    rpm[:, motors] += extra_rpm
    return rotor_wrench(rpm)


def test_left_motors_faster_rolls_right():
    # More lift on the left (+y) side lifts it, so the drone tips toward -y: positive roll about +x.
    _, torque = spin_up([2, 3])
    assert torque[0, 0] > 0
    assert abs(torque[0, 1]) < 1e-12 and abs(torque[0, 2]) < 1e-12


def test_front_motors_faster_pitches_nose_up():
    # More lift at the front (+x) raises the nose. Rotating about +y turns +x toward -z, so nose up is tau_y < 0.
    _, torque = spin_up([0, 3])
    assert torque[0, 1] < 0
    assert abs(torque[0, 0]) < 1e-12 and abs(torque[0, 2]) < 1e-12


def test_cw_motors_faster_yaws_ccw():
    # CW props push the body CCW (seen from above), which is positive yaw about +z.
    _, torque = spin_up([1, 3])
    assert torque[0, 2] > 0
    assert abs(torque[0, 0]) < 1e-12 and abs(torque[0, 1]) < 1e-12


def test_torques_match_plan_formulas():
    rpm = torch.tensor([[12_000.0, 15_000.0, 17_000.0, 20_000.0]])
    f = CRAZYFLIE.kf_n_per_rpm2 * rpm[0].double() ** 2
    q = CRAZYFLIE.km_nm_per_rpm2 * rpm[0].double() ** 2
    d = CRAZYFLIE.arm_xy_m
    expected = torch.tensor(
        [d * (-f[0] - f[1] + f[2] + f[3]), d * (-f[0] + f[1] + f[2] - f[3]), -q[0] + q[1] - q[2] + q[3]]
    )
    thrust, torque = rotor_wrench(rpm)
    assert torch.allclose(torque[0].double(), expected, rtol=1e-5, atol=0)
    assert math.isclose(thrust[0, 2].item(), f.sum().item(), rel_tol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_lag_reaches_63_percent_after_one_time_constant(device):
    start_rpm, target_rpm = 10_000.0, 20_000.0
    rpm = torch.full((4096, 4), start_rpm, device=device)
    cmd = torch.full((4096, 4), target_rpm, device=device)
    steps = round(CRAZYFLIE.motor_time_constant_s / DT_S)
    assert math.isclose(steps * DT_S, CRAZYFLIE.motor_time_constant_s)
    for _ in range(steps):
        rpm = motor_step(rpm, cmd, DT_S)
    fraction_reached = (rpm - start_rpm) / (target_rpm - start_rpm)
    assert torch.all((fraction_reached - (1 - 1 / math.e)).abs() < 0.005)


def test_lag_never_jumps_and_settles_on_command():
    rpm = torch.zeros(1, 4)
    cmd = torch.full((1, 4), CRAZYFLIE.hover_rpm)
    rpm = motor_step(rpm, cmd, DT_S)
    assert torch.all(rpm < 0.2 * CRAZYFLIE.hover_rpm)  # one 2 ms step closes only ~9.5 % of the gap
    for _ in range(1000):  # 2 s = 100 time constants
        rpm = motor_step(rpm, cmd, DT_S)
    assert torch.allclose(rpm, cmd, rtol=1e-5)


def test_commands_are_clamped_to_motor_limits():
    rpm = hover_rpm(2)
    cmd = torch.tensor([[-5_000.0] * 4, [1e6] * 4])
    for _ in range(2000):  # long enough to settle at whatever the motor can reach
        rpm = motor_step(rpm, cmd, DT_S)
        assert torch.all(rpm >= 0) and torch.all(rpm <= CRAZYFLIE.max_rpm)
    assert torch.allclose(rpm[0], torch.zeros(4), atol=1e-3)
    assert torch.allclose(rpm[1], torch.full((4,), CRAZYFLIE.max_rpm), rtol=1e-5)
