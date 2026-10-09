import math

import pytest
import torch

from mindflight.physics.dynamics import OMEGA, POS, QUAT, RPM, VEL, make_state, step
from mindflight.physics.params import CRAZYFLIE, GRAVITY_MPS2
from mindflight.physics.rotations import quat_normalize, quat_rotate

DT_S = 0.002  # physics step from PLAN.md (500 Hz)
HOVER_RPM = CRAZYFLIE.hover_rpm
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEVICES = ["cpu", pytest.param("cuda", marks=cuda_only)]


def run(state, rpm_cmd, seconds, dt=DT_S):
    """Step `state` with a constant command for `seconds` and return the final state."""
    for _ in range(round(seconds / dt)):
        state = step(state, rpm_cmd, dt)
    return state


def hover_cmd(n, device="cpu"):
    return torch.full((n, 4), HOVER_RPM, device=device)


@pytest.mark.parametrize("device", DEVICES)
def test_free_fall(device):
    state = make_state(1, device, pos=[0.0, 0.0, 10.0])
    state = run(state, torch.zeros(1, 4, device=device), seconds=1.0)
    fallen_m = 0.5 * GRAVITY_MPS2 * 1.0**2  # z = z0 − ½ g t² = 10 − 4.905
    # Gravity is constant, so p ← p + v·dt + ½·a·dt² should land within 1 mm (only float32 rounding).
    assert abs(state[0, 2].item() - (10.0 - fallen_m)) < 1e-3
    assert math.isclose(state[0, VEL][2].item(), -GRAVITY_MPS2, rel_tol=0.01)  # vz = −g t
    assert torch.allclose(state[0, :2], torch.zeros(2, device=device))  # straight down, no drift
    assert torch.allclose(state[0, OMEGA], torch.zeros(3, device=device))


def test_tumble_conserves_angular_momentum_and_energy():
    # Motors off, high up, tumbling about all three axes. With no torque, the world angular momentum
    # L = R · (I ω) and the rotational energy ½ ωᵀ I ω must both stay constant.
    inertia_kgm2 = torch.tensor([CRAZYFLIE.ixx_kgm2, CRAZYFLIE.iyy_kgm2, CRAZYFLIE.izz_kgm2])

    def momentum_and_energy(state):
        body_momentum = inertia_kgm2 * state[:, OMEGA]
        world_momentum = quat_rotate(state[:, QUAT], body_momentum)[0].double()
        energy_j = 0.5 * (state[0, OMEGA] * body_momentum[0]).sum().double()
        return world_momentum, energy_j

    state = make_state(1, "cpu", pos=[0.0, 0.0, 1000.0])
    state[:, OMEGA] = torch.tensor([20.0, 5.0, 30.0])
    momentum_start, energy_start = momentum_and_energy(state)
    state = run(state, torch.zeros(1, 4), seconds=2.0)
    momentum_end, energy_end = momentum_and_energy(state)

    momentum_drift = ((momentum_end - momentum_start).norm() / momentum_start.norm()).item()
    energy_drift = abs((energy_end - energy_start) / energy_start).item()
    assert momentum_drift < 1e-3
    assert energy_drift < 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_hover_holds_position_for_5_seconds(device):
    start = torch.tensor([0.5, -0.3, 1.0], device=device)
    state = make_state(1, device, pos=start, rpm=[HOVER_RPM] * 4)
    state = run(state, hover_cmd(1, device), seconds=5.0)
    moved_m = (state[0, POS] - start).norm().item()
    assert moved_m < 1e-3


def test_left_motors_faster_drifts_toward_minus_y():
    # More lift on the left (+y) tips the drone right: thrust leans toward −y and pushes it that way.
    state = make_state(1, "cpu", pos=[0.0, 0.0, 10.0], rpm=[HOVER_RPM] * 4)
    cmd = hover_cmd(1)
    cmd[:, [2, 3]] += 50.0  # m2 back-left, m3 front-left
    state = run(state, cmd, seconds=0.5)
    assert state[0, OMEGA][0] > 0  # rolling right is positive roll about +x
    assert state[0, VEL][1] < 0 and state[0, 1] < -1e-3
    assert abs(state[0, 0].item()) < 1e-4  # no pitch, so no drift along x


def test_cw_faster_ccw_slower_yaws_positive_and_holds_height():
    # Keep total thrust equal to the weight: rpm_cw² + rpm_ccw² = 2 · hover².
    cw_rpm = HOVER_RPM + 500.0
    ccw_rpm = math.sqrt(2 * HOVER_RPM**2 - cw_rpm**2)
    cmd = torch.tensor([[ccw_rpm, cw_rpm, ccw_rpm, cw_rpm]])  # m0 CCW, m1 CW, m2 CCW, m3 CW
    state = make_state(1, "cpu", pos=[0.0, 0.0, 1.0], rpm=[HOVER_RPM] * 4)

    yaw_rates = []
    for _ in range(5):
        state = run(state, cmd, seconds=0.2)
        yaw_rates.append(state[0, OMEGA][2].item())
    assert yaw_rates[0] > 0
    assert all(later > earlier for earlier, later in zip(yaw_rates, yaw_rates[1:]))  # keeps spinning up
    assert abs(state[0, 2].item() - 1.0) < 1e-3  # z barely moves
    assert state[0, OMEGA][:2].abs().max() < 1e-4  # no roll or pitch


def test_torque_free_spin_keeps_angular_velocity():
    # Motors off, falling, spinning about body z: no torque, so ω must not change (and ω × Iω = 0 about a
    # principal axis). Orientation must turn by exactly ω·t.
    spin_radps = 20.0
    state = make_state(1, "cpu", pos=[0.0, 0.0, 100.0])
    state[:, OMEGA] = torch.tensor([0.0, 0.0, spin_radps])
    state = run(state, torch.zeros(1, 4), seconds=1.0)
    assert torch.allclose(state[0, OMEGA], torch.tensor([0.0, 0.0, spin_radps]), rtol=1e-3, atol=1e-6)
    w, _, _, z = state[0, QUAT].tolist()
    yaw_rad = 2 * math.atan2(z, w)
    expected = math.remainder(spin_radps * 1.0, 2 * math.pi)
    assert abs(math.remainder(yaw_rad - expected, 2 * math.pi)) < 1e-3


def maneuver_cmd(t_s):
    """A 2 s maneuver: roll one way, roll back, pitch, then yaw while climbing. Same commands at any dt."""
    cmd = torch.full((1, 4), HOVER_RPM)
    if t_s < 0.5:
        cmd[:, [2, 3]] += 30.0  # roll right
    elif t_s < 1.0:
        cmd[:, [0, 1]] += 30.0  # roll back left
    elif t_s < 1.5:
        cmd[:, [0, 3]] += 30.0  # nose up
    else:
        cmd[:, [1, 3]] += 300.0  # yaw and climb
    return cmd


def fly_maneuver(dt):
    state = make_state(1, "cpu", pos=[0.0, 0.0, 10.0], rpm=[HOVER_RPM] * 4)
    steps = round(2.0 / dt)
    for i in range(steps):
        state = step(state, maneuver_cmd(i * dt), dt)
    return state


def test_convergence_coarse_vs_fine_dt():
    coarse = fly_maneuver(0.002)
    fine = fly_maneuver(0.0005)
    assert (fine[0, POS] - torch.tensor([0.0, 0.0, 10.0])).norm() > 0.05  # it really went somewhere
    assert (coarse[0, POS] - fine[0, POS]).norm() < 0.01


def test_falling_drone_stops_on_floor():
    state = make_state(3, "cpu", pos=[[0.0, 0.0, 0.5], [1.0, 2.0, 2.0], [0.0, 0.0, 0.0]])
    state[1, VEL] = torch.tensor([0.3, -0.2, -1.0])  # also sliding sideways when it lands
    state = run(state, torch.zeros(3, 4), seconds=2.0)
    assert torch.all(state[:, 2] == 0.0)
    assert torch.all(state[:, VEL] == 0.0)
    assert torch.allclose(state[1, :2], torch.tensor([1.0, 2.0]), atol=0.3)  # landed near where it fell


def random_batch(n, device, seed=0):
    """N different drones mid-flight, and N different commands."""
    generator = torch.Generator().manual_seed(seed)
    state = make_state(n, "cpu")
    state[:, POS] = torch.rand(n, 3, generator=generator) * 4 + torch.tensor([-2.0, -2.0, 5.0])
    state[:, QUAT] = quat_normalize(torch.randn(n, 4, generator=generator))
    state[:, VEL] = torch.randn(n, 3, generator=generator)
    state[:, OMEGA] = torch.randn(n, 3, generator=generator) * 3
    state[:, RPM] = HOVER_RPM + torch.randn(n, 4, generator=generator) * 1000
    cmd = HOVER_RPM + torch.randn(n, 4, generator=generator) * 2000
    return state.to(device), cmd.to(device)


@pytest.mark.parametrize("device", DEVICES)
def test_each_drone_same_alone_or_in_batch_of_4096(device):
    state, cmd = random_batch(4096, device)
    batched = state
    for _ in range(50):
        batched = step(batched, cmd)
    for i in [0, 1, 777, 2048, 4095]:
        alone = state[i : i + 1]
        for _ in range(50):
            alone = step(alone, cmd[i : i + 1])
        assert torch.allclose(alone[0], batched[i], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_quaternion_stays_unit_and_shapes_dtype_device(device):
    state, cmd = random_batch(4096, device, seed=1)
    for _ in range(500):  # 1 s of tumbling
        state = step(state, cmd)
    assert state.shape == (4096, 17) and state.dtype == torch.float32 and state.device.type == device
    assert torch.isfinite(state).all()
    assert (state[:, QUAT].norm(dim=1) - 1).abs().max() < 1e-5
    assert torch.all(state[:, RPM] >= 0) and torch.all(state[:, RPM] <= CRAZYFLIE.max_rpm * (1 + 1e-6))


def test_step_does_not_change_its_input():
    state = make_state(2, "cpu", pos=[0.0, 0.0, 1.0])
    before = state.clone()
    step(state, hover_cmd(2))
    assert torch.equal(state, before)


def test_make_state_defaults():
    state = make_state(5, "cpu")
    assert state.shape == (5, 17) and state.dtype == torch.float32
    assert torch.equal(state[:, QUAT], torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 5))
    assert torch.all(state[:, POS] == 0) and torch.all(state[:, VEL] == 0)
    assert torch.all(state[:, OMEGA] == 0) and torch.all(state[:, RPM] == 0)
