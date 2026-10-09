import math

import pytest
import torch

from mindflight.physics.aero import ground_effect_clip_height_m, ground_effect_thrust, rotor_drag_world
from mindflight.physics.params import CRAZYFLIE

LEVEL = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
HOVER_RPM = torch.full((1, 4), CRAZYFLIE.hover_rpm)


def test_drag_opposes_motion_and_matches_the_formula():
    velocity = torch.tensor([[3.0, 0.0, 0.0]])
    drag = rotor_drag_world(LEVEL, velocity, HOVER_RPM)
    total_radps = 4 * CRAZYFLIE.hover_rpm * 2 * math.pi / 60
    assert drag[0, 0].item() == pytest.approx(-total_radps * CRAZYFLIE.drag_coeff_xy * 3.0, rel=1e-5)
    assert drag[0, 1].item() == pytest.approx(0.0, abs=1e-9) and drag[0, 2].item() == pytest.approx(0.0, abs=1e-9)
    # About 6 % of the weight at 3 m/s (the docstring's claim).
    assert abs(drag[0, 0].item()) / CRAZYFLIE.weight_n == pytest.approx(0.063, abs=0.005)


def test_wind_pushes_a_still_drone_downwind():
    still = torch.zeros(1, 3)
    wind = torch.tensor([[0.0, 2.0, 0.0]])
    drag = rotor_drag_world(LEVEL, still, HOVER_RPM, wind)
    assert drag[0, 1].item() > 0  # pushed along +y with the wind
    # Moving with the wind at the same speed: no relative air, no force.
    assert rotor_drag_world(LEVEL, wind, HOVER_RPM, wind).abs().max().item() < 1e-9


def test_no_props_no_drag():
    assert rotor_drag_world(LEVEL, torch.tensor([[5.0, 5.0, 5.0]]), torch.zeros(1, 4)).abs().max().item() == 0


def test_ground_effect_size_and_falloff():
    h_clip = ground_effect_clip_height_m()
    assert h_clip == pytest.approx(0.038, abs=0.002)
    thrust = torch.tensor([CRAZYFLIE.weight_n] * 3)
    extra = ground_effect_thrust(torch.tensor([0.0, 0.10, 0.7]), thrust, torch.ones(3))
    ratio = (extra / thrust).tolist()
    assert ratio[0] == pytest.approx(0.27, abs=0.02)  # at (clipped) floor level
    assert ratio[1] == pytest.approx(0.038, abs=0.003)  # 10 cm
    assert ratio[2] < 0.001  # 70 cm: gone
    assert ground_effect_thrust(torch.tensor([0.05]), thrust[:1], torch.tensor([-0.5])).item() == 0  # upside down


# ---------- the extras inside dynamics.step ----------

from mindflight.physics.aero import DroneVariation  # noqa: E402
from mindflight.physics.dynamics import RPM, make_state, step  # noqa: E402


def fly(variation=None, seconds=1.0, n=1, height=1.0, rpm=None):
    rpm = torch.full((n, 4), CRAZYFLIE.hover_rpm) if rpm is None else rpm
    state = make_state(n, "cpu", pos=[0.0, 0.0, height], rpm=rpm)
    for _ in range(round(seconds / 0.002)):
        state = step(state, rpm, variation=variation)
    return state


def test_neutral_variation_changes_nothing():
    ones = torch.ones(1)
    neutral = DroneVariation(mass_scale=ones, thrust_scale=ones, motor_tau_scale=ones, wind_mps=torch.zeros(1, 3))
    assert torch.allclose(fly(neutral), fly(None), atol=1e-6)


def test_heavier_drone_sinks_at_hover_rpm():
    # 20 % heavier: a = g · (1/1.2 − 1) ≈ −1.635 m/s², so it drops ½ · 1.635 · 1² ≈ 0.82 m in 1 s.
    state = fly(DroneVariation(mass_scale=torch.tensor([1.2])))
    assert 1.0 - state[0, 2].item() == pytest.approx(0.5 * 9.81 * (1 - 1 / 1.2), rel=1e-3)


def test_weaker_props_sink_and_stronger_ones_climb():
    assert fly(DroneVariation(thrust_scale=torch.tensor([0.9])))[0, 2] < 1.0
    assert fly(DroneVariation(thrust_scale=torch.tensor([1.1])))[0, 2] > 1.0


def test_laggier_motors_spin_up_slower():
    start = torch.full((1, 4), CRAZYFLIE.hover_rpm)
    target = torch.full((1, 4), CRAZYFLIE.max_rpm)
    def after(variation):
        state = make_state(1, "cpu", pos=[0, 0, 1], rpm=start)
        for _ in range(10):  # 20 ms
            state = step(state, target, variation=variation)
        return state[0, RPM.start].item()
    assert after(DroneVariation(motor_tau_scale=torch.tensor([2.0]))) < after(None)


def test_wind_blows_the_drone_downwind_and_it_tends_to_wind_speed():
    wind = torch.tensor([[2.0, 0.0, 0.0]])
    state = fly(DroneVariation(wind_mps=wind, drag=True), seconds=3.0)
    assert state[0, 0] > 0.1  # pushed along +x
    assert 0 < state[0, 7] < 2.0  # moving with the wind, never faster than it
    # Without drag the wind can't touch it.
    assert fly(DroneVariation(wind_mps=wind, drag=False), seconds=1.0)[0, 0].abs() < 1e-6


def test_ground_effect_lifts_a_drone_near_the_floor_only():
    near = fly(DroneVariation(ground_effect=True), seconds=0.3, height=0.05)
    high = fly(DroneVariation(ground_effect=True), seconds=0.3, height=1.5)
    assert near[0, 2] > 0.06  # pushed up off the floor
    assert abs(high[0, 2].item() - 1.5) < 1e-3  # nothing at 1.5 m


def test_downwash_pushes_down_on_the_drone_below_only():
    from mindflight.physics.aero import downwash_world

    pos = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [3.0, 0.0, 1.0]])
    same_world = torch.tensor([0, 0, 0])
    force = downwash_world(pos, same_world)
    assert force[0, 2] < 0  # under drone 1: pushed down
    assert force[1, 2] == 0  # nothing above drone 1
    assert force[2, 2].abs() < 1e-9  # 3 m to the side: outside the air column
    assert force[:, :2].abs().max() == 0
    # Formula check for drone 0: Δz = 1, Δxy = 0  ->  α = c1 (r / 4)², times exp(0) = 1.
    expected = CRAZYFLIE.downwash_coeff_1 * (CRAZYFLIE.prop_radius_m / 4) ** 2
    assert -force[0, 2].item() == pytest.approx(expected, rel=1e-5)
    # Different worlds don't feel each other.
    assert downwash_world(pos, torch.tensor([0, 1, 2])).abs().max() == 0


def test_downwash_in_step_sinks_the_lower_drone():
    rpm = torch.full((2, 4), CRAZYFLIE.hover_rpm)
    state = make_state(2, "cpu", pos=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]), rpm=rpm)
    variation = DroneVariation(world_id=torch.tensor([0, 0]), downwash=True)
    for _ in range(100):
        state = step(state, rpm, variation=variation)
    assert state[0, 2] < 0.99 and abs(state[1, 2].item() - 2.0) < 1e-4


def test_downwash_is_capped_at_one_drone_weight():
    from mindflight.physics.aero import downwash_world

    pos = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.05]])  # 5 cm apart, straight above
    force = downwash_world(pos, torch.tensor([0, 0]))
    assert -force[0, 2].item() == pytest.approx(CRAZYFLIE.weight_n)


def test_heavy_nose_pitches_nose_down():
    variation = DroneVariation(com_offset_m=torch.tensor([[0.003, 0.0]]))  # centre of mass 3 mm forward
    state = fly(variation, seconds=0.2)
    assert state[0, 11] > 0  # positive pitch rate = nose down (FLU: rotation about +y)
    assert abs(state[0, 10].item()) < 1e-6  # no roll
