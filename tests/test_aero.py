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
