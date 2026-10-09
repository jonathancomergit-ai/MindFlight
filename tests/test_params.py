import math

from mindflight.physics.params import CRAZYFLIE, GRAVITY_MPS2


def test_hover_rpm_holds_up_the_weight():
    total_thrust_n = 4 * CRAZYFLIE.kf_n_per_rpm2 * CRAZYFLIE.hover_rpm**2
    assert math.isclose(total_thrust_n, CRAZYFLIE.mass_kg * GRAVITY_MPS2, rel_tol=1e-9)


def test_hover_and_max_rpm_match_plan():
    assert abs(CRAZYFLIE.hover_rpm - 14_476) < 1
    assert abs(CRAZYFLIE.max_rpm - 21_714) < 1


def test_max_thrust_is_thrust_to_weight_times_weight():
    max_thrust_n = 4 * CRAZYFLIE.kf_n_per_rpm2 * CRAZYFLIE.max_rpm**2
    assert math.isclose(max_thrust_n / CRAZYFLIE.weight_n, CRAZYFLIE.thrust_to_weight, rel_tol=1e-9)
