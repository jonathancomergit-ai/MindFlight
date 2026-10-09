"""Deploy: the fake Crazyflie behaves like the real firmware (unlock rule, watchdog, delay, logging, battery)."""

import math

import pytest

from mindflight.deploy.sitl import SitlConfig, SitlCrazyflie
from mindflight.physics.params import CRAZYFLIE

P = CRAZYFLIE
HOVER = P.hover_thrust_command


def make(**config) -> SitlCrazyflie:
    sitl = SitlCrazyflie(config=SitlConfig(**config))
    sitl.connect()
    return sitl


def hold(sitl: SitlCrazyflie, roll_deg: float, thrust: int, seconds: float) -> None:
    """Send the same setpoint at 50 Hz for `seconds`."""
    for _ in range(round(seconds / 0.02)):
        sitl.send_setpoint(roll_deg, 0.0, 0.0, thrust)
        sitl.advance(0.02)


def test_thrust_is_locked_until_a_zero_thrust_packet_arrives():
    sitl = make()
    hold(sitl, 0.0, HOVER, 0.3)
    assert not sitl.motors_on and sitl.position_m[2] == 0.0  # locked since boot: thrust ignored
    sitl.unlock(0.1)
    hold(sitl, 0.0, HOVER, 0.3)
    assert sitl.motors_on


def test_thrust_below_the_firmware_minimum_stops_the_motors():
    sitl = make()
    sitl.unlock(0.1)
    hold(sitl, 0.0, 5000, 0.2)  # below THRUST_MIN / THRUST_MAX · 65535 ≈ 7000
    assert not sitl.motors_on
    hold(sitl, 0.0, 12000, 0.2)
    assert sitl.motors_on


def test_watchdog_levels_after_half_a_second_and_cuts_and_locks_after_two():
    sitl = make(start_position_m=(0.0, 0.0, 1.0))
    sitl.state[0, 13:17] = P.hover_rpm
    sitl.unlock(0.04)
    hold(sitl, 15.0, HOVER, 0.5)  # rolled 15° right, then the PC goes quiet
    last_sent_s = sitl.now()
    sitl.advance(0.35)
    assert math.degrees(sitl.euler_rad[0]) > 10  # still holding the last setpoint
    sitl.advance(0.6)  # ~0.95 s without a setpoint
    assert abs(math.degrees(sitl.euler_rad[0])) < 3 and sitl.motors_on  # levelled, thrust kept
    assert sitl.supervisor == "level_out"
    sitl.advance(last_sent_s + 1.9 - sitl.now())
    assert sitl.motors_on
    sitl.advance(0.25)  # > 2 s (plus the radio delay of the last packet)
    assert not sitl.motors_on and sitl.supervisor == "locked"
    sitl.unlock(0.1)
    hold(sitl, 0.0, HOVER, 0.3)
    assert not sitl.motors_on  # locked: "reboot required", setpoints are ignored


def test_stop_setpoint_stops_the_motors():
    sitl = make()
    sitl.unlock(0.1)
    hold(sitl, 0.0, HOVER, 0.2)
    sitl.send_stop_setpoint()
    sitl.advance(0.03)
    assert not sitl.motors_on


def test_setpoints_arrive_after_the_radio_delay_and_in_order():
    sitl = make(uplink_delay_s=(0.005, 0.020))
    sent = []
    for k in range(50):
        sent.append(sitl.now())
        sitl.send_setpoint(0.0, 0.0, 0.0, 0 if k == 0 else 20000 + k)
        sitl.advance(0.002)
    sitl.advance(0.05)
    arrived = [t for t, _ in sitl.received]
    assert len(arrived) == 50
    assert all(b >= a for a, b in zip(arrived, arrived[1:]))  # never overtake each other
    delays = [a - s for a, s in zip(arrived, sent)]
    assert min(delays) >= 0.004 and max(delays) <= 0.050


def test_estimate_comes_at_100_hz_with_noise_and_delay():
    sitl = make(start_position_m=(0.3, -0.2, 0.0))
    arrivals = []
    original = sitl._on_log_block

    def count(name, firmware_ms, values):
        arrivals.append((name, firmware_ms))
        original(name, firmware_ms, values)

    sitl._on_log_block = count
    sitl.advance(1.0)
    for name in ("position_velocity", "attitude", "rates_battery"):
        assert 98 <= sum(1 for n, _ in arrivals if n == name) <= 100  # 100 Hz, minus the ones still in the air
    estimate = sitl.latest_estimate()
    assert sitl.now() - estimate.oldest_s < 0.035  # 10 ms log period + up to 20 ms radio delay
    assert estimate.position_m == pytest.approx((0.3, -0.2, 0.0), abs=0.03)
    assert estimate.position_m != (0.3, -0.2, 0.0)  # noisy, not perfect
    assert estimate.battery_v == pytest.approx(4.15, abs=0.03)


def test_battery_drains_and_sags_under_load():
    sitl = make(battery_hover_s=20.0, battery_noise_v=0.0)
    resting_v = sitl.battery_v
    sitl.unlock(0.1)
    hold(sitl, 0.0, HOVER, 1.0)
    assert sitl.battery_v < resting_v - 0.2  # sag under hover load
    assert sitl.battery_open_v < resting_v - 0.04  # and charge used: 1 s of a 20 s battery


def test_upside_down_while_flying_counts_as_a_crash_and_stops_the_motors():
    sitl = make(start_position_m=(0.0, 0.0, 2.0))
    sitl.unlock(0.04)
    hold(sitl, 0.0, HOVER, 0.2)
    assert sitl.motors_on
    for _ in range(10):  # stuck upside down (pinned there every 20 ms) for 0.2 s
        sitl.state[0, 3:7] = sitl.state.new_tensor([0.0, 1.0, 0.0, 0.0])
        sitl.state[0, 10:13] = 0.0
        hold(sitl, 0.0, HOVER, 0.02)
    assert sitl.supervisor == "crashed" and not sitl.motors_on
