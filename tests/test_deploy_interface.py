"""Deploy: units and signs between our setpoint and the Crazyflie's (mindflight/deploy/interface.py).

The physical tests fly the SITL drone, whose firmware stand-in copies the real firmware's sign handling. A sign flip
in interface.py makes the drone go the wrong way here.
"""

import math
import struct

import pytest

from mindflight.deploy.interface import (
    FLOAT_BYTES,
    LOG_BLOCK_MAX_BYTES,
    LOG_BLOCKS,
    estimate_from_log,
    firmware_hover_command,
    setpoint_to_firmware,
    thrust_command_to_newtons,
    thrust_newtons_to_command,
)
from mindflight.deploy.sitl import RPYT_FORMAT, SitlConfig, SitlCrazyflie
from mindflight.physics.params import CRAZYFLIE

P = CRAZYFLIE
QUIET = dict(position_noise_m=0.0, velocity_noise_mps=0.0, attitude_noise_rad=0.0, gyro_noise_radps=0.0, battery_noise_v=0.0)


def test_signs_of_the_firmware_numbers():
    right = setpoint_to_firmware(P.weight_n, math.radians(10), 0.0, 0.0, P)
    nose_down = setpoint_to_firmware(P.weight_n, 0.0, math.radians(10), 0.0, P)
    counter_clockwise = setpoint_to_firmware(P.weight_n, 0.0, 0.0, math.radians(90), P)
    assert right.roll_deg == pytest.approx(10.0)  # + roll = right side down, same as cflib
    assert nose_down.pitch_deg == pytest.approx(10.0)  # + pitch = nose down in cflib too (cflib negates on the wire)
    assert counter_clockwise.yawrate_deg == pytest.approx(-90.0)  # cflib yaw rate + = clockwise


def test_sitl_packs_the_radio_packet_exactly_like_cflib():
    cflib_commander = pytest.importorskip("cflib.crazyflie.commander")

    class FakeCf:
        def send_packet(self, packet):
            self.data = bytes(packet.data)

    fake = FakeCf()
    cflib_commander.Commander(fake).send_setpoint(3.0, 7.0, -20.0, 40000)
    sitl = SitlCrazyflie(config=SitlConfig(uplink_delay_s=(0.0, 0.0)))
    sitl.connect()
    sitl.send_setpoint(3.0, 7.0, -20.0, 40000)
    assert sitl._uplink[0][3] == fake.data
    assert struct.unpack(RPYT_FORMAT, fake.data)[1] == pytest.approx(-7.0)  # pitch is negated on the wire


@pytest.mark.parametrize(
    "setpoint",
    [(P.weight_n, 0.2, -0.1, 0.5), (1.3 * P.weight_n, -0.3, 0.25, -1.0), (0.7 * P.weight_n, 0.0, 0.0, 3.0)],
)
def test_round_trip_through_the_firmware_gives_back_our_setpoint(setpoint):
    sitl = SitlCrazyflie(config=SitlConfig(uplink_delay_s=(0.0, 0.0), **QUIET))
    sitl.connect()
    sitl.unlock(0.04)
    sitl.send(setpoint_to_firmware(*setpoint, P))
    sitl.advance(0.004)
    thrust_n, roll, pitch, yaw_rate = sitl._setpoint_in_our_units(sitl.setpoint)
    assert thrust_n == pytest.approx(setpoint[0], rel=1e-4)  # one thrust step is 1/36163 of the weight
    assert (roll, pitch, yaw_rate) == pytest.approx(setpoint[1:], abs=1e-5)


def fly_sitl(thrust_n, roll, pitch, yaw_rate, seconds=0.4):
    """Hover 1 m up in SITL, then hold a setpoint (ours, through interface.py) for `seconds`."""
    sitl = SitlCrazyflie(config=SitlConfig(start_position_m=(0.0, 0.0, 1.0), **QUIET))
    sitl.connect()
    sitl.state[0, 13:17] = P.hover_rpm  # already hovering
    sitl.unlock(0.04)
    for _ in range(round(seconds / 0.02)):
        sitl.send(setpoint_to_firmware(thrust_n, roll, pitch, yaw_rate, P))
        sitl.advance(0.02)
    return sitl


def test_commands_move_the_real_drone_the_right_way():
    roll_right = fly_sitl(P.weight_n, math.radians(10), 0.0, 0.0)
    assert roll_right.state[0, 8] < -0.1  # velocity y < 0: moving right (y is left)
    nose_down = fly_sitl(P.weight_n, 0.0, math.radians(10), 0.0)
    assert nose_down.state[0, 7] > 0.1  # velocity x > 0: moving forward
    turn_left = fly_sitl(P.weight_n, 0.0, 0.0, math.radians(90))
    assert turn_left.euler_rad[2] > math.radians(15)  # yaw grew: counter-clockwise seen from above
    climb = fly_sitl(1.3 * P.weight_n, 0.0, 0.0, 0.0)
    assert climb.state[0, 9] > 0.3  # velocity z > 0: going up


def test_thrust_command_is_monotone_hovers_and_stays_in_range():
    newtons = [P.weight_n * k / 20 for k in range(0, 61)]
    commands = [thrust_newtons_to_command(n, P) for n in newtons]
    assert all(b >= a for a, b in zip(commands, commands[1:]))
    assert commands[0] == P.thrust_command_min and commands[-1] == P.thrust_command_max
    assert thrust_newtons_to_command(P.weight_n, P) == P.hover_thrust_command
    assert thrust_newtons_to_command(float("nan"), P) == P.thrust_command_min  # never full power by accident
    for n in (0.5 * P.weight_n, P.weight_n, 1.4 * P.weight_n):
        assert thrust_command_to_newtons(thrust_newtons_to_command(n, P), P) == pytest.approx(n, rel=1e-4)


def test_default_hover_command_is_the_firmware_thrust_line():
    # weight / (4 · 0.12 N) · 65535: the drone file's number must match the documented curve for its mass
    assert abs(firmware_hover_command(P) - P.hover_thrust_command) < 1.0


def test_log_blocks_fit_in_one_radio_packet_each():
    for name, variables in LOG_BLOCKS.items():
        assert len(variables) * FLOAT_BYTES <= LOG_BLOCK_MAX_BYTES, name


def test_estimate_from_log_turns_gyro_degrees_into_radians_and_keeps_the_rest():
    values = {v: 0.0 for vs in LOG_BLOCKS.values() for v in vs}
    values.update({"stateEstimate.z": 1.5, "stateEstimate.qw": 1.0, "gyro.y": 180.0, "pm.vbat": 3.9})
    estimate = estimate_from_log(values, time_s=1.0, oldest_s=0.99, firmware_ms=1000)
    assert estimate.position_m == (0.0, 0.0, 1.5)
    assert estimate.quaternion_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert estimate.gyro_radps[1] == pytest.approx(math.pi)
    assert estimate.battery_v == 3.9 and estimate.tilt_rad == pytest.approx(0.0)
