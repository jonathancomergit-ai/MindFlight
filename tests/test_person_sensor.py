"""P6: the person sensors (camera on the drone, RTK GPS, phone GPS): delay, rate, noise size, field of view."""

import math

import pytest
import torch

from mindflight.envs.person_sensor import PersonTracker, in_view

DT = 0.02


def tracker(profile: str, n: int, seed: int = 0) -> PersonTracker:
    return PersonTracker(profile, n, "cpu", torch.Generator().manual_seed(seed), DT)


def walking_records(steps: int, n: int) -> list[torch.Tensor]:
    """Person rows (x, y, heading, vx, vy, turn rate) walking along +x at 1 m/s from x = 0."""
    rows = []
    for k in range(steps):
        row = torch.zeros(n, 6)
        row[:, 0] = k * DT
        row[:, 3] = 1.0
        rows.append(row)
    return rows


def test_delay_shifts_by_the_right_number_of_steps():
    t = tracker("gps_rtk", 4)  # 10 Hz, 5 steps late
    rows = walking_records(40, 4)
    t.reset(torch.arange(4), rows[0], torch.zeros(4, 3))
    t.noise_m[:] = 0.0  # no position noise (velocity noise doesn't matter here)
    t.phase[:] = 0.0
    drone = torch.zeros(4, 3)
    for k in range(1, 40):
        t.step(rows[k], drone, torch.zeros(4))
        if k % 5 == 0:  # 10 Hz = a measurement every 5th step, of the person 5 steps ago
            assert torch.allclose(t.seen_xy[:, 0], torch.full((4,), (k - 5) * DT))
            assert torch.allclose(t.age_s, torch.full((4,), 5 * DT))
        else:  # in between, the old measurement is kept and gets older
            assert torch.allclose(t.age_s, torch.full((4,), (5 + k % 5) * DT))


def test_rates_are_right():
    for profile, expected_hz in (("camera", 30.0), ("gps_rtk", 10.0), ("perfect", 50.0)):
        t = tracker(profile, 64)
        rows = walking_records(2, 64)
        t.reset(torch.arange(64), rows[0], torch.zeros(64, 3))
        drone = torch.zeros(64, 3)  # the person straight ahead of the nose: always in view
        fresh = 0
        for _ in range(500):  # 10 s
            t.step(rows[1], drone + torch.tensor([-1.0, 0.0, 0.0]), torch.zeros(64))
            fresh += int((t.age_s <= t.delay_steps.float() * DT + 1e-6).sum())
        assert fresh / 64 / 10.0 == pytest.approx(expected_hz, rel=0.02)
    gps = tracker("gps", 4000)
    gps.reset(torch.arange(4000), torch.zeros(4000, 6), torch.zeros(4000, 3))
    assert gps.rate_hz.min() >= 1.0 and gps.rate_hz.max() <= 5.0


def test_camera_noise_grows_with_range():
    """σ = 3 cm + 2 % of the range: 7 cm at 2 m, 13 cm at 5 m."""
    for range_m, sigma in ((2.0, 0.07), (5.0, 0.13)):
        t = tracker("camera", 20000)
        person = torch.zeros(20000, 6)
        drone = torch.zeros(20000, 3)
        drone[:, 0] = -range_m
        t.reset(torch.arange(20000), person, drone)
        assert (t.seen_xy - person[:, :2]).std().item() == pytest.approx(sigma, rel=0.03)
        assert (t.seen_vel - person[:, 3:5]).std().item() == pytest.approx(0.2, rel=0.03)


def test_rtk_noise_and_phone_gps_wanders():
    t = tracker("gps_rtk", 20000)
    t.reset(torch.arange(20000), torch.zeros(20000, 6), torch.zeros(20000, 3))
    error = t.seen_xy.std().item()
    assert 0.02 <= error <= 0.03

    gps = tracker("gps", 2000, seed=1)
    gps.reset(torch.arange(2000), torch.zeros(2000, 6), torch.zeros(2000, 3))
    start = gps.wander.clone()
    assert start.std().item() == pytest.approx(2.5, rel=0.1)  # σ 2–3 m
    for _ in range(50):  # 1 s later the error has barely moved: it wanders slowly (τ = 20 s)...
        gps.step(torch.zeros(2000, 6), torch.zeros(2000, 3), torch.zeros(2000))
    moved = (gps.wander - start).norm(dim=1)
    assert moved.median().item() < 1.0
    assert torch.corrcoef(torch.stack((start[:, 0], gps.wander[:, 0])))[0, 1] > 0.9
    for _ in range(50 * 60):  # ...and after a minute it is mostly a new error, but still 2–3 m in size
        gps.step(torch.zeros(2000, 6), torch.zeros(2000, 3), torch.zeros(2000))
    assert torch.corrcoef(torch.stack((start[:, 0], gps.wander[:, 0])))[0, 1] < 0.3
    assert gps.wander.std().item() == pytest.approx(2.5, rel=0.15)


def test_field_of_view():
    drone = torch.zeros(5, 2)
    yaw = torch.zeros(5)
    half = torch.full((5,), math.radians(35))
    person = torch.tensor([[3.0, 0.0], [3.0, 1.5], [3.0, 3.0], [-3.0, 0.0], [12.0, 0.0]])  # 0°, 27°, 45°, behind, far
    assert in_view(drone, yaw, person, half, 10.0).tolist() == [True, True, False, False, False]
    yaw_left = torch.full((5,), math.pi / 4)  # turning the nose 45° left brings the 45° person into view
    assert in_view(drone, yaw_left, person, half, 10.0)[2]


def test_out_of_view_the_camera_holds_the_last_measurement_and_ages():
    t = tracker("camera", 2)
    rows = walking_records(60, 2)
    drone = torch.tensor([[-2.0, 0.0, 1.5], [-2.0, 0.0, 1.5]])
    t.reset(torch.arange(2), rows[0], drone)
    yaw = torch.tensor([0.0, math.pi])  # drone 0 looks at the person, drone 1 looks away
    for k in range(1, 60):
        t.step(rows[k], drone, yaw)
    assert t.visible.tolist() == [True, False]
    assert t.age_s[0].item() < 0.2  # fresh
    assert t.age_s[1].item() == pytest.approx(59 * DT + t.delay_steps[1].item() * DT, abs=1e-5)  # never updated
    assert abs(t.seen_xy[1, 0].item() - rows[0][1, 0].item()) < 0.5  # still where the person was at the start
    assert t.seen_xy[0, 0].item() > 0.8  # drone 0 saw them walk on
