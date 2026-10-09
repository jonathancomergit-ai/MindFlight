import math

import pytest
import torch

from mindflight.envs import make_env
from mindflight.envs.sensing import PROFILES, SensorModel
from mindflight.physics.dynamics import make_state


def model(profile, n=20000, seed=0):
    m = SensorModel(profile, n, "cpu", torch.Generator().manual_seed(seed), step_dt_s=0.02)
    m.reset(torch.arange(n))
    return m


def test_ideal_estimate_is_the_truth_and_has_no_delay():
    m = model("ideal", n=8)
    state = make_state(8, "cpu", pos=[1.0, 2.0, 3.0])
    assert m.estimate(state) is state
    obs = torch.randn(8, 5)
    assert torch.equal(m.delay_observation(obs), obs)


@pytest.mark.parametrize("name", ["lighthouse", "flowdeck"])
def test_noise_matches_the_profile(name):
    p = PROFILES[name]
    m = model(name)
    state = make_state(m.n, "cpu", pos=[0.0, 0.0, 1.0])
    est = m.estimate(state)
    pos_err = est[:, 0:3] - state[:, 0:3]
    vel_err = est[:, 7:10] - state[:, 7:10]
    assert pos_err.std().item() == pytest.approx(p.position_noise_m, rel=0.05)
    assert vel_err.std().item() == pytest.approx(p.velocity_noise_mps, rel=0.05)
    # Gyro: noise plus a per-episode bias uniform in ±b, so the spread is √(σ² + b²/3).
    gyro_err = est[:, 10:13] - state[:, 10:13]
    expected = math.sqrt(p.gyro_noise_radps**2 + p.gyro_bias_radps**2 / 3)
    assert gyro_err.std().item() == pytest.approx(expected, rel=0.05)
    # Attitude: the quaternion is still unit length, and the tilt error is about the profile's size.
    assert torch.allclose(est[:, 3:7].norm(dim=1), torch.ones(m.n), atol=1e-6)
    angle = 2 * torch.acos(est[:, 3].clamp(-1, 1))
    assert angle.mean().item() == pytest.approx(p.attitude_noise_rad * math.sqrt(8 / math.pi), rel=0.1)


def test_flowdeck_position_drifts_like_a_random_walk():
    m = model("flowdeck")
    for _ in range(500):  # 10 s
        m.advance()
    expected = PROFILES["flowdeck"].position_drift_m_per_sqrt_s * math.sqrt(10.0)
    assert m.position_drift.std().item() == pytest.approx(expected, rel=0.05)


def test_observation_delay_shows_the_past():
    m = model("flowdeck", n=3)
    m.obs_delay[:] = torch.tensor([0, 1, 2])
    seen = [m.delay_observation(torch.full((3, 1), float(t))) for t in range(5)]
    assert seen[4].flatten().tolist() == [4.0, 3.0, 2.0]  # each drone sees its own number of steps back
    # A reset drone's history is its first new observation only.
    m.restart_history(torch.tensor([2]), torch.full((3, 1), 99.0))
    assert m.delay_observation(torch.full((3, 1), 5.0)).flatten().tolist() == [5.0, 4.0, 99.0]


def test_action_delay_delivers_old_actions():
    m = model("lighthouse", n=2)
    m.action_delay[:] = torch.tensor([0, 1])
    first = m.delay_action(torch.ones(2, 4))
    second = m.delay_action(2 * torch.ones(2, 4))
    assert first[1].tolist() == [0.0] * 4  # drone 1 still gets the hover action from before
    assert second[0].tolist() == [2.0] * 4 and second[1].tolist() == [1.0] * 4


def test_env_rewards_use_the_truth_but_observations_use_the_estimate():
    env = make_env("hover", 512, device="cpu", sensors="flowdeck", action_mode="attitude", seed=2)
    env.reset()
    obs, _, _, _ = env.step(torch.zeros(512, 4))
    true_task_obs = env.target - env.state[:, 0:3]
    error = (obs[:, 19:22] - true_task_obs).std().item()
    assert 0.01 < error < 0.06  # noisy (and a step or two late), but close
    assert torch.equal(env.distance(), (env.target - env.state[:, 0:3]).norm(dim=1))  # reward side: the truth


def test_real_world_randomizes_each_drone_and_drains_the_battery():
    env = make_env("hover", 256, device="cpu", real_world=True, seed=1)
    env.reset()
    v = env.variation
    assert 0.92 <= v.mass_scale.min() and v.mass_scale.max() <= 1.08 and v.mass_scale.std() > 0.02
    assert v.com_offset_m.abs().max() <= 0.003 and v.drag and v.ground_effect
    start = v.thrust_scale.clone()
    for _ in range(100):  # 2 s
        env.step(torch.zeros(256, 4))
    alive = env.step_count >= 100  # drones still in their first episode
    assert (env.variation.thrust_scale[alive] <= start[alive] + 1e-6).all()


def test_estimate_errors_wander_smoothly():
    """Filter-like errors: the same spread, but neighbouring steps are close (correlated over about τ)."""
    m = model("flowdeck", n=5000)
    state = make_state(m.n, "cpu", pos=[0.0, 0.0, 1.0])
    first = m.estimate(state)[:, 0:3] - state[:, 0:3]
    m.advance()
    second = m.estimate(state)[:, 0:3] - state[:, 0:3]
    tau = PROFILES["flowdeck"].noise_correlation_s
    correlation = torch.corrcoef(torch.stack((first[:, 0], second[:, 0])))[0, 1].item()
    assert correlation == pytest.approx(math.exp(-0.02 / tau), abs=0.03)
    assert second.std().item() == pytest.approx(PROFILES["flowdeck"].position_noise_m, rel=0.06)
