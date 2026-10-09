import math

import pytest
import torch

from mindflight.envs import make_env
from mindflight.envs.base import POLICY_DT_S, SUBSTEPS, action_to_rpm
from mindflight.envs.hover import CRASH_PENALTY, HoverEnv
from mindflight.physics.dynamics import OMEGA, POS, QUAT, RPM, VEL
from mindflight.physics.params import CRAZYFLIE
from mindflight.physics.rotations import quat_to_matrix


def place_at_target(env: HoverEnv) -> None:
    """Put every drone exactly on its target: level, still, motors at hover."""
    env.state[:] = 0.0
    env.state[:, POS] = env.target
    env.state[:, QUAT.start] = 1.0
    env.state[:, RPM] = CRAZYFLIE.hover_rpm


def test_timing_is_50_hz_with_10_substeps():
    assert SUBSTEPS == 10
    assert POLICY_DT_S == pytest.approx(0.02)


def test_action_maps_to_hover_and_full_throttle():
    rpm = action_to_rpm(torch.tensor([[0.0, 1.0, -1.0, 5.0]]))
    assert rpm[0, 0].item() == pytest.approx(CRAZYFLIE.hover_rpm)
    # +1 is full throttle: 1.5 × hover RPM gives 1.5² = 2.25 × hover thrust, the Crazyflie's thrust-to-weight.
    assert rpm[0, 1].item() == pytest.approx(CRAZYFLIE.max_rpm, rel=1e-5)
    assert rpm[0, 2].item() == pytest.approx(0.5 * CRAZYFLIE.hover_rpm)
    assert rpm[0, 3].item() == pytest.approx(CRAZYFLIE.max_rpm, rel=1e-5)  # clamped


def test_shapes_and_reset_ranges():
    env = make_env("hover", 512, device="cpu", seed=3)
    obs = env.reset()
    assert obs.shape == (512, env.obs_size) == (512, 22)
    pos = env.state[:, POS]
    assert pos[:, :2].abs().max() <= 1.0
    assert pos[:, 2].min() >= 0.3 and pos[:, 2].max() <= 2.0
    tilt = torch.acos(quat_to_matrix(env.state[:, QUAT])[:, 2, 2].clamp(-1, 1))
    assert tilt.max() <= math.radians(30) + 1e-4
    assert torch.allclose(env.state[:, RPM], torch.full_like(env.state[:, RPM], CRAZYFLIE.hover_rpm))
    # Task columns: target − position.
    assert torch.allclose(obs[:, 19:22], env.target - pos)

    obs, reward, done, info = env.step(torch.zeros(512, 4))
    assert obs.shape == (512, 22) and reward.shape == (512,) and done.dtype == torch.bool
    assert info["final_obs"].shape == (512, 22)


def test_same_seed_same_starts():
    a = make_env("hover", 64, device="cpu", seed=7)
    b = make_env("hover", 64, device="cpu", seed=7)
    a.reset()
    b.reset()
    assert torch.equal(a.state, b.state)


def test_perfect_hover_scores_about_one_per_step_and_stays_put():
    env = HoverEnv(8, device="cpu")
    env.reset()
    place_at_target(env)
    total = 0.0
    for _ in range(100):  # 2 s
        _, reward, done, _ = env.step(torch.zeros(8, 4))
        assert not done.any()
        total += reward.mean().item()
    assert total / 100 == pytest.approx(1.0, abs=1e-3)
    assert env.distance().max() < 1e-3


def test_reward_drops_with_distance_tilt_and_spin():
    env = HoverEnv(4, device="cpu")
    env.reset()
    place_at_target(env)
    env.state[1, 2] += 0.5  # 0.5 m above the target
    env.state[2, QUAT] = torch.tensor([math.cos(0.25), math.sin(0.25), 0.0, 0.0])  # tipped 0.5 rad
    env.state[3, OMEGA] = torch.tensor([0.0, 0.0, 3.0])  # spinning
    env.action[:] = 0.0
    reward = env._reward()
    assert reward[0] > reward[1] and reward[0] > reward[2] and reward[0] > reward[3]
    assert reward[1].item() == pytest.approx(0.5, abs=1e-3)  # 1 − 0.5 m


def test_crash_ends_episode_with_penalty_and_resets():
    env = HoverEnv(3, device="cpu")
    env.reset()
    place_at_target(env)
    env.state[0, 2] = 0.01  # on the floor
    env.state[1, QUAT] = torch.tensor([0.0, 1.0, 0.0, 0.0])  # upside down
    obs, reward, done, info = env.step(torch.zeros(3, 4))
    assert done.tolist() == [True, True, False]
    assert info["terminated"].tolist() == [True, True, False]
    assert reward[0] < -CRASH_PENALTY + 1.5 and reward[2] > 0.9
    assert info["episodes"]["crashed"].tolist() == [1.0, 1.0]
    # Reset straight away: the returned obs belongs to the new episode, in the start ranges.
    assert env.step_count[:2].tolist() == [0, 0] and env.state[0, 2] >= 0.3
    assert not torch.equal(obs[0], info["final_obs"][0])


def test_time_limit_truncates_and_counts_success():
    env = HoverEnv(2, episode_s=0.2, device="cpu")  # 10 policy steps
    env.reset()
    place_at_target(env)
    env.state[1, 0] += 0.2  # 20 cm off: survives, but not within 10 cm
    for i in range(10):
        _, _, done, info = env.step(torch.zeros(2, 4))
        if i < 9:
            assert not done.any()
    assert info["truncated"].tolist() == [True, True]
    assert info["terminated"].tolist() == [False, False]
    assert info["episodes"]["success"].tolist() == [1.0, 0.0]
    assert info["episodes"]["length_s"].tolist() == pytest.approx([0.2, 0.2])


def test_left_motors_up_rolls_right():
    """The env's action wiring keeps the PLAN.md signs: more thrust on the left (m2, m3) rolls toward −y."""
    env = HoverEnv(1, device="cpu")
    env.reset()
    place_at_target(env)
    for _ in range(5):
        env.step(torch.tensor([[0.0, 0.0, 0.1, 0.1]]))
    assert env.state[0, OMEGA.start] > 0  # positive roll rate = right side goes down
    assert env.state[0, VEL.start + 1] < 0  # drifting toward −y


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_runs_on_gpu():
    env = make_env("hover", 4096, device="cuda")
    obs = env.reset()
    obs, reward, done, _ = env.step(torch.zeros(4096, 4, device="cuda"))
    assert obs.is_cuda and reward.is_cuda and done.is_cuda

