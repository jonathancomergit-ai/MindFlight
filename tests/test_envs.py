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


def test_waypoint_targets_change_every_2_5_s_and_count_when_settled():
    from mindflight.envs.waypoint import WaypointEnv

    env = WaypointEnv(2, device="cpu", seed=1)
    env.reset()
    assert env.targets_per_episode == 4
    first_targets = env.target.clone()
    # Drone 0 sits on its target the whole time; drone 1 sits 20 cm away.
    for i in range(env.hold_steps):
        env.state[:, POS] = env.target
        env.state[1, 0] += 0.2
        env.state[:, VEL] = 0.0
        env.step(torch.zeros(2, 4))
    assert not torch.equal(env.target, first_targets)  # new targets after 2.5 s
    assert env.targets_reached.tolist() == [1.0, 0.0]


def test_waypoint_leaving_the_arena_is_a_crash():
    from mindflight.envs.waypoint import WaypointEnv

    env = WaypointEnv(1, device="cpu")
    env.reset()
    env.state[0, POS] = torch.tensor([3.2, 0.0, 1.0])
    _, _, _, info = env.step(torch.zeros(1, 4))
    assert info["terminated"].tolist() == [True]


def straight_course(env, opening=1.0):
    """Gates every 2 m along +x at 1 m height, facing +x."""
    from mindflight.envs.gates import N_GATES

    env.gate_centre[:] = 0.0
    env.gate_centre[:, :, 0] = 2.0 * torch.arange(1, N_GATES + 1, dtype=torch.float32)
    env.gate_centre[:, :, 2] = 1.0
    env.gate_heading[:] = 0.0
    env.gate_opening[:] = opening


def test_gates_course_shapes_and_obs():
    from mindflight.envs.gates import N_GATES, GatesEnv

    env = GatesEnv(16, device="cpu", seed=2, difficulty=1.0)
    obs = env.reset()
    assert obs.shape == (16, 19 + 11)
    assert env.gate_centre.shape == (16, N_GATES, 3)
    assert env.gate_opening.tolist() == pytest.approx([0.5] * 16)
    assert env.gate_centre[:, :, 2].min() >= 0.6 and env.gate_centre[:, :, 2].max() <= 2.0
    # Neighbouring gates are 1.5–2.5 m apart sideways.
    steps = (env.gate_centre[:, 1:, :2] - env.gate_centre[:, :-1, :2]).norm(dim=-1)
    assert steps.min() >= 1.5 - 1e-4 and steps.max() <= 2.5 + 1e-4


def teleport(env, x, y=0.0, z=1.0):
    """Move drone 0 to (x, y, z) between two steps, so the env sees a straight path from prev_pos."""
    env.state[0, POS] = torch.tensor([x, y, z])
    env.state[0, VEL] = 0.0


def test_gate_passed_through_the_middle():
    from mindflight.envs.gates import GatesEnv

    env = GatesEnv(1, device="cpu", difficulty=0.0)
    env.reset()
    straight_course(env)
    env.prev_pos[0] = torch.tensor([1.9, 0.0, 1.0])
    teleport(env, 2.1)
    reward = env._reward()
    assert env.passed.tolist() == [True] and env.hit_frame.tolist() == [False]
    assert env.next_gate.tolist() == [1]
    assert reward.item() > 9.9  # the pass bonus (no progress: 0.1 m from the centre before and after)


def test_gate_frame_hit_is_a_crash_and_a_wide_miss_is_not():
    from mindflight.envs.gates import GatesEnv

    env = GatesEnv(1, device="cpu", difficulty=0.0)
    env.reset()
    straight_course(env, opening=1.0)
    env.prev_pos[0] = torch.tensor([1.9, 0.48, 1.0])  # through the edge of the 1 m opening, on the frame
    teleport(env, 2.1, y=0.48)
    env._reward()
    assert env.hit_frame.tolist() == [True] and env.passed.tolist() == [False]
    assert env._crashed().tolist() == [True]

    env.reset()
    straight_course(env, opening=1.0)
    env.prev_pos[0] = torch.tensor([1.9, 1.5, 1.0])  # well wide of the gate
    teleport(env, 2.1, y=1.5)
    env._reward()
    assert env.hit_frame.tolist() == [False] and env.passed.tolist() == [False]
    assert env.next_gate.tolist() == [0]


def test_all_gates_in_order_finishes():
    from mindflight.envs.gates import N_GATES, GatesEnv

    env = GatesEnv(1, device="cpu", difficulty=0.0)
    env.reset()
    straight_course(env)
    for k in range(N_GATES):
        env.prev_pos[0] = torch.tensor([2.0 * (k + 1) - 0.1, 0.0, 1.0])
        teleport(env, 2.0 * (k + 1) + 0.1)
        env._reward()
    assert env.next_gate.tolist() == [N_GATES]
    assert env._finished().tolist() == [True]


def test_curriculum_climbs_when_success_is_high():
    from mindflight.envs.gates import GatesEnv

    env = GatesEnv(4, device="cpu")
    assert env.difficulty == 0.0
    for _ in range(20):
        env.update_curriculum({"episode_success": 1.0})
    assert env.difficulty > 0.0
    fixed = GatesEnv(4, device="cpu", difficulty=0.3)
    for _ in range(20):
        fixed.update_curriculum({"episode_success": 1.0})
    assert fixed.difficulty == 0.3


def test_sturdy_randomizes_drones_and_wind_and_gusts_move():
    from mindflight.envs.sturdy import SturdyEnv

    env = SturdyEnv(256, device="cpu", seed=4)
    env.reset()
    v = env.variation
    assert 0.8 <= v.mass_scale.min() and v.mass_scale.max() <= 1.2 and v.mass_scale.std() > 0.05
    assert 0.7 <= v.motor_tau_scale.min() and v.motor_tau_scale.max() <= 1.5
    assert env.steady_wind[:, :2].norm(dim=1).max() <= 2.0 + 1e-5 and env.steady_wind[:, 2].abs().max() == 0
    wind_before = v.wind_mps.clone()
    env.step(torch.zeros(256, 4))
    assert not torch.equal(v.wind_mps, wind_before)  # gusts changed it

    calm = SturdyEnv(8, device="cpu", randomize=False)
    calm.reset()
    calm.step(torch.zeros(8, 4))
    assert torch.all(calm.variation.mass_scale == 1) and torch.all(calm.variation.wind_mps == 0)


def test_staggered_reset_spreads_episode_clocks():
    env = HoverEnv(512, device="cpu", seed=1)
    env.reset(stagger=True)
    assert env.step_count.min() >= 0 and env.step_count.max() < env.max_steps
    assert env.step_count.float().std() > env.max_steps / 4  # spread out, not all at 0
    env.reset()
    assert env.step_count.max() == 0  # evaluation: everyone starts fresh
