"""P6: the follow / orbit / point task. The person, the targets, the safety rules, the viewer shapes.
(The person sensors have their own tests in test_person_sensor.py.)"""

import math

import pytest
import torch

from mindflight.envs import follow, make_env
from mindflight.envs.base import POLICY_DT_S
from mindflight.envs.follow import (
    ARENA_M,
    FOLLOW,
    JOG_SPEED_MPS,
    ORBIT,
    POINT,
    SIDEWAYS_ACCEL_MAX_MPS2,
    TURN_MAX_RAD_S,
    FollowEnv,
    person_distance,
)
from mindflight.physics.dynamics import POS, QUAT, RPM, VEL
from mindflight.physics.params import CRAZYFLIE


def make(n=64, seed=0, difficulty=1.0, **kwargs) -> FollowEnv:
    """The full task (difficulty 1) unless asked otherwise."""
    env = make_env("follow", n, device="cpu", seed=seed, difficulty=difficulty, **kwargs)
    env.reset()
    return env


def park_drones(env: FollowEnv, pos: torch.Tensor) -> None:
    """Put every drone at `pos` (N, 3): level, still, motors at hover."""
    env.state[:] = 0.0
    env.state[:, POS] = pos
    env.state[:, QUAT.start] = 1.0
    env.state[:, RPM] = CRAZYFLIE.hover_rpm


def test_shapes_and_observation_columns():
    env = make(32)
    obs = env.observe()
    assert env.action_mode == "attitude"
    assert obs.shape == (32, 19 + 16) == (32, env.obs_size)
    pos = env.state[:, POS]
    assert torch.allclose(obs[:, 19:22], env.seen_target - pos)  # same first 3 task numbers as waypoint
    assert torch.allclose(obs[:, 31:34].sum(1), torch.ones(32))
    assert torch.equal(obs[:, 31:34].argmax(1), env.mode)
    assert torch.allclose(obs[:, 34], env.tracker.age_s)  # how old the person measurement is
    person = torch.cat((env.seen_xy, torch.full((32, 1), follow.PERSON_CENTRE_Z_M)), dim=1)
    assert torch.allclose(obs[:, 25:28], person - pos)
    obs, reward, done, info = env.step(torch.zeros(32, 4))
    assert obs.shape == (32, 35) and reward.shape == (32,)


def test_person_stays_in_arena_and_moves_like_a_person():
    """30 s of walking for 512 people: inside the arena, never faster than a jog, smooth turns, and a mix of
    standing, walking and jogging."""
    env = make(512, seed=1)
    max_speed = 0.0
    max_xy = 0.0
    stood = torch.zeros(512, dtype=torch.bool)
    jogged = torch.zeros(512, dtype=torch.bool)
    for _ in range(1500):
        heading_before = env.person_heading.clone()
        speed_before = env.person_speed.clone()
        turn_before = env.person_turn_rate.clone()
        env.move_person()
        turn_change = (env.person_turn_rate - turn_before).abs() / POLICY_DT_S
        assert turn_change.max() <= follow.TURN_ACCEL_MAX_RAD_S2 + 1e-3  # no snapping from left to right
        turned = follow.wrap_angle(env.person_heading - heading_before).abs() / POLICY_DT_S
        assert turned.max() <= TURN_MAX_RAD_S + 1e-4
        # A jogger can't turn on the spot: speed × turn rate ≤ the sideways-acceleration limit.
        assert (turned * env.person_speed).max() <= SIDEWAYS_ACCEL_MAX_MPS2 + 1e-3
        accel = (env.person_speed - speed_before) / POLICY_DT_S
        assert accel.max() <= follow.SPEED_UP_MPS2 + 1e-3 and accel.min() >= -follow.SLOW_DOWN_MPS2 - 1e-3
        max_speed = max(max_speed, env.person_speed.max().item())
        max_xy = max(max_xy, env.person_xy.abs().max().item())
        stood |= env.person_speed < 0.05
        jogged |= env.person_speed > 2.4
    assert max_xy <= ARENA_M
    assert max_speed <= JOG_SPEED_MPS[1] + 1e-5
    assert stood.float().mean() > 0.3 and jogged.float().mean() > 0.15  # many stopped, some jogged, in 30 s


def test_mode_targets_by_hand():
    env = make(4, mode="follow")
    xy = torch.tensor([[1.0, 2.0]] * 4)
    vel = torch.tensor([[1.0, 0.0]] * 4)
    heading = torch.tensor([0.0, math.pi / 2, math.pi, 0.0])
    turn = torch.tensor([0.0, 0.0, 0.0, 0.5])

    env.mode[:] = FOLLOW
    target, target_vel = env.mode_target(xy, vel, heading, turn)
    # 2 m behind, 1.5 m up: heading +x -> (−1, 2); heading +y -> (1, 0); heading −x -> (3, 2).
    assert torch.allclose(target[:3], torch.tensor([[-1.0, 2.0, 1.5], [1.0, 0.0, 1.5], [3.0, 2.0, 1.5]]), atol=1e-5)
    # Turning at 0.5 rad/s swings the point 2 m behind at 1 m/s sideways (to −y when turning left from +x).
    assert torch.allclose(target_vel[3], torch.tensor([1.0, -1.0, 0.0]), atol=1e-5)

    env.mode[:] = ORBIT
    env.orbit_angle[:] = torch.tensor([0.0, math.pi / 2, 0.0, 0.0])
    env.orbit_rate[:] = 0.5
    target, target_vel = env.mode_target(xy, vel, heading, turn)
    assert torch.allclose(target[0], torch.tensor([3.0, 2.0, 1.5]), atol=1e-5)
    assert torch.allclose(target[1], torch.tensor([1.0, 4.0, 1.5]), atol=1e-5)
    assert torch.allclose(target_vel[0], torch.tensor([1.0, 1.0, 0.0]), atol=1e-5)  # walk + 2 m × 0.5 rad/s

    env.mode[:] = POINT
    env.has_spot[:] = torch.tensor([True, True, False, False])
    env.spot[:] = torch.tensor([4.0, -3.0, 1.5])
    target, target_vel = env.mode_target(xy, vel, heading, turn)
    assert torch.allclose(target[0], torch.tensor([4.0, -3.0, 1.5]))
    assert torch.allclose(target_vel[0], torch.zeros(3))
    assert torch.allclose(target[2], torch.tensor([3.0, 2.0, 1.5]), atol=1e-5)  # no spot yet: the follow point


def test_orbit_target_goes_round_at_the_set_rate():
    env = make(8, mode="orbit")
    env.person_speed[:] = 0.0  # a standing person, so only the orbit moves the target
    env.turn_drive[:] = 0.0
    env.speed_goal[:] = 0.0
    env.update_targets()
    start = env.orbit_angle.clone()
    for _ in range(50):  # 1 s
        env._post_step()
    turned = follow.wrap_angle(env.orbit_angle - start)
    assert torch.allclose(turned, env.orbit_rate * 1.0, atol=1e-4)
    radius = (env.target[:, :2] - env.person_xy).norm(dim=1)
    assert torch.allclose(radius, torch.full_like(radius, follow.ORBIT_RADIUS_M), atol=1e-5)


def test_pointing_waits_for_the_person_to_stop_and_spots_are_2_to_4_m_away():
    env = make(256, seed=2, mode="point", person_sensor="perfect")
    env.person_speed[:] = 2.0  # walking into point mode: they must stop first
    saw_spot_while_moving = False
    for _ in range(300):
        moving = env.person_speed > 0.1
        had = env.has_spot.clone()
        env._post_step()
        new = env.has_spot & ~had
        saw_spot_while_moving |= bool((new & moving).any())
    assert not saw_spot_while_moving
    assert env.has_spot.all()
    assert env.person_speed.max() == 0.0  # pointing = standing still
    reach = (env.spot[:, :2] - env.person_xy).norm(dim=1)
    assert reach.min() >= 2.0 - 1e-4 and reach.max() <= 4.0 + 1e-4
    assert env.spot[:, :2].abs().max() <= follow.SPOT_LIMIT_M
    assert torch.allclose(env.spot[:, 2], torch.full((256,), 1.5))


def test_easy_curriculum_keeps_spots_on_the_drones_side_and_no_jogging():
    env = make(512, seed=7, mode="point", person_sensor="perfect", difficulty=0.0)
    assert env.random_gait_speeds(5000).max() <= 0.75
    assert env.random_orbit_rates(5000).abs().max() <= 0.3 + 1e-6
    env.person_xy = env.uniform(512, -1.0, 1.0, size=2)  # near the middle: every direction fits in the arena
    env.tracker.seen_xy = env.person_xy.clone()
    park_drones(env, torch.cat((env.person_xy + torch.tensor([2.0, 0.0]), torch.full((512, 1), 1.5)), dim=1))
    env.point_new_spots(torch.ones(512, dtype=torch.bool))
    to_spot = env.spot[:, :2] - env.person_xy
    angle = torch.atan2(to_spot[:, 1], to_spot[:, 0]).abs()  # the drone is on the +x side
    assert angle.max() <= math.pi / 3 + 1e-4


def test_spots_at_the_arena_edge_are_not_behind_the_person():
    """A person at the edge, the drone outside them: the spot must not be through the person (the old flip bug)."""
    env = make(512, seed=8, mode="point", person_sensor="perfect", difficulty=0.0)
    env.person_xy = torch.tensor([[4.8, 0.0]]).repeat(512, 1)
    env.tracker.seen_xy = env.person_xy.clone()
    park_drones(env, torch.tensor([[6.8, 0.0, 1.5]]).repeat(512, 1))
    env.point_new_spots(torch.ones(512, dtype=torch.bool))
    assert env.spot[:, :2].abs().max() <= follow.SPOT_LIMIT_M + 1e-5
    reach = (env.spot[:, :2] - env.person_xy).norm(dim=1)
    assert reach.min() >= 1.2 - 1e-4 and reach.max() <= 4.0 + 1e-4  # pulled in from the edge, never on the person
    # The spots stay on the drone's side: none is behind the person (more than 90° away from the drone's side).
    to_spot = env.spot[:, :2] - env.person_xy
    angle = torch.atan2(to_spot[:, 1], to_spot[:, 0]).abs()
    assert angle.max() <= math.pi / 2 + 1e-4


def test_curriculum_climbs_when_crashes_are_rare():
    env = make(8, difficulty=None)
    assert env.difficulty == 0.0
    for _ in range(30):
        env.update_curriculum({"episode_crashed": 0.0})
    assert env.difficulty > 0.2
    stuck = make(8, difficulty=None)
    for _ in range(30):
        stuck.update_curriculum({"episode_crashed": 0.2})
    assert stuck.difficulty == pytest.approx(30 / follow.CURRICULUM_RAMP, abs=1e-3)  # only the slow floor
    for _ in range(follow.CURRICULUM_RAMP):
        stuck.update_curriculum({"episode_crashed": 0.2})
    assert stuck.difficulty == 1.0  # the full task always arrives


def test_person_distance_is_to_the_body_surface():
    person = torch.zeros(4, 2)
    drones = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 2.75], [0.1, 0.0, 1.0], [1.25, 0.0, 2.75]])
    gap = person_distance(drones, person)
    assert torch.allclose(gap, torch.tensor([0.75, 1.0, 0.0, math.sqrt(2.0)]), atol=1e-5)


def test_too_close_to_the_person_is_a_crash_and_near_is_a_soft_penalty():
    env = make(4, mode="follow")
    env.person_xy[:] = 0.0
    env.person_speed[:] = 0.0  # standing, so the closing-speed term is 0 for still drones
    # Gaps to the body: 0.6 m (crash), 0.85 m (soft zone), 1.2 m (fine), 3 m (fine).
    drones = torch.tensor([[0.85, 0.0, 1.0], [1.10, 0.0, 1.0], [1.45, 0.0, 1.0], [3.25, 0.0, 1.0]])
    park_drones(env, drones)
    env.target = drones.clone()  # on target, so only the person terms differ
    crashed = env._crashed()
    assert crashed.tolist() == [True, False, False, False]
    reward = env._reward()
    assert reward[0].item() < -25.0  # the −30
    assert reward[2].item() == pytest.approx(reward[3].item(), abs=1e-5)  # 1.2 m: no penalty
    assert reward[3].item() - reward[1].item() == pytest.approx(0.5, abs=1e-4)  # half-way through the soft zone

    # Through `step`: the crash ends the episode with the penalty.
    env.reset()
    env.person_xy[:] = 0.0
    park_drones(env, drones)
    _, reward, done, info = env.step(torch.zeros(4, 4))
    assert done[0] and info["terminated"][0] and reward[0].item() < -25.0
    assert info["episodes"]["hit_person"][0].item() == 1.0


def test_flying_at_the_person_costs_more_than_flying_away():
    env = make(3, mode="follow")
    env.person_xy[:] = 0.0
    env.person_speed[:] = 0.0
    drones = torch.tensor([[1.5, 0.0, 1.5]] * 3)  # 1.25 m from the body: half of the closing cost
    park_drones(env, drones)
    env.target = drones.clone()
    env.state[:, VEL] = torch.tensor([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])  # toward, away, past
    assert torch.allclose(env.closing_speed(), torch.tensor([2.0, 0.0, 0.0]))
    reward = env._reward()
    assert reward[1].item() - reward[0].item() == pytest.approx(0.3 * 2.0 * 0.5, abs=1e-4)
    assert reward[1].item() == pytest.approx(reward[2].item(), abs=1e-5)


def test_low_near_the_person_is_a_crash():
    env = make(3, mode="follow")
    env.person_xy[:] = 0.0
    park_drones(env, torch.tensor([[1.5, 0.0, 0.4], [3.0, 0.0, 0.4], [1.5, 0.0, 0.6]]))
    assert env._crashed().tolist() == [True, False, False]


def test_reward_never_negative_before_a_crash():
    """4 m from the target and right at the 0.7 m line (but level and calm): the reward is still ≥ 0."""
    env = make(1, mode="follow")
    env.person_xy[:] = 0.0
    park_drones(env, torch.tensor([[0.951, 0.0, 1.0]]))
    env.target = torch.tensor([[0.951, 4.0, 1.0]])
    assert not env._crashed()[0]
    assert env._reward()[0].item() >= -1e-3


def test_with_a_perfect_sensor_the_estimate_is_the_truth():
    env = make(8, person_sensor="perfect")
    for _ in range(10):
        env._post_step()
    assert torch.allclose(env.seen_xy, env.person_xy)
    assert torch.allclose(env.seen_target, env.target)


def test_flights_start_looking_at_the_person():
    env = make(512, seed=9)
    rel = env.person_xy - env.state[:, POS][:, :2]
    off = follow.wrap_angle(torch.atan2(rel[:, 1], rel[:, 0]) - env.drone_yaw()).abs()
    assert off.max() <= follow.START_YAW_RAD + 0.1  # (+ a little: the start tilt moves the nose's heading too)
    assert env.tracker.visible.all()


def test_camera_view_bonus_only_while_in_view():
    env = make(2, mode="follow")
    env.person_xy[:] = 0.0
    env.person_speed[:] = 0.0
    park_drones(env, torch.tensor([[2.25, 0.0, 1.5]] * 2))
    env.target = env.state[:, POS].clone()
    env.tracker.visible = torch.tensor([True, False])
    reward = env._reward()
    assert reward[0].item() - reward[1].item() == pytest.approx(follow.VIEW_BONUS, abs=1e-5)


def test_scores_follow_time_after_settling_and_reached_spots():
    env = make(4, mode="follow", episode_s=6.0)
    for _ in range(env.max_steps - 1):
        park_drones(env, env.target.clone())  # a perfect drone: always on the true target
        env._post_step()
    settle = round(follow.SETTLE_S / POLICY_DT_S)
    assert torch.allclose(env.follow_scored, torch.full((4,), env.max_steps - 1 - settle, dtype=torch.float32))
    assert torch.equal(env.follow_good, env.follow_scored)

    env = make(4, mode="point", episode_s=20.0)
    for _ in range(env.max_steps - 1):
        if env.has_spot.any():
            park_drones(env, torch.where(env.has_spot[:, None], env.spot, env.target))
        env._post_step()
    info = env._episode_info(torch.arange(4))
    assert info["spots"].min() >= 2  # 4–6 s per spot in 20 s
    assert torch.equal(info["spots_reached"], info["spots"])
    assert torch.allclose(info["success"], torch.ones(4))


def test_modes_switch_mid_episode_only_when_asked():
    env = make(256, seed=6)
    start = env.mode.clone()
    assert set(start.tolist()) == {FOLLOW, ORBIT, POINT}
    switched = torch.zeros(256, dtype=torch.bool)
    for _ in range(round(10.5 / POLICY_DT_S)):
        env._post_step()
        switched |= env.mode != start
    assert switched.all()  # every drone switched within 10 s (switches come every 4–10 s)

    fixed = make(64, mode="orbit")
    for _ in range(600):
        fixed._post_step()
    assert (fixed.mode == ORBIT).all()


def test_display_objects_and_viewer_scene():
    from mindflight.rl.ppo import ActorCritic
    from mindflight.viewer.policy_source import SHOW, PolicySource

    assert "follow" in SHOW
    env = make(4, mode="point", person_sensor="perfect")
    env.person_speed[:] = 0.0
    for _ in range(30):
        env._post_step()
    objects = env.display_objects()
    assert objects["cylinders"].shape == (4, 1, 5)
    assert objects["spheres"].shape == (4, 2, 4)
    assert objects["lines"].shape == (4, 1, 6)
    assert torch.allclose(objects["cylinders"][:, 0, :2], env.person_xy)
    head_top = objects["spheres"][:, 0, 2] + objects["spheres"][:, 0, 3]
    assert torch.allclose(head_top, torch.full((4,), follow.PERSON_HEIGHT_M))
    assert torch.allclose(objects["lines"][:, 0, 3:6], env.spot)  # the pointing line ends at the spot

    source = PolicySource(env, ActorCritic(env.obs_size, env.action_size), layout="grid", spacing_m=16.0)
    source.env.person_speed[:] = 0.0
    for _ in range(30):
        source.env._post_step()
    scene = source.scene()
    assert len(scene["cylinders"]) == 4 and len(scene["spheres"]) == 8 and len(scene["lines"]) == 4
    assert len(scene["targets"]) == 4

    walking = make(4, mode="follow")
    assert torch.equal(walking.display_objects()["lines"], torch.zeros(4, 1, 6))  # no pointing line


def test_widened_policy_flies_like_the_old_one():
    """scripts/widen_policy.py on waypoint -> follow: the new inputs start with zero weights, so nothing changes."""
    from mindflight.rl.ppo import ActorCritic, PPOConfig

    widen_policy = load_script("widen_policy")
    torch.manual_seed(0)
    old = ActorCritic(22, 4, PPOConfig().hidden)
    with torch.no_grad():
        old.obs_norm.mean.uniform_(-1, 1)
        old.obs_norm.var.uniform_(0.5, 2)
        for p in old.actor.parameters():
            p.normal_(0, 0.1)
    follow_obs = make(2).obs_size
    new = widen_policy.widen(old, follow_obs)
    widen_policy.measure_new_columns(new, "follow", {"difficulty": 1.0}, 22, steps=5)
    obs = torch.randn(100, follow_obs)
    assert torch.allclose(new.act(obs, deterministic=True), old.act(obs[:, :22], deterministic=True), atol=1e-5)
    assert torch.allclose(new.value(obs), old.value(obs[:, :22]), atol=1e-5)


def load_script(name: str):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_teacher_follows_orbits_and_points_without_crashing():
    """scripts/teach_follow.py: the hand-written controller PPO starts from really does the job (short CPU check)."""
    from mindflight.rl.train import evaluate

    teach = load_script("teach_follow")
    for mode, key, bar in (("follow", "follow", 0.5), ("orbit", "orbit", 0.85)):
        env = make_env("follow", 64, device="cpu", seed=1, mode=mode, difficulty=1.0, episode_s=8.0)
        r = evaluate(teach.Teacher(env), env, seed=1)
        assert r["crashed"] == 0.0
        assert r[f"{key}_good_s"] / r[f"{key}_scored_s"] > bar
    env = make_env("follow", 64, device="cpu", seed=1, mode="point", difficulty=1.0, episode_s=10.0)
    r = evaluate(teach.Teacher(env), env, seed=1)
    assert r["crashed"] == 0.0 and r["spots_reached"] / r["spots"] > 0.9
