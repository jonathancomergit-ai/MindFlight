"""P7 intercept: ball physics, aiming, catches, the safety distance, the tracker and the curriculum."""

import math

import pytest
import torch

from mindflight.envs import make_env
from mindflight.envs.intercept import (
    BALL_DRAG_PER_MASS,
    BALL_DT_S,
    BALL_RADIUS_M,
    CRASH_PENALTY,
    DRONE_RADIUS_M,
    GUARD_POINT,
    INTERCEPT_BONUS,
    PERSON_RADIUS_M,
    SAFETY_DISTANCE_M,
    TOUCH_DISTANCE_M,
    InterceptEnv,
    aim_throws,
    ball_step,
    distance_to_body,
    fly_for,
    allowed_pieces,
    nearest_on_path,
    predict_path,
    throw_settings,
)
from mindflight.physics.controller import attitude_action_to_setpoint
from mindflight.physics.dynamics import OMEGA, POS, QUAT, VEL
from mindflight.physics.params import GRAVITY_MPS2


def make(n=16, **kwargs) -> InterceptEnv:
    kwargs.setdefault("bank_size", 256)
    return make_env("intercept", n, device="cpu", seed=0, **kwargs)


def hold_drone(env: InterceptEnv, where: torch.Tensor) -> None:
    """Pin every drone at `where` (N, 3): level, still. (Used to test the ball, not the flying.)"""
    env.state[:, POS] = where
    env.state[:, VEL] = 0.0
    env.state[:, OMEGA] = 0.0
    env.state[:, QUAT] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    env.prev_pos = where.clone()


# ---------- ball physics ----------


def test_without_drag_the_ball_flies_the_exact_parabola():
    pos0 = torch.tensor([[0.0, 0.0, 2.0]])
    vel0 = torch.tensor([[7.0, -2.0, 5.0]])
    pos, vel = pos0.clone(), vel0.clone()
    steps = 600  # 1.2 s
    for _ in range(steps):
        pos, vel = ball_step(pos, vel, drag_per_mass=0.0)
    t = steps * BALL_DT_S
    expected = pos0 + vel0 * t
    expected[0, 2] -= 0.5 * GRAVITY_MPS2 * t**2
    assert torch.allclose(pos, expected, atol=1e-4)
    assert torch.allclose(vel, vel0 + torch.tensor([[0.0, 0.0, -GRAVITY_MPS2 * t]]), atol=1e-4)


def test_drag_alone_matches_the_exact_solution():
    # No gravity, straight line: v(t) = v0 / (1 + k v0 t), x(t) = ln(1 + k v0 t) / k.
    k, v0 = BALL_DRAG_PER_MASS, 15.0
    pos, vel = torch.zeros(1, 3), torch.tensor([[v0, 0.0, 0.0]])
    steps = 500  # 1 s
    for _ in range(steps):
        pos, vel = ball_step(pos, vel, gravity=0.0)
    t = steps * BALL_DT_S
    assert pos[0, 0].item() == pytest.approx(math.log(1 + k * v0 * t) / k, abs=5e-4)
    assert vel[0, 0].item() == pytest.approx(v0 / (1 + k * v0 * t), rel=1e-4)
    assert vel[0, 0].item() < 0.6 * v0  # a foam ball loses a lot: under 60 % of its speed after 1 s


def test_drag_slows_the_ball_and_shortens_the_throw():
    pos0, vel0 = torch.tensor([[0.0, 0.0, 1.5]]), torch.tensor([[12.0, 0.0, 4.0]])
    steps = torch.tensor([500])
    with_air, _ = fly_for(pos0, vel0, steps)
    a, b = pos0.clone(), vel0.clone()
    speeds = []
    for _ in range(500):
        a, b = ball_step(a, b, drag_per_mass=0.0)
    vacuum = a
    assert with_air[0, 0] < vacuum[0, 0] - 1.0  # more than a metre shorter after 1 s
    assert with_air[0, 2] < vacuum[0, 2]  # and lower
    # Energy (kinetic + potential, per kg) only ever goes down with drag.
    p, v = pos0.clone(), vel0.clone()
    for _ in range(300):
        p, v = ball_step(p, v)
        speeds.append(0.5 * (v**2).sum().item() + GRAVITY_MPS2 * p[0, 2].item())
    assert all(later < earlier for earlier, later in zip(speeds, speeds[1:]))


def test_aimed_throws_arrive_on_target_at_the_right_time():
    g = torch.Generator().manual_seed(1)
    n = 200
    release = torch.stack((6 + 4 * torch.rand(n, generator=g), torch.zeros(n), 1.3 + 0.7 * torch.rand(n, generator=g)), 1)
    target = torch.stack((torch.zeros(n), 0.2 * torch.rand(n, generator=g) - 0.1, 1.0 + 0.45 * torch.rand(n, generator=g)), 1)
    flight = 0.8 + 0.9 * torch.rand(n, generator=g)
    velocity = aim_throws(release, target, flight)
    end, _ = fly_for(release, velocity, (flight / BALL_DT_S).round().long())
    assert (end - target).norm(dim=1).max() < 0.002  # within 2 mm


# ---------- the env ----------


def test_shapes_and_attitude_mode_by_default():
    env = make()
    obs = env.reset()
    assert env.action_mode == "attitude"
    assert obs.shape == (16, 19 + 18)
    # No ball yet: the ball columns and the flag are 0; guard − drone is filled in.
    assert torch.all(obs[:, 19:30] == 0)
    assert torch.allclose(obs[:, 30:33], torch.tensor(GUARD_POINT) - env.state[:, POS])


def test_unblocked_throws_hit_the_person():
    env = make(32, difficulty=1.0)
    env.reset()
    far = torch.tensor([-3.0, 3.0, 1.5]).expand(32, 3).clone()  # behind the person, out of the way
    hits_near_body = 0
    for _ in range(500):  # 10 s: 3 throws each
        hold_drone(env, far)
        env.step(torch.zeros(32, 4))
        if env.hit.any():
            assert distance_to_body(env.event_pos[env.hit]).max() < BALL_RADIUS_M + 0.01  # the body, not the floor
            hits_near_body += int(env.hit.sum())
    assert env.intercepts.sum() == 0
    assert hits_near_body == env.throws.sum() == env.hits.sum()
    assert env.throws.min() >= 3


def test_throws_come_from_the_arc_at_the_set_flight_time():
    env = make(64, difficulty=1.0)
    arc, shortest, longest = throw_settings(1.0)
    env.reset()
    env.throw(torch.arange(64))
    angle = torch.atan2(env.ball_pos[:, 1], env.ball_pos[:, 0])
    assert angle.abs().max() <= arc + 1e-4
    assert env.ball_pos[:, :2].norm(dim=1).min() >= 6.0 - 1e-4
    assert env.bank["flight_s"].min() >= shortest - BALL_DT_S and env.bank["flight_s"].max() <= longest + BALL_DT_S


def test_a_drone_on_the_path_intercepts_and_one_beside_it_does_not():
    for offset, should_catch in ((0.0, True), (TOUCH_DISTANCE_M + 0.05, False)):
        env = make(8, difficulty=1.0)
        env.reset()
        env.throw(torch.arange(8))
        # A point on the ball's path 2 m from the person, moved sideways by `offset`.
        radius = env.path[:, :, :2].norm(dim=-1)
        index = (radius - 2.0).abs().argmin(dim=1)
        spot = env.path[torch.arange(8), index].clone()
        side = torch.stack((-spot[:, 1], spot[:, 0], torch.zeros(8)), dim=1)
        spot += offset * side / side.norm(dim=1, keepdim=True)
        caught = torch.zeros(8, dtype=torch.bool)
        bonus_seen = False
        for _ in range(100):
            hold_drone(env, spot)
            _, reward, _, _ = env.step(torch.zeros(8, 4))
            caught |= env.intercepted
            if env.intercepted.any():
                bonus_seen = bonus_seen or bool((reward[env.intercepted] > INTERCEPT_BONUS - 2).all())
        assert bool(caught.all()) == should_catch
        assert bool(caught.any()) == should_catch
        if should_catch:
            assert bonus_seen
            assert env.intercepts.sum() == 8 and env.hits.sum() == 0


def test_too_close_to_the_person_is_a_crash():
    env = make(2)
    env.reset()
    # Drone edge 0.45 m and 0.55 m from the body surface, at chest height.
    centre = PERSON_RADIUS_M + DRONE_RADIUS_M + torch.tensor([SAFETY_DISTANCE_M - 0.05, SAFETY_DISTANCE_M + 0.05])
    where = torch.stack((centre, torch.zeros(2), torch.full((2,), 1.2)), dim=1)
    hold_drone(env, where)
    _, reward, done, info = env.step(torch.zeros(2, 4))
    assert info["terminated"].tolist() == [True, False]
    assert reward[0] < -CRASH_PENALTY + 2 and reward[1] > 0
    # Above the head, the distance is measured to the top of the cylinder.
    env.reset()
    hold_drone(env, torch.tensor([[0.0, 0.0, 1.8 + 0.06 + 0.45], [0.0, 0.0, 1.8 + 0.06 + 0.55]]))
    _, _, _, info = env.step(torch.zeros(2, 4))
    assert info["terminated"].tolist() == [True, False]


def test_waiting_at_the_guard_point_pays_and_reward_is_never_negative_when_idle():
    env = make(4)
    env.reset()
    hold_drone(env, torch.tensor(GUARD_POINT).expand(4, 3).clone())
    _, reward, _, _ = env.step(torch.zeros(4, 4))
    assert reward.min() == pytest.approx(1.0, abs=0.05)
    env.reset()
    hold_drone(env, torch.tensor([[1.2, 4.0, 1.5], [-2.0, -2.0, 0.5], [3.0, 0.0, 3.0], [1.2, 0.0, 1.5]]))
    _, reward, _, _ = env.step(torch.zeros(4, 4))
    assert reward.min() >= 0.0


def test_tracker_is_late_and_zero_without_a_ball():
    env = make(4, tracker_noise=False)
    env.reset()
    env.delay[:] = torch.tensor([1, 2, 1, 2])
    hold_drone(env, torch.tensor(GUARD_POINT).expand(4, 3).clone())
    env.throw(torch.arange(4))
    env._record_ball()
    history = []
    for _ in range(3):
        hold_drone(env, torch.tensor(GUARD_POINT).expand(4, 3).clone())
        env.step(torch.zeros(4, 4))
        history.append(env.ball_pos.clone())
        hold_drone(env, torch.tensor(GUARD_POINT).expand(4, 3).clone())
    obs = env.observe()
    seen = obs[:, 19:22] + env.state[:, POS]
    assert torch.allclose(seen[0], history[-2][0], atol=1e-5)  # 1 step late
    assert torch.allclose(seen[1], history[-3][1], atol=1e-5)  # 2 steps late
    assert torch.all(obs[:, 25] == 1)


def test_tracker_error_model_matches_flying_from_the_wrong_start():
    # The 20 ms prediction stays within 1 cm of the 2 ms ball over a whole 1.2 s flight...
    pos, vel = torch.tensor([[8.0, 1.0, 1.6]]), torch.tensor([[-9.0, -1.0, 5.0]])
    true_path = predict_path(pos, vel)
    _, fine = fly_for(pos, vel, torch.tensor([600]), sample_every=10)
    assert (true_path - fine).norm(dim=-1).max() < 0.01
    # ...and "true path + pos error + vel error · t" (what the env uses) is close to really flying the prediction
    # from the wrong start: for the tracker's typical error, within 2.5 cm up to 0.8 s ahead. (Drag slowly damps a
    # velocity error, so the simple model makes the error grow a little too fast: it errs on the hard side.)
    pos_error, vel_error = torch.tensor([[0.015, -0.01, 0.01]]), torch.tensor([[0.1, -0.07, 0.08]])
    wrong_start = predict_path(pos + pos_error, vel + vel_error)
    ahead_s = torch.arange(true_path.shape[1]) * 0.02
    model = true_path + pos_error[:, None] + vel_error[:, None] * ahead_s[None, :, None]
    gap = (model - wrong_start).norm(dim=-1)[0]
    assert gap[:41].max() < 0.025
    assert (model - true_path).norm(dim=-1)[0, 40] > (wrong_start - true_path).norm(dim=-1)[0, 40]


def test_tracker_points_at_the_path_without_noise():
    env = make(8, difficulty=1.0, tracker_noise=False)
    env.reset()
    env.throw(torch.arange(8))
    drone = torch.tensor(GUARD_POINT).expand(8, 3).clone()
    for _ in range(10):  # 0.2 s into the flight
        hold_drone(env, drone)
        env.step(torch.zeros(8, 4))
    obs = env.observe()
    go_to, time_left = obs[:, 26:29], obs[:, 29]
    # "Go here" is on the true path (the tracker is late, but a late ball is still on the same curve)...
    index = torch.arange(env.path.shape[1] - 1)
    valid = (index[None, :] >= 0) & allowed_pieces(env.path)
    _, _, off_path = nearest_on_path(drone + go_to, env.path, valid)
    assert off_path.max() < 0.01
    # ...it is the nearest such point, and the ball gets there later, not before now.
    _, _, true_nearest = nearest_on_path(drone, env.path, valid)
    assert torch.allclose(go_to.norm(dim=1), true_nearest, atol=0.01)
    assert time_left.min() > 0


def test_curriculum_climbs_on_success_and_throws_get_harder():
    env = make(8)
    assert env.difficulty == 0.0
    easy_shortest = env.bank["flight_s"].min().item()
    for _ in range(60):  # each step up needs ~8 good iterations (the smoothed rate starts again from 0)
        env.update_curriculum({"episode_success": 1.0})
    assert env.difficulty >= 0.3
    env.throw(torch.arange(8))  # refills the bank at the new difficulty
    assert env.bank["flight_s"].min().item() < easy_shortest - 0.05  # faster throws in the mix
    assert throw_settings(1.0)[0] > throw_settings(0.0)[0]  # wider arc
    fixed = make(8, difficulty=0.5)
    for _ in range(30):
        fixed.update_curriculum({"episode_success": 1.0})
    assert fixed.difficulty == 0.5


def test_tilt_limit_option_reaches_the_controller():
    action = torch.tensor([[0.0, 1.0, -1.0, 0.0]])
    setpoint = attitude_action_to_setpoint(action)
    assert setpoint[0, 1].item() == pytest.approx(math.radians(30))
    env = make(1, max_tilt_deg=45.0)
    setpoint = env.command(action)
    assert setpoint[0, 1].item() == pytest.approx(math.radians(45))
    assert setpoint[0, 2].item() == pytest.approx(-math.radians(45))


def test_episode_info_counts_throws():
    env = make(4, difficulty=1.0)
    env.reset()
    far = torch.tensor([-3.0, 3.0, 1.5]).expand(4, 3).clone()
    episodes = None
    for _ in range(env.max_steps):
        hold_drone(env, far)
        _, _, done, info = env.step(torch.zeros(4, 4))
        if done.any():
            episodes = info["episodes"]
    assert episodes is not None
    assert episodes["throws"].min() >= 4  # 15 s: first throw at ~1.25 s, then every 2.5–3 s
    assert torch.all(episodes["success"] == 0) and torch.all(episodes["hits"] == episodes["throws"])


def test_viewer_scene_shows_person_ball_and_path():
    from mindflight.rl.ppo import ActorCritic
    from mindflight.viewer.policy_source import SHOW, PolicySource

    assert "intercept" in SHOW
    env = make(2, difficulty=1.0)
    source = PolicySource(env, ActorCritic(env.obs_size, env.action_size), layout="grid", spacing_m=12.0)
    env.throw(torch.arange(2))
    scene = source.scene()
    assert len(scene["cylinders"]) == 2  # one person per world
    radii = sorted({round(s[3], 3) for s in scene["spheres"]})
    assert BALL_RADIUS_M in radii and 0.13 in radii
    assert len(scene["lines"]) > 2 * 24  # safety rings plus the ball's path
