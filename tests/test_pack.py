"""P5 packs: worlds of G drones, neighbours in the observation, collisions, grouped downwash, the success rule."""

import math

import pytest
import torch

from mindflight.envs import make_env
from mindflight.envs.pack import CRASH_PENALTY, SPACING_M, PackEnv, formation_links, formation_offsets, start_cells
from mindflight.physics.aero import DroneVariation, downwash_grouped, downwash_world
from mindflight.physics.dynamics import OMEGA, POS, QUAT, RPM, VEL, make_state, step
from mindflight.physics.params import CRAZYFLIE


def pack(n=12, **kwargs) -> PackEnv:
    env = make_env("pack", n, device="cpu", seed=0, **kwargs)
    env.reset()
    return env


def put(env: PackEnv, positions: torch.Tensor, velocities: torch.Tensor | None = None) -> None:
    """Place every drone: level, motors at hover, at `positions` (N, 3), moving at `velocities` (or still)."""
    env.state[:] = 0.0
    env.state[:, QUAT.start] = 1.0
    env.state[:, RPM] = CRAZYFLIE.hover_rpm
    env.state[:, POS] = positions
    if velocities is not None:
        env.state[:, VEL] = velocities
    env.prev_pos = positions.clone()


# ---------- shapes and grouping ----------


def test_shapes_world_grouping_and_rounding_to_whole_packs():
    env = pack(12)
    assert env.obs_size == 19 + 3 + 6 * 3 + 1 + 3 * 3 == 50  # 3 nearest pack-mates
    assert env.observe().shape == (12, 50)
    assert make_env("pack", 12, device="cpu", share_goals=False).obs_size == 41
    assert make_env("pack", 12, device="cpu", neighbours=5).obs_size == 68  # the whole pack
    assert env.world_id.tolist() == [0] * 6 + [1] * 6
    assert env.action_mode == "attitude"  # real-ready setpoints by default
    assert make_env("pack", 1000, device="cpu").n_envs == 1002  # rounded up to whole packs of 6
    small = make_env("pack", 8, device="cpu", group_size=4, neighbours=2)
    assert small.n_envs == 8 and small.obs_size == 19 + 3 + 12 + 1 + 6


def test_pack_mates_start_together_and_targets_are_formation_slots():
    env = pack(12, formations="hexagon")
    pos = env.per_world(env.state[:, POS])
    assert (pos.max(dim=1).values - pos.min(dim=1).values).max() < 2.3  # each pack starts within ~2 m
    slots = env.slot_positions(torch.arange(2))
    targets = env.per_world(env.target)
    for w in range(2):
        # Each drone's target is one of the slots, and every slot is taken once (any assignment).
        match = (targets[w, :, None] - slots[w, None]).norm(dim=-1) < 1e-5
        assert (match.sum(dim=0) == 1).all() and (match.sum(dim=1) == 1).all()
        gaps = (slots[w, :, None] - slots[w, None]).norm(dim=-1)
        links = formation_links("hexagon", 6)
        ring = torch.tensor([gaps[k, links[k]].item() for k in range(6)])
        assert torch.allclose(ring, torch.full((6,), SPACING_M), atol=1e-5)  # hexagon sides = spacing
        assert slots[w, :, 2].std() < 1e-6  # flat formation
    # start points are distinct and at least 0.8 m apart (before the ±5 cm jitter)
    cells = start_cells(6)
    gaps = (cells[:, None] - cells[None]).norm(dim=-1) + torch.eye(len(cells)) * 9
    assert len(cells) == 18 and gaps.min() >= 0.8 - 1e-6


def test_formation_shapes():
    for shape in ("hexagon", "line", "vee"):
        offsets = formation_offsets(shape, 6, 0.7)
        assert torch.allclose(offsets.mean(0), torch.zeros(3), atol=1e-6)  # centred
        links = formation_links(shape, 6)
        linked = [(offsets[k] - offsets[links[k]]).norm().item() for k in range(6) if links[k] != k]
        assert linked and all(d == pytest.approx(0.7, abs=1e-5) for d in linked)  # drawn links = spacing
    vee, links = formation_offsets("vee", 6, 0.7), formation_links("vee", 6)
    leader = [k for k in range(6) if links[k] == k][0]
    assert vee[leader, 0] == vee[:, 0].max()  # the leader is in front
    for shape in ("hexagon", "line", "vee"):  # slots are numbered from right (−y) to left (+y)
        y = formation_offsets(shape, 6, 0.7)[:, 1]
        assert (y[1:] >= y[:-1] - 1e-6).all()
    with pytest.raises(ValueError):
        formation_offsets("blob", 6, 0.7)


# ---------- observation ----------


def test_neighbour_observation_is_sorted_and_relative():
    env = pack(6, neighbours=5)
    positions = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.5, 1.0], [0.0, 0.0, 3.0], [-2.0, 0.0, 1.0], [0.0, -0.7, 1.0]]
    )
    velocities = torch.zeros(6, 3)
    velocities[2] = torch.tensor([0.1, 0.2, 0.3])
    velocities[0] = torch.tensor([0.0, 0.0, 0.1])
    put(env, positions, velocities)
    task = env.observe()[0, 19:]
    assert torch.allclose(task[:3], env.target[0] - positions[0])
    mates = task[3:33].reshape(5, 6)
    # drone 0's pack-mates by distance: 2 (0.5 m), 5 (0.7), 1 (1.0), 3 (2.0), 4 (2.0, tie with 3)
    assert torch.allclose(mates[0], torch.tensor([0.0, 0.5, 0.0, 0.1, 0.2, 0.2]))  # rel pos, then rel vel
    assert torch.allclose(mates[1, :3], torch.tensor([0.0, -0.7, 0.0]))
    assert torch.allclose(mates[2, :3], torch.tensor([1.0, 0.0, 0.0]))
    assert sorted(mates[3:, :3].norm(dim=1).tolist()) == pytest.approx([2.0, 2.0])
    assert task[33].item() == pytest.approx(0.5)  # distance to the nearest
    goals = task[34:49].reshape(5, 3)  # where each of those pack-mates is heading, same order
    assert torch.allclose(goals[0], env.target[2] - positions[2])
    assert torch.allclose(goals[2], env.target[1] - positions[1])


# ---------- collisions and whole-pack resets ----------


def test_collision_crashes_both_and_restarts_only_that_pack():
    env = pack(12)
    positions = torch.cat((env.target[:6], env.target[6:]))
    positions[1] = positions[0] + torch.tensor([0.10, 0.0, 0.0])  # 10 cm < 2 × 6 cm: a collision
    put(env, positions)
    env.step_count[:] = 10
    _, reward, done, info = env.step(torch.zeros(12, 4))
    assert info["terminated"][:2].all() and not info["terminated"][2:].any()
    assert info["truncated"][2:6].all()  # pack-mates: cut short, not blamed
    assert not done[6:].any()  # the other pack never noticed
    assert (reward[:2] < -CRASH_PENALTY + 4).all() and (reward[2:] >= 0).all()
    assert info["episodes"]["collided"].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert info["episodes"]["pack_collided"].tolist() == [1.0] * 6
    assert (env.step_count[:6] == 0).all() and (env.step_count[6:] == 11).all()  # pack 0 started over


def test_collision_check_follows_the_path_and_ignores_other_worlds():
    env = pack(12)
    far = torch.tensor([[5.0 * k, 0.0, 1.0] for k in range(12)])
    put(env, far)
    env.prev_pos[0], env.state[0, POS] = torch.tensor([0.0, 0.0, 1.0]), torch.tensor([1.0, 0.0, 1.0])
    env.prev_pos[1], env.state[1, POS] = torch.tensor([1.0, 0.0, 1.0]), torch.tensor([0.0, 0.0, 1.0])
    hit = env.collisions()
    assert hit[:2].all() and not hit[2:].any()  # swapped places in one step: they met in the middle

    put(env, far)
    env.prev_pos[0], env.state[0, POS] = torch.tensor([0.0, 0.0, 1.0]), torch.tensor([1.0, 0.0, 1.0])
    env.prev_pos[1], env.state[1, POS] = torch.tensor([0.0, 0.2, 1.0]), torch.tensor([1.0, 0.2, 1.0])
    assert not env.collisions().any()  # side by side, 20 cm apart: fine

    put(env, far)
    env.state[6, POS] = env.state[0, POS]  # same spot, different worlds: they never meet
    env.prev_pos[6] = env.prev_pos[0]
    assert not env.collisions().any()


def test_reward_stays_non_negative_when_far_and_crowded():
    env = pack(6)
    ring = torch.tensor([[0.15 * math.cos(a), 0.15 * math.sin(a), 0.0] for a in torch.arange(5) * 2 * math.pi / 5])
    positions = torch.cat((torch.tensor([[0.0, 0.0, 1.5]]), ring + torch.tensor([0.0, 0.0, 1.5])))
    put(env, positions)
    env.target[0] = torch.tensor([0.0, 0.0, 1.5]) + torch.tensor([3.0, 3.0, 0.0])  # 4.2 m away
    assert env.crowding()[0].item() == pytest.approx(1.0)  # five pack-mates at 15 cm: the cap
    assert not env._crashed().any()  # 15 cm apart is crowded, not a collision
    assert env._reward()[0].item() >= -1e-6


# ---------- success ----------


def test_waypoint_counts_only_when_the_whole_pack_has_settled():
    env = pack(12)
    positions = env.target.clone()
    positions[7] += torch.tensor([0.2, 0.0, 0.0])  # one drone of pack 1 sits 20 cm off its slot
    put(env, positions)
    env.step_count[:] = env.hold_steps - 30  # the formation moves on in 30 steps (0.6 s)
    old_centre = env.centre.clone()
    for _ in range(30):
        env.step(torch.zeros(12, 4))  # zero action = hold still (attitude mode)
    assert env.waypoints_reached.tolist() == [1.0, 0.0]
    assert not torch.allclose(env.centre, old_centre)  # new waypoints
    assert env.close_steps.max() <= 1  # the settle count starts again


def test_episode_success_is_the_fraction_of_waypoints():
    env = pack(6, episode_s=12.0)
    assert env.waypoints_per_episode == 4
    env.waypoints_reached[0] = 3.0
    info = env._episode_info(torch.arange(6))
    assert info["success"].tolist() == [0.75] * 6


def test_staggered_reset_gives_each_pack_one_clock():
    env = make_env("pack", 600, device="cpu", seed=2)
    env.reset(stagger=True)
    clocks = env.per_world(env.step_count)
    assert (clocks == clocks[:, :1]).all()
    assert clocks[:, 0].float().std() > env.max_steps / 4


# ---------- grouped downwash ----------


def test_grouped_downwash_equals_the_all_pairs_version():
    gen = torch.Generator().manual_seed(0)
    for worlds, group in ((1, 2), (3, 4), (5, 6)):
        pos = torch.rand(worlds * group, 3, generator=gen) * torch.tensor([0.3, 0.3, 2.0])  # stacked: real pushes
        world_id = torch.arange(worlds * group) // group
        expected = downwash_world(pos, world_id)
        grouped = downwash_grouped(pos.reshape(worlds, group, 3)).reshape(-1, 3)
        assert expected[:, 2].abs().max() > 0.001  # the case has real downwash in it
        torch.testing.assert_close(grouped, expected, rtol=1e-6, atol=1e-9)


def test_step_with_group_size_matches_world_id_downwash():
    pos = torch.tensor([[0.0, 0.0, 1.0], [0.02, 0.0, 2.0], [0.0, 0.0, 1.5], [0.0, 0.01, 1.0]])
    rpm = torch.full((4, 4), CRAZYFLIE.hover_rpm)
    state = make_state(4, "cpu", pos=pos, rpm=rpm)
    by_id = DroneVariation(world_id=torch.tensor([0, 0, 1, 1]), downwash=True)
    by_group = DroneVariation(world_id=torch.tensor([0, 0, 1, 1]), downwash=True, group_size=2)
    a, b = state, state
    for _ in range(50):
        a, b = step(a, rpm, variation=by_id), step(b, rpm, variation=by_group)
    torch.testing.assert_close(a, b)
    assert a[0, 2] < 0.999 and a[2, 2] == pytest.approx(1.5, abs=1e-4)  # under a pack-mate: pushed down


def test_pack_env_turns_downwash_on_and_off():
    assert pack(6).variation.downwash and pack(6).variation.group_size == 6
    assert pack(6, downwash=False).variation is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graph_stepper_with_grouped_downwash_matches_eager():
    from mindflight.physics.fast import CudaGraphStepper

    n, group = 60, 6
    gen = torch.Generator().manual_seed(1)
    pos = (torch.rand(n, 3, generator=gen) * torch.tensor([0.2, 0.2, 1.5]) + torch.tensor([0, 0, 0.8])).cuda()
    variation = DroneVariation(world_id=torch.arange(n, device="cuda") // group, downwash=True, group_size=group)
    stepper = CudaGraphStepper(n, "cuda", substeps=10, variation=variation, compile=True)
    rpm = torch.full((n, 4), CRAZYFLIE.hover_rpm, device="cuda")
    state = make_state(n, "cuda", pos=pos, rpm=rpm)
    graph_state = stepper.run(state, rpm, variation=variation)
    eager = state
    for _ in range(10):
        eager = step(eager, rpm, variation=variation)
    torch.testing.assert_close(graph_state, eager, atol=1e-5, rtol=1e-5)
    assert (graph_state[:, 2] < pos[:, 2] - 1e-4).any()  # someone really was pushed down


def test_runs_on_gpu_with_cuda_graph():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    env = make_env("pack", 60, device="cuda", seed=0)
    obs = env.reset()
    for _ in range(3):
        obs, reward, done, info = env.step(torch.zeros(60, 4, device="cuda"))
    assert env.stepper is not None and obs.shape == (60, 50) and torch.isfinite(obs).all()
    assert env.state[:, OMEGA].abs().max() < 40


def test_formation_can_change_shape_at_a_waypoint():
    for chance, changes in ((0.0, False), (1.0, True)):
        env = make_env("pack", 600, device="cpu", seed=3, reshape_chance=chance)
        env.reset()
        before = env.shape.clone()
        env.step_count[:] = env.hold_steps - 1
        env.step(torch.zeros(600, 4))
        assert (env.shape != before).any() == changes  # ~2/3 of 100 packs pick a different shape at chance 1


# ---------- viewer ----------


def test_viewer_puts_each_pack_on_its_own_grid_spot_and_draws_the_formation():
    from mindflight.rl.ppo import ActorCritic
    from mindflight.viewer.policy_source import SHOW, PolicySource

    env = make_env("pack", 24, device="cpu", seed=0, formations="hexagon")
    source = PolicySource(env, ActorCritic(env.obs_size, env.action_size), layout="grid", spacing_m=9.0)
    offsets = source.offsets
    assert torch.equal(offsets[:6], offsets[:1].expand(6, 3))  # one offset per pack, shared by its drones
    assert len({tuple(o) for o in offsets[::6].tolist()}) == 4  # 4 packs, 4 different spots
    scene = source.scene()
    assert len(scene["targets"]) == 24  # every slot
    assert len(scene["lines"]) == 24  # hexagon: a ring of 6 links per pack
    first = scene["lines"][0]
    assert first[:3] == [round(v, 3) for v in (env.target[0] + offsets[0]).tolist()]
    partner = formation_links("hexagon", 6)[0]
    assert first[3:6] == [round(v, 3) for v in (env.target[partner] + offsets[0]).tolist()]
    assert SHOW["pack"][1] % 6 == 0


def test_next_centres_are_short_steps_inside_the_box():
    env = make_env("pack", 600, device="cpu", seed=4)
    env.reset()
    worlds = torch.arange(env.n_worlds)
    before = env.centre.clone()
    after = env.next_centres(worlds)
    step_m = (after - before).norm(dim=1)
    assert step_m.max() <= 2.0 + 1e-5 and step_m.median() > 0.5  # clamping at the walls can only shorten a step
    assert after[:, :2].abs().max() <= 1.25 + 1e-6 and after[:, 2].min() >= 1.0 and after[:, 2].max() <= 2.0


def test_crowding_warns_about_closing_in_not_flying_side_by_side():
    env = pack(6)
    positions = torch.tensor([[0.0, 0.0, 1.0], [0.8, 0.0, 1.0]] + [[5.0 * k, 5.0, 1.0] for k in range(4)])
    put(env, positions)
    assert env.crowding()[0].item() == 0.0  # 80 cm apart and still: fine
    toward = torch.zeros(6, 3)
    toward[1, 0] = -2.0  # drone 1 flies at drone 0 at 2 m/s
    put(env, positions, toward)
    assert env.crowding()[0].item() == pytest.approx(0.5 * 2.0 * (1 - 0.8))
    together = torch.zeros(6, 3)
    together[:2, 1] = 2.0  # both fly the same way side by side: the gap isn't shrinking
    put(env, positions, together)
    assert env.crowding()[0].item() == 0.0


def test_ordered_start_puts_drone_k_kth_from_the_right():
    env = make_env("pack", 600, device="cpu", seed=5)
    env.reset()
    pos = env.per_world(env.state[:, POS]) - env.centre[:, None, :]
    h = env.heading[:, None]
    left = -torch.sin(h) * pos[..., 0] + torch.cos(h) * pos[..., 1]  # in the first formation's frame
    assert ((left[:, 1:] - left[:, :-1]) > -0.11).all()  # right to left, give or take the ±5 cm jitter
    scattered = make_env("pack", 600, device="cpu", seed=5, start="scattered")
    scattered.reset()
    pos = scattered.per_world(scattered.state[:, POS]) - scattered.centre[:, None, :]
    h = scattered.heading[:, None]
    left = -torch.sin(h) * pos[..., 0] + torch.cos(h) * pos[..., 1]
    assert ((left[:, 1:] - left[:, :-1]) < -0.2).any()


def test_real_world_and_noisy_sensors_keep_working_with_downwash():
    env = make_env("pack", 12, device="cpu", seed=0, real_world=True, sensors="lighthouse")
    v = env.variation
    assert v.downwash and v.group_size == 6 and v.mass_scale is not None and v.com_offset_m is not None
    obs = env.reset()
    for _ in range(5):
        obs, reward, done, info = env.step(torch.zeros(12, 4))
    assert obs.shape == (12, env.obs_size) and torch.isfinite(obs).all()


def test_best_assignment_gives_each_drone_the_nearby_slot_and_no_crossings():
    from mindflight.envs import make_env

    env = make_env("pack", 6, device="cpu", seed=0, assignment="best")
    env.reset()
    worlds = torch.tensor([0])
    slots = env.slot_positions(worlds)  # (1, 6, 3)
    # Put the drones exactly on the slots, but in reversed order: drone k sits on slot 5 − k.
    env.state[:6, 0:3] = slots[0].flip(0)
    targets = env.assign_slots(worlds, slots)[0]
    assert torch.allclose(targets, env.state[:6, 0:3])  # each drone keeps the slot it's already on
    fixed = make_env("pack", 6, device="cpu", seed=0, assignment="fixed")
    assert torch.equal(fixed.assign_slots(worlds, slots), slots)
