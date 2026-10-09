import math

import pytest
import torch

from mindflight.envs.sensors import ray_circle_distance, ray_directions, ray_wall_distance


def test_ray_directions_follow_the_heading():
    d = ray_directions(torch.tensor([0.0, math.pi / 2]), 4)
    assert d.shape == (2, 4, 2)
    assert d[0].flatten().tolist() == pytest.approx([1, 0, 0, 1, -1, 0, 0, -1], abs=1e-6)
    assert d[1, 0].tolist() == pytest.approx([0, 1], abs=1e-6)  # drone facing +y: ray 0 points +y


def test_known_hit_and_miss():
    origin = torch.tensor([[0.0, 0.0]])
    directions = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])  # +x, +y, −x
    centres = torch.tensor([[[2.0, 0.0]]])
    radii = torch.tensor([[0.5]])
    d = ray_circle_distance(origin, directions, centres, radii, max_range_m=3.0)
    assert d[0, 0].item() == pytest.approx(1.5)  # hits the near side of the circle: 2 − 0.5
    assert d[0, 1].item() == 3.0  # misses: out of range value
    assert d[0, 2].item() == 3.0  # circle is behind this ray


def test_glancing_hit_and_nearest_of_two():
    origin = torch.tensor([[0.0, 0.0]])
    directions = torch.tensor([[[1.0, 0.0]]])
    centres = torch.tensor([[[2.0, 0.4], [1.0, 0.0]]])
    radii = torch.tensor([[0.5, 0.1]])
    d = ray_circle_distance(origin, directions, centres, radii, max_range_m=3.0)
    assert d[0, 0].item() == pytest.approx(0.9)  # the small pillar at x = 1 is nearer
    # Only the far, offset pillar: t = 2 − √(0.5² − 0.4²) = 1.7
    d = ray_circle_distance(origin, directions, centres[:, :1], radii[:, :1], max_range_m=3.0)
    assert d[0, 0].item() == pytest.approx(1.7)


def test_empty_spots_and_inside_a_pillar():
    origin = torch.tensor([[0.0, 0.0]])
    directions = torch.tensor([[[1.0, 0.0]]])
    assert ray_circle_distance(origin, directions, torch.tensor([[[1.0, 0.0]]]), torch.tensor([[0.0]]), 3.0).item() == 3.0
    assert ray_circle_distance(origin, directions, torch.tensor([[[0.1, 0.0]]]), torch.tensor([[0.3]]), 3.0).item() == 0.0


def test_walls():
    origin = torch.tensor([[0.0, 1.0]])
    directions = torch.tensor([[[0.0, 1.0], [0.0, -1.0], [1.0, 0.0]]])
    d = ray_wall_distance(origin, directions, wall_y_m=3.0, max_range_m=10.0)
    assert d[0].tolist() == pytest.approx([2.0, 4.0, 10.0])


def test_obstacles_env_lidar_sees_a_pillar_in_front():
    from mindflight.envs.obstacles import N_RAYS, ObstaclesEnv
    from mindflight.physics.dynamics import POS, QUAT

    env = ObstaclesEnv(1, device="cpu", difficulty=1.0)
    obs = env.reset()
    assert obs.shape == (1, 19 + 3 + N_RAYS)
    env.state[0, POS] = torch.tensor([0.0, 0.0, 1.0])
    env.state[0, QUAT] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # facing +x
    env.pillar_radius[:] = 0.0
    env.pillar_xy[0, 0] = torch.tensor([1.0, 0.0])
    env.pillar_radius[0, 0] = 0.2
    lidar = env.scan()
    assert lidar[0, 0].item() == pytest.approx(0.8)  # ray 0 = straight ahead
    assert lidar[0, N_RAYS // 2].item() == 3.0  # straight behind: nothing
    env.state[0, POS] = torch.tensor([0.75, 0.0, 1.0])
    assert env.touching_pillar().tolist() == [True]  # 0.25 m from the axis < 0.2 + 0.06
