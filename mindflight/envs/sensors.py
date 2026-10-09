"""Fake sensors. Level 4: a ring of lidar rays that measure the distance to the nearest pillar or wall.

A real lidar spins a laser and times the echo. Ours does the geometry directly: for each ray, where does it first
hit something? Everything is batched: N drones × R rays × P pillars in one go on the GPU, no Python loops.

The world for the lidar is flat (2-D, seen from above): the pillars are vertical cylinders from floor to ceiling, so a
horizontal ray hits one exactly where its line hits the pillar's circle.
"""

import math

import torch


def ray_directions(heading_rad: torch.Tensor, n_rays: int) -> torch.Tensor:
    """(N, R, 2) unit directions of R rays spread evenly around the circle, ray 0 pointing along each drone's heading.

    heading_rad: (N,) which way the drone's nose points, seen from above (rad, 0 = world +x, counter-clockwise).
    Ray k points at heading + 2πk / R, so the rays turn with the drone, like a lidar bolted to it.
    """
    offsets = torch.arange(n_rays, device=heading_rad.device) * (2 * math.pi / n_rays)
    angles = heading_rad[:, None] + offsets[None, :]
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)


def ray_circle_distance(
    origin: torch.Tensor, directions: torch.Tensor, centres: torch.Tensor, radii: torch.Tensor, max_range_m: float
) -> torch.Tensor:
    """Distance along each ray to the first circle it hits, (N, R), capped at max_range_m.

    origin (N, 2), directions (N, R, 2) unit length, centres (N, P, 2), radii (N, P); a radius ≤ 0 means "no pillar".

    A point on the ray is o + t·u (t ≥ 0). It is on the circle when |o + t·u − c|² = r². With w = c − o that is
        t² − 2·t·(u·w) + |w|² − r² = 0   ->   t = b − √(b² − k),   b = u·w,  k = |w|² − r²
    No real root (b² < k) means the line misses; t < 0 means the circle is behind. If the drone is inside the
    circle (k < 0) the distance is 0.
    """
    w = centres[:, None, :, :] - origin[:, None, None, :]  # (N, 1, P, 2)
    b = (w * directions[:, :, None, :]).sum(-1)  # (N, R, P)
    k = (w**2).sum(-1) - radii[:, None, :] ** 2  # (N, 1, P)
    discriminant = b**2 - k
    t = b - torch.sqrt(discriminant.clamp_min(0.0))
    hit = (discriminant >= 0) & (t >= 0)
    inside = k < 0
    t = torch.where(inside, torch.zeros_like(t), t)
    hit = (hit | inside) & (radii[:, None, :] > 0)
    distance = torch.where(hit, t, torch.full_like(t, math.inf)).amin(dim=-1)
    return distance.clamp(max=max_range_m)


def ray_wall_distance(origin: torch.Tensor, directions: torch.Tensor, wall_y_m: float, max_range_m: float):
    """Distance along each ray to the side walls at y = +wall_y_m and y = −wall_y_m, (N, R), capped at max_range_m.

    A ray going toward +y (u_y > 0) meets y = W at t = (W − o_y) / u_y; toward −y it meets y = −W.
    """
    uy = directions[..., 1]
    oy = origin[:, None, 1]
    target_y = torch.where(uy > 0, torch.full_like(uy, wall_y_m), torch.full_like(uy, -wall_y_m))
    t = (target_y - oy) / uy.where(uy.abs() > 1e-9, torch.full_like(uy, 1e-9))
    t = torch.where(t >= 0, t, torch.full_like(t, math.inf))
    return t.clamp(max=max_range_m)
