"""Batched quaternion and rotation math.

Conventions (see PLAN.md):
- A quaternion is stored as (w, x, y, z): w is the scalar part, (x, y, z) the vector part.
- An orientation quaternion rotates vectors from the **body** frame into the **world** frame.
- Every function takes tensors shaped (N, ...), one row per drone, and runs on whatever device they are on.

A unit quaternion for a rotation of `angle` radians about the unit axis `a` is

    q = (cos(angle / 2), sin(angle / 2) * a)

and rotating a vector v by it is v_world = q ⊗ (0, v_body) ⊗ q*.
"""

import torch

from mindflight.physics.device_cache import cached_on_device


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 ⊗ q2, shapes (N, 4) -> (N, 4).

    Applying q1 ⊗ q2 to a vector means: first rotate by q2, then by q1.
    With q = (w, v) split into scalar w and vector v:

        q1 ⊗ q2 = (w1 w2 − v1 · v2,  w1 v2 + w2 v1 + v1 × v2)
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


@cached_on_device
def conjugate_signs_on(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """(1, −1, −1, −1) as a (4,) tensor already on `device`, made once so `quat_conjugate` never copies from the CPU."""
    return torch.tensor([1.0, -1.0, -1.0, -1.0], device=device, dtype=dtype)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate q* = (w, −x, −y, −z), shape (N, 4).

    For a unit quaternion this is the inverse rotation: q ⊗ q* = (1, 0, 0, 0).
    """
    return q * conjugate_signs_on(q.device, q.dtype)


def quat_normalize(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Scale q back to unit length, q / |q|, shape (N, 4).

    Integration slowly drifts |q| away from 1; renormalizing keeps it a pure rotation.
    `eps` only guards against dividing by zero.
    """
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate body-frame vectors into the world frame: v_world = q ⊗ v_body ⊗ q*.

    q: (N, 4) unit quaternions, v: (N, 3) vectors (any units). Returns (N, 3) in the same units.
    Uses the expanded form, with u = (x, y, z) the vector part of q:

        v' = v + 2 w (u × v) + 2 u × (u × v)
    """
    w = q[..., :1]
    u = q[..., 1:]
    uv = torch.linalg.cross(u, v, dim=-1)
    return v + 2.0 * (w * uv + torch.linalg.cross(u, uv, dim=-1))


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors into the body frame: v_body = q* ⊗ v_world ⊗ q.

    q: (N, 4) unit quaternions, v: (N, 3). Returns (N, 3) in the same units as v.
    """
    return quat_rotate(quat_conjugate(q), v)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Rotation matrix R for each quaternion, (N, 4) -> (N, 3, 3), so that v_world = R @ v_body.

        R = [[1 − 2(y² + z²),  2(xy − wz),      2(xz + wy)    ],
             [2(xy + wz),      1 − 2(x² + z²),  2(yz − wx)    ],
             [2(xz − wy),      2(yz + wx),      1 − 2(x² + y²)]]
    """
    w, x, y, z = q.unbind(-1)
    row0 = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), dim=-1)
    row1 = torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), dim=-1)
    row2 = torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def quat_from_axis_angle(axis: torch.Tensor, angle_rad: torch.Tensor) -> torch.Tensor:
    """Quaternion for a rotation of `angle_rad` (rad) about `axis`.

    axis: (N, 3), normalized here so it need not be unit length. angle_rad: (N,).
    Returns (N, 4):  q = (cos(angle / 2), sin(angle / 2) * axis / |axis|)
    Positive angles turn counterclockwise when looking down the axis toward the origin (right-hand rule).
    """
    unit_axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    half_angle = (0.5 * angle_rad).unsqueeze(-1)
    return torch.cat((torch.cos(half_angle), torch.sin(half_angle) * unit_axis), dim=-1)


def quat_to_euler(q: torch.Tensor) -> torch.Tensor:
    """Roll, pitch, yaw (rad) for each quaternion, (N, 4) -> (N, 3). For display and debugging only.

    Z-Y-X order: the rotation is yaw about world z, then pitch about the new y, then roll about the new x.

        roll  = atan2(2(wx + yz), 1 − 2(x² + y²))
        pitch = asin(2(wy − zx))                    (clamped to ±π/2 at the gimbal-lock poles)
        yaw   = atan2(2(wz + xy), 1 − 2(y² + z²))
    """
    w, x, y, z = q.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return torch.stack((roll, pitch, yaw), dim=-1)


def quat_integrate(q: torch.Tensor, omega_body: torch.Tensor, dt: float) -> torch.Tensor:
    """Advance orientation by one step of body angular velocity, then renormalize.

    q: (N, 4) body → world, omega_body: (N, 3) in rad/s measured in the **body** frame, dt in s.
    Uses the exact exponential map, which is exact when ω is constant over the step:

        q_next = normalize(q ⊗ exp(½ ω dt))
        exp(½ ω dt) = (cos(θ), sin(θ) · ω / |ω|),   θ = ½ |ω| dt

    Body-frame ω multiplies on the right; a world-frame ω would multiply on the left.
    sin(θ) · ω / |ω| is written as ½ dt · ω · sin(θ)/θ so it stays finite when ω = 0.
    """
    half_rotation = 0.5 * dt * omega_body  # (N, 3), rad
    theta = half_rotation.norm(dim=-1, keepdim=True)  # (N, 1), rad
    sin_theta_over_theta = torch.sinc(theta / torch.pi)  # torch.sinc(x) = sin(πx) / (πx)
    delta_q = torch.cat((torch.cos(theta), sin_theta_over_theta * half_rotation), dim=-1)
    return quat_normalize(quat_multiply(q, delta_q))
