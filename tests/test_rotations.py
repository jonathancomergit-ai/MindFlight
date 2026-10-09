import math

import pytest
import torch

from mindflight.physics.rotations import (
    quat_conjugate,
    quat_from_axis_angle,
    quat_integrate,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    quat_rotate_inverse,
    quat_to_euler,
    quat_to_matrix,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEVICES = ["cpu", pytest.param("cuda", marks=cuda_only)]
BATCH_SIZES = [1, 4096]


def random_unit_quats(n, device="cpu", seed=0):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=generator)
    return quat_normalize(q).to(device)


def random_vectors(n, device="cpu", seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, generator=generator).to(device)


def identity_quats(n, device="cpu"):
    q = torch.zeros(n, 4, device=device)
    q[:, 0] = 1.0
    return q


def same_rotation(q1, q2, atol):
    """q and −q are the same rotation, so compare up to sign."""
    return torch.allclose(q1, q2, atol=atol) or torch.allclose(q1, -q2, atol=atol)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", BATCH_SIZES)
def test_identity_leaves_vectors_unchanged(device, n):
    v = random_vectors(n, device)
    q = identity_quats(n, device)
    assert torch.equal(quat_rotate(q, v), v)
    assert torch.equal(quat_rotate_inverse(q, v), v)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", BATCH_SIZES)
def test_90_degrees_about_z_turns_x_into_y(device, n):
    z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(n, 3)
    q = quat_from_axis_angle(z_axis, torch.full((n,), math.pi / 2, device=device))
    plus_x = torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(n, 3)
    plus_y = torch.tensor([[0.0, 1.0, 0.0]], device=device).expand(n, 3)
    assert torch.allclose(quat_rotate(q, plus_x), plus_y, atol=1e-6)
    assert torch.allclose(quat_to_matrix(q) @ plus_x.unsqueeze(-1), plus_y.unsqueeze(-1), atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", BATCH_SIZES)
def test_rotate_inverse_undoes_rotate(device, n):
    q = random_unit_quats(n, device)
    v = random_vectors(n, device)
    assert torch.allclose(quat_rotate_inverse(q, quat_rotate(q, v)), v, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", BATCH_SIZES)
def test_matrix_matches_rotate(device, n):
    q = random_unit_quats(n, device)
    v = random_vectors(n, device)
    rotated_by_matrix = (quat_to_matrix(q) @ v.unsqueeze(-1)).squeeze(-1)
    assert quat_to_matrix(q).shape == (n, 3, 3)
    assert torch.allclose(rotated_by_matrix, quat_rotate(q, v), atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_rotation_matrix_is_orthonormal(device):
    matrices = quat_to_matrix(random_unit_quats(4096, device))
    identity = torch.eye(3, device=device).expand(4096, 3, 3)
    assert torch.allclose(matrices @ matrices.transpose(-1, -2), identity, atol=1e-5)
    assert torch.allclose(torch.linalg.det(matrices), torch.ones(4096, device=device), atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_multiply_composes_rotations(device):
    """Rotating by q1 ⊗ q2 equals rotating by q2, then by q1."""
    q1 = random_unit_quats(4096, device, seed=2)
    q2 = random_unit_quats(4096, device, seed=3)
    v = random_vectors(4096, device)
    composed = quat_rotate(quat_multiply(q1, q2), v)
    assert torch.allclose(composed, quat_rotate(q1, quat_rotate(q2, v)), atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_conjugate_is_inverse(device):
    q = random_unit_quats(4096, device)
    assert torch.allclose(quat_multiply(q, quat_conjugate(q)), identity_quats(4096, device), atol=1e-6)


def test_euler_angles_of_simple_rotations():
    axes = torch.eye(3)
    angles = torch.tensor([0.3, -0.4, 1.2])
    euler = quat_to_euler(quat_from_axis_angle(axes, angles))
    assert torch.allclose(euler, torch.diag(angles), atol=1e-6)


def test_euler_angles_of_combined_rotation():
    """Yaw, then pitch, then roll (Z-Y-X) comes back out as the same three angles."""
    roll, pitch, yaw = 0.2, -0.5, 2.0
    x_axis, y_axis, z_axis = torch.eye(3).unsqueeze(1)
    q = quat_multiply(
        quat_from_axis_angle(z_axis, torch.tensor([yaw])),
        quat_multiply(
            quat_from_axis_angle(y_axis, torch.tensor([pitch])),
            quat_from_axis_angle(x_axis, torch.tensor([roll])),
        ),
    )
    assert torch.allclose(quat_to_euler(q), torch.tensor([[roll, pitch, yaw]]), atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", BATCH_SIZES)
def test_integrating_constant_omega_matches_axis_angle(device, n):
    generator = torch.Generator().manual_seed(4)
    omega_body = (3.0 * torch.randn(n, 3, generator=generator)).to(device)  # rad/s
    dt, steps = 0.002, 500  # 1 s at 500 Hz
    q = identity_quats(n, device)
    for _ in range(steps):
        q = quat_integrate(q, omega_body, dt)

    speed = omega_body.norm(dim=-1)
    expected = quat_from_axis_angle(omega_body, speed * dt * steps)
    assert same_rotation(q, expected, atol=1e-4)


def test_integration_from_a_tilted_start_uses_body_frame():
    """Body ω composes on the right: pitched 90° about y, a spin about body z turns about world +x."""
    pitched = quat_from_axis_angle(torch.tensor([[0.0, 1.0, 0.0]]), torch.tensor([math.pi / 2]))
    omega_body = torch.tensor([[0.0, 0.0, 1.0]])  # rad/s about body z
    q = pitched
    for _ in range(100):
        q = quat_integrate(q, omega_body, 0.01)  # 1 rad in total
    expected = quat_multiply(pitched, quat_from_axis_angle(omega_body, torch.tensor([1.0])))
    assert same_rotation(q, expected, atol=1e-5)
    # Body z (thrust axis) is unchanged by spinning about it: still pointing along world +x.
    body_z_in_world = quat_rotate(q, omega_body)
    assert torch.allclose(body_z_in_world, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-5)


def test_zero_omega_leaves_orientation_unchanged():
    q = random_unit_quats(16)
    assert torch.allclose(quat_integrate(q, torch.zeros(16, 3), 0.002), q, atol=1e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_norm_stays_one_after_10000_steps(device):
    generator = torch.Generator().manual_seed(5)
    omega_body = (10.0 * torch.randn(4096, 3, generator=generator)).to(device)
    q = random_unit_quats(4096, device)
    for _ in range(10_000):
        q = quat_integrate(q, omega_body, 0.002)
    assert torch.allclose(q.norm(dim=-1), torch.ones(4096, device=device), atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_outputs_are_float32_on_input_device(device):
    q = random_unit_quats(8, device)
    v = random_vectors(8, device)
    outputs = [
        quat_multiply(q, q),
        quat_conjugate(q),
        quat_normalize(q),
        quat_rotate(q, v),
        quat_rotate_inverse(q, v),
        quat_to_matrix(q),
        quat_from_axis_angle(v, v[:, 0]),
        quat_to_euler(q),
        quat_integrate(q, v, 0.002),
    ]
    for output in outputs:
        assert output.dtype == torch.float32
        assert output.device.type == device
