import numpy as np
import pytest
from fastapi.testclient import TestClient

from mindflight.physics.params import CRAZYFLIE
from mindflight.viewer.__main__ import HOST
from mindflight.viewer.server import FakeSwarm, create_app


@pytest.mark.parametrize("n", [1, 64, 4096])
def test_websocket_frame_is_n_by_7_float32(n):
    with TestClient(create_app(FakeSwarm(n))) as client, client.websocket_connect("/ws") as websocket:
        message = websocket.receive_bytes()
    assert len(message) == n * 7 * 4  # N drones × 7 values × 4 bytes per float32
    poses = np.frombuffer(message, dtype="<f4").reshape(n, 7)
    assert np.all(np.isfinite(poses))
    # Quaternions (columns 3:7) are unit length, as PLAN.md requires.
    assert np.allclose(np.linalg.norm(poses[:, 3:7], axis=1), 1.0, atol=1e-6)


def test_fake_swarm_bobs_in_a_grid_and_yaws_about_z():
    swarm = FakeSwarm(64, spacing_m=0.5, hover_height_m=1.0, bob_amplitude_m=0.2)
    poses = swarm.frame(1.234)
    assert poses.shape == (64, 7) and poses.dtype == np.float32
    # 8 × 8 grid, 0.5 m apart, centred on the origin.
    assert sorted(set(np.round(poses[:, 0], 4))) == pytest.approx(np.arange(-1.75, 1.76, 0.5))
    assert np.allclose(poses[:, 0].mean(), 0) and np.allclose(poses[:, 1].mean(), 0)
    # Heights stay within hover ± amplitude.
    assert np.all(np.abs(poses[:, 2] - 1.0) <= 0.2 + 1e-6)
    # Pure yaw: qx = qy = 0.
    assert np.all(poses[:, 4:6] == 0)
    # And they actually move.
    assert not np.allclose(swarm.frame(0.0)[:, 2], swarm.frame(0.5)[:, 2])


def test_page_and_config_are_served():
    client = TestClient(create_app(FakeSwarm(8)))
    page = client.get("/")
    assert page.status_code == 200 and "importmap" in page.text
    assert client.get("/static/main.js").status_code == 200
    config = client.get("/config").json()
    assert config["n_drones"] == 8
    assert config["arm_m"] == CRAZYFLIE.arm_m


def test_server_listens_on_localhost_only():
    assert HOST == "127.0.0.1"
