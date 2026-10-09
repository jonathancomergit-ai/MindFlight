import json
import subprocess
import sys

import pytest
import torch

from mindflight.envs import make_env
from mindflight.physics.dynamics import make_state, step
from mindflight.physics.params import CRAZYFLIE, DRONES_DIR, load_drone, save_drone
from dataclasses import replace


def test_crazyflie_drone_file_matches_the_built_in_numbers():
    assert load_drone("crazyflie_2_1_plus") == CRAZYFLIE
    assert load_drone() == CRAZYFLIE
    assert load_drone(CRAZYFLIE) is CRAZYFLIE


def test_unknown_drone_and_unknown_field_are_clear_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="known drones"):
        load_drone("no_such_drone")
    bad = tmp_path / "typo.json"
    bad.write_text(json.dumps({"mass_kgg": 0.03}))
    with pytest.raises(ValueError, match="unknown fields"):
        load_drone(str(bad))


def test_save_and_load_round_trip_keeps_numbers_and_skips_notes(tmp_path):
    heavy = replace(CRAZYFLIE, name="heavy", mass_kg=0.035)
    path = save_drone(heavy, tmp_path / "heavy.json", notes={"about": "test"})
    assert json.loads(path.read_text())["_about"] == "test"
    assert load_drone(str(path)) == heavy


def test_a_heavier_drone_sinks_at_the_old_hover_rpm(tmp_path):
    heavy = load_drone(str(save_drone(replace(CRAZYFLIE, name="heavy", mass_kg=0.035), tmp_path / "heavy.json")))
    rpm = torch.full((1, 4), CRAZYFLIE.hover_rpm)
    state = make_state(1, "cpu", pos=[0, 0, 1], rpm=rpm)
    for _ in range(250):
        state = step(state, rpm, params=heavy)
    assert state[0, 2] < 0.9
    # ...and hovers at its own hover RPM.
    rpm = torch.full((1, 4), heavy.hover_rpm)
    state = make_state(1, "cpu", pos=[0, 0, 1], rpm=rpm)
    for _ in range(250):
        state = step(state, rpm, params=heavy)
    assert abs(state[0, 2].item() - 1.0) < 1e-3


def test_envs_fly_the_drone_they_are_given(tmp_path):
    path = save_drone(replace(CRAZYFLIE, name="heavy", mass_kg=0.035), tmp_path / "heavy.json")
    env = make_env("hover", 4, device="cpu", drone=str(path))
    env.reset()
    assert env.drone == "heavy"
    assert torch.allclose(env.state[:, 13:17], torch.full((4, 4), env.params.hover_rpm))
    default = make_env("hover", 4, device="cpu")
    assert default.drone == "crazyflie_2_1_plus"


def test_fit_thrust_recovers_kf_from_measurements(tmp_path):
    kf_true, km_true = 4.0e-10, 9.0e-12
    lines = ["rpm,thrust_g,torque_nmm"]
    for rpm in range(6000, 24001, 2000):
        thrust_g = kf_true * rpm**2 / 9.81 * 1000 * (1 + 0.005 * ((rpm // 2000) % 3 - 1))  # ±0.5 % noise
        lines.append(f"{rpm},{thrust_g:.4f},{km_true * rpm**2 * 1000:.6f}")
    csv_path = tmp_path / "stand.csv"
    csv_path.write_text("\n".join(lines))
    name = "pytest_fit_drone"
    try:
        out = subprocess.run(
            [sys.executable, "scripts/fit_thrust.py", str(csv_path), "--name", name, "--mass-g", "40"],
            capture_output=True, text=True, check=True,
        ).stdout
        drone = load_drone(name)
        assert drone.kf_n_per_rpm2 == pytest.approx(kf_true, rel=0.01)
        assert drone.km_nm_per_rpm2 == pytest.approx(km_true, rel=0.01)
        assert drone.mass_kg == pytest.approx(0.040)
        assert "R²" in out
    finally:
        (DRONES_DIR / f"{name}.json").unlink(missing_ok=True)
