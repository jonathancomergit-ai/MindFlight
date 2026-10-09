"""A drone's physical parameters, and drone files.

Every physical number lives in a **drone file**, `mindflight/drones/<name>.json`, so a different drone (one day,
a custom build) is a new file, not new code. `load_drone(name)` reads one into a `CrazyflieParams`.

The default numbers below are the Crazyflie 2.x, and `drones/crazyflie_2_1_plus.json` holds exactly the same values
(a test checks). Source: cf2x.urdf in utiasDSL/gym-pybullet-drones, which cites Förster 2015 (ETH system
identification of the Crazyflie 2.0), Landry 2015 (MIT) and Luis & Le Ny 2016. Values not from that source are
marked ASSUMED.
"""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

GRAVITY_MPS2 = 9.81


DRONES_DIR = Path(__file__).resolve().parent.parent / "drones"
DEFAULT_DRONE = "crazyflie_2_1_plus"


@dataclass(frozen=True)
class CrazyflieParams:
    """Physical numbers for one quadcopter in the X layout (PLAN.md -> Motor layout). Named after the first drone it
    described; `DroneParams` is the same class for drones that aren't Crazyflies."""

    name: str = DEFAULT_DRONE
    mass_kg: float = 0.027
    arm_m: float = 0.0397
    ixx_kgm2: float = 1.4e-5
    iyy_kgm2: float = 1.4e-5
    izz_kgm2: float = 2.17e-5
    kf_n_per_rpm2: float = 3.16e-10  # thrust per motor = kf * rpm^2
    km_nm_per_rpm2: float = 7.94e-12  # drag torque per motor = km * rpm^2
    thrust_to_weight: float = 2.25
    max_speed_kmh: float = 30.0
    prop_radius_m: float = 2.31348e-2
    drag_coeff_xy: float = 9.1785e-7
    drag_coeff_z: float = 10.311e-7
    ground_effect_coeff: float = 11.36859
    # Downwash between drones (Level 6 race option). From gym-pybullet-drones BaseAviary (DW_COEFF_1..3), not the
    # URDF; an empirical fit, used as-is. See physics/aero.py -> downwash_world for the formula and its limits.
    downwash_coeff_1: float = 2267.18
    downwash_coeff_2: float = 0.16
    downwash_coeff_3: float = -0.11
    collision_radius_m: float = 0.06
    collision_height_m: float = 0.025
    motor_time_constant_s: float = 0.02  # ASSUMED: not in the source, revisit

    @property
    def weight_n(self) -> float:
        """Force of gravity on the drone, m * g (N)."""
        return self.mass_kg * GRAVITY_MPS2

    @property
    def hover_rpm(self) -> float:
        """RPM where 4 motors exactly hold up the weight: 4 * kf * rpm^2 = m * g."""
        return math.sqrt(self.weight_n / (4 * self.kf_n_per_rpm2))

    @property
    def max_rpm(self) -> float:
        """RPM at full throttle: total thrust = thrust_to_weight * weight."""
        return math.sqrt(self.thrust_to_weight * self.weight_n / (4 * self.kf_n_per_rpm2))

    @property
    def arm_xy_m(self) -> float:
        """d in PLAN.md: each motor's x and y offset in the X layout, arm / sqrt(2)."""
        return self.arm_m / math.sqrt(2)


DroneParams = CrazyflieParams

CRAZYFLIE = CrazyflieParams()


def load_drone(drone: "str | Path | CrazyflieParams | None" = None) -> CrazyflieParams:
    """Read a drone file -> its parameters. `drone` is a name in mindflight/drones/ (without .json), a path to a
    .json file, an already-made CrazyflieParams (returned as is), or None (the default Crazyflie 2.1+).

    The file is a JSON object of the field names above. Keys starting with "_" are notes (source, units) and are
    ignored; a missing field keeps its Crazyflie default, and an unknown field is an error (a typo would otherwise
    silently do nothing).
    """
    if isinstance(drone, CrazyflieParams):
        return drone
    path = Path(drone) if drone is not None and str(drone).endswith(".json") else DRONES_DIR / f"{drone or DEFAULT_DRONE}.json"
    if not path.exists():
        known = sorted(p.stem for p in DRONES_DIR.glob("*.json"))
        raise FileNotFoundError(f"no drone file {path}; known drones: {known}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    values = {k: v for k, v in raw.items() if not k.startswith("_")}
    allowed = {f.name for f in fields(CrazyflieParams)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"{path.name}: unknown fields {sorted(unknown)}; allowed: {sorted(allowed)}")
    values.setdefault("name", path.stem)
    return CrazyflieParams(**values)


def save_drone(params: CrazyflieParams, path: str | Path, notes: dict | None = None) -> Path:
    """Write `params` as a drone file (plus optional "_" note keys) and return the path."""
    path = Path(path)
    data = {f"_{k}": v for k, v in (notes or {}).items()}
    data.update(asdict(params))
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path
