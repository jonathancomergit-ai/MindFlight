"""Turn thrust-stand measurements into a drone file: `python scripts/fit_thrust.py stand.csv --name my_drone ...`

For a new drone (like a custom build), measure one motor + prop on a thrust stand at several speeds and save
a CSV with a header row:

    rpm,thrust_g,torque_nmm          (torque_nmm is optional: N·mm of spin torque the stand measured)
    8000,12.1,1.9
    12000,27.3,4.4
    ...

The model in the sim (physics/motors.py) is thrust = kf · rpm² and spin torque = km · rpm². This script finds the
kf (and km) that fit the measurements best, by least squares through zero:

    kf = Σ(F · rpm²) / Σ(rpm⁴)          (F in newtons = grams × 9.81 / 1000)

and reports how well the curve fits (R², 1.0 = perfect). It then writes mindflight/drones/<name>.json, starting from
the Crazyflie's numbers and replacing whatever you measured. Weigh the drone and measure its arm length (centre to
motor axle) and prop radius; inertia can come from a CAD model or a pendulum test, and the motor time constant from
a step test (time to reach 63 % of a speed jump).
"""

import argparse
import csv
from dataclasses import replace

from mindflight.physics.params import CRAZYFLIE, DRONES_DIR, GRAVITY_MPS2, save_drone


def fit_through_zero(x2: list[float], y: list[float]) -> tuple[float, float]:
    """Best k for y = k · x2 (least squares, no offset) and R² of that fit."""
    k = sum(a * b for a, b in zip(x2, y)) / sum(a * a for a in x2)
    mean = sum(y) / len(y)
    total = sum((v - mean) ** 2 for v in y)
    residual = sum((v - k * a) ** 2 for a, v in zip(x2, y))
    return k, 1.0 - residual / total if total > 0 else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(prog="python scripts/fit_thrust.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("csv", help="thrust-stand measurements: rpm,thrust_g[,torque_nmm]")
    parser.add_argument("--name", required=True, help="drone file name to write (mindflight/drones/<name>.json)")
    parser.add_argument("--mass-g", type=float, help="whole drone, ready to fly (g)")
    parser.add_argument("--arm-mm", type=float, help="centre of the drone to a motor axle (mm)")
    parser.add_argument("--prop-radius-mm", type=float)
    parser.add_argument("--inertia", type=float, nargs=3, metavar=("IXX", "IYY", "IZZ"), help="kg·m²")
    parser.add_argument("--motor-tau-ms", type=float, help="motor time constant (ms)")
    parser.add_argument("--max-rpm", type=float, help="full-throttle RPM (default: the highest RPM measured)")
    args = parser.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rpm = [float(r["rpm"]) for r in rows]
    thrust_n = [float(r["thrust_g"]) * GRAVITY_MPS2 / 1000 for r in rows]
    rpm2 = [v * v for v in rpm]
    kf, r2_thrust = fit_through_zero(rpm2, thrust_n)
    print(f"kf = {kf:.4e} N/RPM²   (fit R² = {r2_thrust:.4f}, {len(rows)} points)")

    changes = {"name": args.name, "kf_n_per_rpm2": kf}
    if rows and "torque_nmm" in rows[0] and rows[0]["torque_nmm"] not in ("", None):
        torque_nm = [float(r["torque_nmm"]) / 1000 for r in rows]
        km, r2_torque = fit_through_zero(rpm2, torque_nm)
        changes["km_nm_per_rpm2"] = km
        print(f"km = {km:.4e} N·m/RPM²   (fit R² = {r2_torque:.4f})")
    if args.mass_g:
        changes["mass_kg"] = args.mass_g / 1000
    if args.arm_mm:
        changes["arm_m"] = args.arm_mm / 1000
    if args.prop_radius_mm:
        changes["prop_radius_m"] = args.prop_radius_mm / 1000
    if args.inertia:
        changes.update(ixx_kgm2=args.inertia[0], iyy_kgm2=args.inertia[1], izz_kgm2=args.inertia[2])
    if args.motor_tau_ms:
        changes["motor_time_constant_s"] = args.motor_tau_ms / 1000

    mass_kg = changes.get("mass_kg", CRAZYFLIE.mass_kg)
    max_rpm = args.max_rpm or max(rpm)
    changes["thrust_to_weight"] = 4 * kf * max_rpm**2 / (mass_kg * GRAVITY_MPS2)
    drone = replace(CRAZYFLIE, **changes)
    print(f"thrust-to-weight = {drone.thrust_to_weight:.2f}, hover RPM = {drone.hover_rpm:,.0f}")
    if drone.thrust_to_weight < 1.5:
        print("warning: thrust-to-weight under 1.5 leaves little room to steer; most quadcopters need 2+")

    measured = sorted(changes)
    path = save_drone(
        drone,
        DRONES_DIR / f"{args.name}.json",
        notes={"about": f"made by scripts/fit_thrust.py from {args.csv}", "measured": ", ".join(measured),
               "not_measured": "everything else is still the Crazyflie's number: measure and edit it"},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
