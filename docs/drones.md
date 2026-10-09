# Drone files

Every physical number the simulator uses lives in a **drone file**: `mindflight/drones/<name>.json`.
A different drone is a new file, not new code.

| File | Drone |
|---|---|
| `crazyflie_2_1_plus.json` | Bitcraze Crazyflie 2.1+ (the default) |

```
python -m mindflight.train --task hover --drone my_drone --view
python -m mindflight.play --policy policies/hover.pt --drone my_drone      # a trained brain on a different drone
python -m mindflight.evaluate policies/hover.pt --env drone=my_drone
```

Checkpoints remember which drone they trained on.

## Making a drone file for a new drone

What to measure, and how:

| Number | How to get it |
|---|---|
| `mass_kg` | kitchen scale, ready to fly (battery in) |
| `arm_m` | centre of the drone to a motor axle |
| `prop_radius_m` | half the prop diameter |
| `kf_n_per_rpm2`, `km_nm_per_rpm2` | thrust stand: thrust (and spin torque) at several RPMs, then `scripts/fit_thrust.py` |
| `thrust_to_weight` | worked out by `fit_thrust.py` from the top RPM |
| `ixx/iyy/izz_kgm2` | CAD model, or a pendulum (bifilar) test |
| `motor_time_constant_s` | step test: time for a motor to reach 63 % of a speed jump |

```
python scripts/fit_thrust.py stand.csv --name my_drone --mass-g 42 --arm-mm 46 --prop-radius-mm 25 --motor-tau-ms 30
```

The CSV has a header row `rpm,thrust_g,torque_nmm` (torque is optional). The script fits thrust = kf · rpm² by least
squares, prints how good the fit is (R²), and writes the file. Anything you didn't measure keeps the Crazyflie's
number, and the file says so in its `_not_measured` note. Open it and fill those in when you can.

Keys starting with `_` are notes and are ignored. A misspelled key is an error, so a typo can't silently do nothing.
