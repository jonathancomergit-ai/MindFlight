# MindFlight plan

A learning project and proof of concept: a **GPU-parallel Crazyflie 2.x simulator** in PyTorch where thousands of
drones learn to fly with **reinforcement learning (PPO)**, watched live in a **3D browser viewer**.

Goals, in order:
1. **Accurate.** Physics uses published Crazyflie numbers and every piece has a test that proves it.
2. **Readable.** Someone learning ML should be able to read any file top to bottom. No magic frameworks.
3. **Visual.** You can always *see* what the drones are doing and what the brain is learning.

---

## How it fits together

```
 ┌──────────── GPU (PyTorch) ────────────┐        ┌──── browser ────┐
 │ physics  →  env (task + reward)  →  PPO │ ──ws──▶ │ Three.js viewer │
 │ N drones = N rows of one big tensor     │        │ swarm + charts  │
 └─────────────────────────────────────────┘        └─────────────────┘
```

| Folder | What lives there |
|---|---|
| `mindflight/physics/` | Crazyflie params, rotation math, motor model, rigid-body step. **No ML here.** |
| `mindflight/envs/` | Tasks (hover, waypoint, gates, obstacles): observations, reward, reset, done |
| `mindflight/rl/` | PPO in one readable file, the policy network, training loop, checkpoints |
| `mindflight/viewer/` | FastAPI server + WebSocket stream + `static/` Three.js page (no build step) |
| `tests/` | pytest. Physics tests are the proof that the sim is accurate |
| `docs/` | Notes that teach the concepts (one short page per level) |

---

## Conventions (everyone follows these, so parts fit together)

| Thing | Rule |
|---|---|
| Units | SI everywhere: m, s, kg, N, rad. Motor speed in **RPM** (matches the reference params) |
| World frame | **Z up.** x forward, y left. Gravity = (0, 0, −9.81) |
| Body frame | FLU: x forward, y left, z up (thrust is +z in body) |
| Quaternion | **(w, x, y, z)**, rotates body → world, always unit length (renormalize every step) |
| Batch shape | Every physics function takes tensors shaped **(N, ...)**: N drones at once, no Python loops over drones |
| State | `(N, 17)`: pos 0:3 (world), quat 3:7, lin vel 7:10 (world), ang vel 10:13 (**body**), motor RPM 13:17 |
| Device | Functions run on whatever device their input tensors are on. Default `cuda` if available |
| dtype | `float32` |
| Timing | Physics **500 Hz** (`dt = 0.002`), policy **50 Hz** (10 physics substeps per action) |

### Motor layout (X configuration, seen from above)

```
        front (+x)
   m3 (CW)    m0 (CCW)
         \    /
          [CF]        +y is left
         /    \
   m2 (CCW)   m1 (CW)
```

| Motor | Position (x, y) | Spin |
|---|---|---|
| m0 | (+d, −d) front-right | CCW |
| m1 | (−d, −d) back-right | CW |
| m2 | (−d, +d) back-left | CCW |
| m3 | (+d, +d) front-left | CW |

`d = arm / √2`. Per motor: thrust `F_i = kf · rpm_i²`, drag torque `Q_i = km · rpm_i²`.

| Body torque | Formula (from τ = r × F, and spin reaction) |
|---|---|
| roll τx | `d · (−F0 − F1 + F2 + F3)` |
| pitch τy | `d · (−F0 + F1 + F2 − F3)` |
| yaw τz | `−Q0 + Q1 − Q2 + Q3` (a CCW prop pushes the body CW) |

Tests must check these signs physically (e.g. spinning up the left motors rolls the drone right, toward −y).

---

## Crazyflie 2.x numbers

Source: `cf2x.urdf` from [utiasDSL/gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones), which cites
Förster 2015 (ETH, Crazyflie 2.0 system identification), Landry 2015 (MIT) and Luis & Le Ny 2016.

| Param | Value | Unit |
|---|---|---|
| mass | 0.027 | kg |
| arm | 0.0397 | m |
| Ixx, Iyy, Izz | 1.4e-5, 1.4e-5, 2.17e-5 | kg·m² |
| kf (thrust coeff) | 3.16e-10 | N / RPM² |
| km (torque coeff) | 7.94e-12 | N·m / RPM² |
| thrust-to-weight | 2.25 | – |
| max speed | 30 | km/h |
| prop radius | 2.31348e-2 | m |
| drag coeff xy / z | 9.1785e-7 / 10.311e-7 | – |
| ground effect coeff | 11.36859 | – |
| collision cylinder | r 0.06, h 0.025 | m |

Derived (tests check these): hover RPM = √(m·g / 4kf) ≈ **14,476**, max RPM = hover · √2.25 ≈ **21,714**.

**Assumed, not from the source** (mark clearly in code, revisit later): motor time constant **0.02 s**.

---

## Levels

Each level ends with something you can **see** in the viewer and a short `docs/levelN.md` explaining the idea.

| # | Level | Done when | You learn |
|---|---|---|---|
| 0 | **Physics** | Free fall, hover, roll/pitch/yaw sign tests pass; 4,096 drones step on GPU; viewer shows them | Vectors, Newton, quaternions, integration |
| 1 | **Hover** | PPO holds a drone at 1 m for 10 s from random starts | Reward, policy, value, PPO |
| 2 | **Fly to a point** | Reaches random targets within 10 cm | Observations, normalization |
| 3 | **Gates** | Flies a course of gates in order | Curriculum learning |
| 4 | **Obstacles** | Dodges pillars using fake lidar rays | Sensors, raycasting |
| 5 | **Sturdy** | Survives wind gusts and ±20 % mass | Domain randomization |
| 6 | **Race** | Best policies race in the viewer | Evaluation, putting it together |

Later physics upgrades (only after Level 1 works): body drag, ground effect, downwash. **All three are in
`physics/aero.py`** (drag and ground effect used in Level 5; downwash is a race option, `--downwash`).

Status (2026-10-09): **all levels done**. Results and how to run each one: README "Where it stands".

---

## Level 0 issues (first batch)

| Issue | Depends on | Can run in parallel with |
|---|---|---|
| A. Batched quaternion / rotation math + tests | – | B, D |
| B. Motor model: RPM command → RPM with lag + clamp, thrust & torques per the table above + tests | – | A, D |
| C. Rigid-body step + integrator + physics tests (free fall, hover, signs, quat norm) | A, B | D |
| D. Viewer v0: FastAPI + WebSocket + Three.js page showing N drones from a fake stream | – | A, B |
| E. Benchmark: steps/sec for N = 1 … 16,384 on GPU, written to `docs/benchmark.md` | C | – |
| F. Wire sim → viewer: `python -m mindflight.demo` shows 64 drones falling / hovering live | C, D | – |

---

# Phase 2: from the simulator to real drones

The long-term goal: the brain is trained here, the drone is custom-built, and one day that brain
flies the real thing. Then: packs, following a person, flying where they point, and intercepting things thrown at
them. Until the custom drone is ready, everything targets a stock **Crazyflie 2.1+** (the drone the sim already
models), so the whole path can be proven on real hardware first. Swapping drones later = a new drone file.

## How a trained brain will fly a real Crazyflie

```
 PC                                                     Crazyflie (stock firmware)
 ┌──────────────────────────────┐   Crazyradio 2.0     ┌──────────────────────────────┐
 │ state estimate (from drone) ─┼──◀── log stream ─────┤ state estimator (Flow/Lighthouse)
 │ → policy (ONNX, 50 Hz)       │                      │                              │
 │ → roll°, pitch°, yaw-rate°/s,│── send_setpoint ───▶ │ onboard attitude + rate PID  │
 │   thrust (10001–60000)       │   (≥ every 0.5 s)    │ → motors (500 Hz)            │
 └──────────────────────────────┘                      └──────────────────────────────┘
```

So the policy must learn to give **attitude setpoints**, not raw motor speeds, and the sim must contain a
stand-in for the onboard controller, noisy and late state estimates, and radio delay.

## Phase 2 conventions (additions to the contract above)

| Thing | Rule |
|---|---|
| Drone file | `drones/<name>.json` holds every physical number (today's `CrazyflieParams`). Code loads it with `load_drone(name)`. Default `crazyflie_2_1_plus`. |
| Attitude action | 4 numbers in [−1, 1]: collective thrust, roll, pitch, yaw rate. Thrust: `T = hover · (1 + 1.25 · a)` clamped to [0, max]; roll/pitch: `a · 30°`; yaw rate: `a · 180 °/s` |
| Onboard controller | cascaded: angle P → body-rate PI(D) → torque → motor mixing, at the physics rate (500 Hz). Lives in `physics/controller.py`. Approximates the stock firmware in physical units (documented gains, not the firmware's PWM-space numbers). |
| Sensor model | observations come from a noisy, delayed *estimate* of the state; noise and delay are per-drone random ("profiles": `ideal`, `flowdeck`, `lighthouse`) |
| Old tasks | keep working unchanged with `action_mode="motors"` (the default for them); real-ready policies use `action_mode="attitude"` |

## Phase 2 issues

| Issue | What | Depends on |
|---|---|---|
| P1. Drone files | `drones/crazyflie_2_1_plus.json`, `load_drone`, `--drone` flag; `scripts/fit_thrust.py` turns thrust-stand measurements into a drone file (for the custom drone) | – |
| P2. Onboard controller + attitude actions | `physics/controller.py`, CUDA-graph capture of controller + physics, `action_mode="attitude"`; tests: step responses, mixing signs; retrain hover + waypoint in attitude mode | P1 |
| P3. Sensors and delays | noise/bias/delay profiles, observation + action latency, more randomization (motor lag, thrust, mass, centre-of-mass offset, battery sag); retrain → `policies/real_hover.pt`, `real_waypoint.pt` | P2 |
| P4. Deploy | ONNX export; `python -m mindflight.deploy` offboard runner on cflib with safety (geofence, thrust limit, kill, takeoff/land), and a **software-in-the-loop** mode that fakes the Crazyflie (radio delay, watchdog, thrust scale) so the whole loop is tested before hardware; `docs/hardware.md` shopping list + first-flight checklist | P3 |
| P5. Packs | shared-world multi-drone env: neighbours in the observation, drone–drone collisions, downwash on; formation flying to waypoints | P2 |
| P6. Follow-me + pointing | simulated walking person; keep an offset / orbit; fly toward a pointed direction. Vision (pose detection) is documented as the real-world front end | P2 |
| P7. Intercept | balls thrown at a protected point; the drone must touch the ball first; noisy ball tracking; safety rules in the docs | P2 |
| P8. The custom drone | its measured numbers → drone file → retrain everything → fly | P1–P4 + the drone |

Each issue ends like the levels did: tests that check real physics, a trained policy that passes a written
"done when", something to watch in the viewer, and a short `docs/` page.
