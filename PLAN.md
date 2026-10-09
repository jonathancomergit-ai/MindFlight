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

Later physics upgrades (only after Level 1 works): body drag, ground effect, downwash.

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
