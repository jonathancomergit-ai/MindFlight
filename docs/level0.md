# Level 0: physics you can see

Watch it live:

```
python -m mindflight.demo --n 64
```

Open http://127.0.0.1:8000/.

- Press **R** (or the button) to start over.
- It also starts over by itself every 3 s.
- Slow motion: add `--speed 0.25`.
- Tick **size ×10** to see the tiny drones from far away.

---

## 1. Thrust: the push

Each spinning prop pushes the drone **up** (along its own body z axis).

```
F = kf · rpm²
```

- `kf` = 3.16e-10 N per RPM².
- Twice the speed gives **4 ×** the push. That is the "squared".

| Thing | Value |
|---|---|
| Drone weight, m · g | 0.027 kg × 9.81 = **0.265 N** |
| Hover speed | **14,476 RPM** |
| Push of one motor at hover | 0.265 N / 4 = **0.066 N** |

At hover speed, 4 motors hold up exactly the weight. So the drone floats.

If the drone tilts, the push tilts too. A tilted push moves the drone sideways.

---

## 2. Torque: the twist

Torque makes the drone **turn**. There are three kinds.

| Twist | Turns about | Made by |
|---|---|---|
| **roll** τx | x (nose) axis | more push on the left than the right |
| **pitch** τy | y (left) axis | more push at the back than the front |
| **yaw** τz | z (up) axis | the props dragging on the air |

The motor layout, seen from above (x forward, y left):

```
        front (+x)
   m3 (CW)    m0 (CCW)
         \    /
          [CF]        +y is left
         /    \
   m2 (CCW)   m1 (CW)
```

| Motor | Where | Spins |
|---|---|---|
| m0 | front-right | CCW |
| m1 | back-right | CW |
| m2 | back-left | CCW |
| m3 | front-left | CW |

The torque formulas (d = 0.028 m, the motor's distance from the centre along x and along y):

| Twist | Formula |
|---|---|
| roll τx | d · (−F0 − F1 + F2 + F3) |
| pitch τy | d · (−F0 + F1 + F2 − F3) |
| yaw τz | −Q0 + Q1 − Q2 + Q3 |

**Yaw needs one more idea.**

- A prop drags on the air: `Q = km · rpm²` (km = 7.94e-12 N·m per RPM²).
- The air drags back on the body, the **other way**.
- So a CCW prop turns the body CW, and a CW prop turns it CCW.
- Two props spin each way, so at hover the drags cancel.

Positive roll tips the right side (−y) down. Positive yaw turns the nose to the left (CCW from above).

---

## 3. The 4 groups in the demo

Every drone starts level at 1 m, motors already at their speed.

| Colour | Group | Motors | You see |
|---|---|---|---|
| green | hover | all at hover speed | stays put at 1 m |
| red | off | all off | falls, stops on the floor |
| blue | tip | left m2, m3 **+2 %** | tips right, drifts toward −y |
| purple | spin | CW m1, m3 **+2 %**, CCW m0, m2 **−2 %** | spins in place |

**Why each one does that:**

| Group | Why |
|---|---|
| hover | Push = weight. No tilt. No twist. Nothing changes. |
| off | No push. Only gravity: z = 1 − ½ · g · t². It hits the floor after **0.45 s**. |
| tip | Left push is bigger, so roll τx > 0. It leans right. Its push now points a bit toward −y, so it slides that way. Nothing levels it, so it keeps rolling, flips and falls. It lands about **1.3 m** toward −y after about **0.85 s**. |
| spin | CW props are faster, so their drag wins: yaw τz > 0, it turns CCW. Push left = push right and front = back, so no roll or pitch. It speeds up by about **12 rad/s every second**. There is no air drag on the body yet, so nothing slows it. |

Two small notes:

- **spin**: +2 % and −2 % gives a tiny bit more total push (0.04 %). The demo trims all four motors by 0.02 % so push = weight exactly.
- **Crash rule** (demo only): a drone that touches the floor has crashed. Its props stop and it lies still. The physics has no floor forces yet, so without this a crashed drone would keep tumbling.

---

## 4. What the tests prove

Run them all: `python -m pytest -q`

| Test | What it checks | Why it matters |
|---|---|---|
| free fall | after 1 s with motors off: fell ½ · g · t² = 4.905 m, speed g · t | gravity and the position update are right |
| hover | at hover speed, moves less than 1 mm in 5 s | thrust and weight balance |
| signs | left motors faster: rolls right, drifts −y. Front faster: nose up. CW faster: yaw +, keeps height | each torque points the right way, matching the table above |
| quaternion norm | 4,096 tumbling drones for 1 s: length stays 1 (within 0.00001) | a quaternion that drifts from length 1 would squash the drone |
| RK4 energy | a drone tumbling freely for 2 s keeps its spin energy and angular momentum within 0.1 % | the rotation step does not make up or lose energy |
| demo groups | after 1 s: hover within 1 cm, off on the floor, tip moved −y and rolled, spin yawed with no drift | what you see in the viewer is what the physics says |

The physics tests are in `tests/test_dynamics.py`, `tests/test_motors.py` and `tests/test_rotations.py`.
The demo tests are in `tests/test_demo.py`.
