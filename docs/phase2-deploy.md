# Phase 2 · P4: flying a real Crazyflie (and a fake one first)

**Why:** a trained brain is only useful if it can fly the real drone. The code that will fly it must be tested
before the drone arrives. So there is a fake drone (SITL, "software in the loop") that looks exactly like the real
one to our code.

```
python -m mindflight.deploy --policy policies/waypoint_attitude.pt --sitl --waypoints 0.5,0,0.5 0.5,0.5,0.8 0,0,0.6
python -m mindflight.deploy --policy policies/hover_attitude.pt --uri radio://0/80/2M/E7E7E7E7E7 --height 0.3
python -m mindflight.deploy.export policies/waypoint_attitude.pt     # -> policies/waypoint_attitude.onnx
```

What to buy and the first-flight checklist: `docs/hardware.md`.

## The loop, 50 times a second

```
 PC (runner.py)                                    Crazyflie
 ┌────────────────────────────────────┐   radio   ┌───────────────────────────┐
 │ 1. newest estimate  ◀──────────────┼── log ────┤ state estimator (100 Hz)  │
 │ 2. safety checks                   │           │                           │
 │ 3. mission: where to be now        │           │                           │
 │ 4. observation (shadow env)        │           │                           │
 │ 5. brain -> action                 │           │                           │
 │ 6. action -> setpoint (N, rad)     │           │                           │
 │ 7. interface.py -> degrees, thrust ┼─ setpoint ▶ attitude controller (500 Hz)│
 └────────────────────────────────────┘           └───────────────────────────┘
```

| Step | In plain words |
|---|---|
| estimate | the drone tells us where it is, how fast, how it is tilted, how fast it turns, its battery |
| safety | kill key? radio gone? tipped over? left the box? battery low? |
| mission | take off slowly to 0.5 m, hover, visit the waypoints, come down slowly, motors off |
| observation | built by the *training env's own code* (the "shadow env"), so it can never differ from training |
| brain | the trained network: PyTorch file or ONNX file |
| setpoint | thrust in newtons, roll and pitch in radians, yaw rate in rad/s: the same function training used |
| send | `interface.py` turns it into what the firmware wants |

### The shadow env

The brain was trained on observations from `env.observe()`. If we wrote our own copy of that for the real drone,
one day the two would drift apart. So we don't. We make a 1-drone copy of the training env, write the real estimate
into its state, set its target, and call its own `observe()`. It never steps.

Its simulated sensor noise and delays are switched off (`sensors="ideal"`, `real_world=False`): the real estimate
already has real noise and real delay.

### The carrot

The brain only ever saw targets up to about 1.5 m away. So it never sees one further than 0.6 m: the target it is
shown is "0.6 m along the way to the real target". Like a carrot on a stick.

## Our numbers vs the firmware's numbers

All of this lives in one file: `mindflight/deploy/interface.py`.

| Number | Ours | Sent to cflib `send_setpoint` |
|---|---|---|
| roll | rad, + = right side down | `+degrees(roll)` |
| pitch | rad, + = nose down | `+degrees(pitch)` (cflib flips it, the firmware flips it back) |
| yaw rate | rad/s, + = counter-clockwise | `−degrees(yaw rate)` (cflib's + is clockwise) |
| thrust | newtons, all 4 motors | `thrust / weight · hover command`, kept in 10001..60000 |

| Log variable | Unit | Ours |
|---|---|---|
| `stateEstimate.x/y/z`, `.vx/vy/vz` | m, m/s, world (z up) | same |
| `stateEstimate.qw/qx/qy/qz` | body -> world | same |
| `gyro.x/y/z` | deg/s, body | rad/s |
| `pm.vbat` | V | same |

Sources (read 2026-10-09):

| Fact | Where |
|---|---|
| cflib sends `-pitch` | cflib 0.1.34 `crazyflie/commander.py`, `send_setpoint` |
| firmware pitch is "legacy" (+ = nose up) | [kalman_core.c](https://github.com/bitcraze/crazyflie-firmware/blob/master/src/modules/src/kalman_core/kalman_core.c) (`.pitch = -pitch`), [controller_pid.c](https://github.com/bitcraze/crazyflie-firmware/blob/master/src/modules/src/controller/controller_pid.c) (`-sensors->gyro.y`) |
| yaw rate is flipped, thrust lock, MIN_THRUST | [crtp_commander_rpyt.c](https://github.com/bitcraze/crazyflie-firmware/blob/master/src/modules/src/crtp_commander_rpyt.c) |
| thrust number -> newtons | [motors.c](https://github.com/bitcraze/crazyflie-firmware/blob/master/src/drivers/src/motors.c) `motorsCompensateBatteryVoltage`, [platform_defaults_cf2.h](https://github.com/bitcraze/crazyflie-firmware/blob/master/src/platform/interface/platform_defaults_cf2.h), Bitcraze blog [Keeping thrust consistent as the battery drains](https://www.bitcraze.io/2025/10/keeping-thrust-consistent-as-the-battery-drains/) |
| watchdog 0.5 s / 2 s, lock, tumble | [supervisor.c](https://github.com/bitcraze/crazyflie-firmware/blob/master/src/modules/src/supervisor.c), [supervisor_state_machine.c](https://github.com/bitcraze/crazyflie-firmware/blob/master/src/modules/src/supervisor_state_machine.c) |
| log units | [stabilizer.c](https://github.com/bitcraze/crazyflie-firmware/blob/master/src/modules/src/stabilizer.c) log groups `stateEstimate`, `gyro`; `pm.vbat` in pm_stm32f4.c |
| 26 bytes per log block, 10 ms log steps | cflib `crazyflie/log.py` (`LogConfig.MAX_LEN`, `period_in_ms / 10`) |

### Thrust: the hover command

Stock firmware turns the thrust number into newtons in a straight line:
**thrust per motor = number / 65535 · 0.12 N**. So hover needs `weight / 0.48 N · 65535`:

| Drone | Hover number |
|---|---|
| 27 g (our sim's mass) | 36,163 (the drone file default) |
| 29 g (bare 2.1+, the firmware's CF_MASS) | about 38,840 |
| 30.6 g (2.1+ with a Flow deck) | about 41,000 |

Decks, battery and props change it, so measure it: `--calibrate-hover` takes off, hovers 6 s, lands, and reports
the average thrust number while the drone hung still. That is the true hover number (thrust = weight when nothing
moves). Pass it with `--hover-command`, or put it in the drone file (`hover_thrust_command`).

## SITL: the fake Crazyflie

`mindflight/deploy/sitl.py` runs our simulator at 500 Hz with our onboard controller, and copies the firmware's
rules around it:

| Real thing | In SITL |
|---|---|
| radio delay | 5–20 ms each way, packets stay in order |
| thrust lock | thrust ignored until one setpoint with thrust 0 |
| watchdog | no setpoint for 0.5 s: level out, keep thrust. 2 s: motors off, locked |
| tumble | upside down 0.1 s, or past 60° for 1 s: crashed, motors off |
| estimate | 100 Hz, three log blocks, Gaussian noise on position, velocity, attitude, gyro |
| battery | drains with thrust, sags under load; a tired battery limits the thrust |
| a weaker or stronger drone | `thrust_scale` (to test `--calibrate-hover`) |

It is fast: time is simulated, so a 12 s flight takes about 5 s. `--realtime` runs it at real speed.

## Results (SITL, tests/test_deploy_runner.py)

| Brain | Mission | Result |
|---|---|---|
| `hover_attitude.pt` | take off 0.5 m, hover, 3 waypoints, land | all 3 reached within 15 cm (closest 1–3 cm), landed, motors off |
| `waypoint_attitude.pt` (as ONNX) | same | all 3 reached within 15 cm (closest 3–6 cm), landed, motors off |
| `hover_attitude.pt` | `--calibrate-hover` on a drone 15 % weaker / stronger | found the true hover number within 0.1 % |

| Safety test | Result |
|---|---|
| a gust pushes it out of the fence | lands back inside the fence |
| kill key | motors-off arrives within one cycle + radio delay; no thrust after it |
| radio cut | runner stops in < 0.5 s; the drone levels, then cuts its motors at 2 s and locks |
| battery sags below 3.2 V | lands |
| brain outputs NaN | motors off |
| tilt past the limit | motors off |
| started in the air / flat battery | refuses to take off |

## Honest limits

- The thrust line assumes **current firmware** (battery compensation with the 2.1+ propeller numbers; the line
  changed in late 2025). Update the firmware first. `--calibrate-hover` covers the rest.
- With the firmware's line, 60000 is only about **1.7 × the weight** of our 27 g model. Our sim lets the brain ask
  for 2.25 ×. The extra is cut off, so very hard climbs will be slower on the real drone.
- The real firmware turns a yaw rate into a slowly moving yaw angle target; our controller follows the rate
  directly. Both keep yaw still when the brain asks for 0.
- The hover thrust command jumps around (about ±15 %) with the SITL's estimate noise. The real drone will feel
  that as a buzzing throttle; P3's noisy training should calm it.
- Nothing here has flown real hardware yet. `crazyflie.py` is thin on purpose, but it is untested.
