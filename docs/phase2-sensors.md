# Phase 2 · P3: sensors, delays and real-world drones

**Why:** in the simulator we know exactly where each drone is. A real Crazyflie only has an *estimate*. It's noisy,
it drifts, and it reaches the PC a little late over the radio. Real drones also differ from each other and from
the published numbers. A brain trained on perfect data can fall apart on real data. So we train on realistic data.

**Result:** brains trained on realistic data hold up on realistic drones. Brains trained on perfect data don't.
All tests: 1,000 flights, attitude mode, every drone randomized (`real_world=true`).

| Brain | Lighthouse, within 10 cm | Flow deck, within 15 cm | Crashes |
|---|---|---|---|
| hover, perfect data (`hover_attitude.pt`) | 97.4 % | 81.6 % | 0.3–0.8 % |
| **hover, realistic data (`real_hover.pt`)** | **99.8 %** | **95.3 %** | **0 %** |
| waypoint, perfect data (`waypoint_attitude.pt`) | 44.9 % | 33.2 % | 7–16 % |
| **waypoint, realistic data (`real_waypoint.pt`)** | **95.8 %** | **94.9 %** (just under the 95 % bar) | **0.1 %** |

```
python -m mindflight.train --task hover --env action_mode=attitude sensors=flowdeck real_world=true --init policies/hover_attitude.pt --reset-std 0.15
python -m mindflight.evaluate policies/hover_attitude.pt policies/real_hover.pt --env action_mode=attitude sensors=flowdeck real_world=true
```

## What the brain sees now (`mindflight/envs/sensing.py`)

| Profile | Position | Velocity | Attitude | Gyro | Delay |
|---|---|---|---|---|---|
| `ideal` | exact | exact | exact | exact | none |
| `lighthouse` | ±5 mm | ±2 cm/s | ±0.5° | ±0.02 rad/s + small bias | 0–20 ms |
| `flowdeck` | ±2 cm + slow drift | ±5 cm/s | ±1° | ±0.03 rad/s + bias | 20–40 ms |

- **Noise** is new every step. **Bias** is fixed for a whole flight (gyros have it).
- **Drift**: the Flow deck works out position by adding up motion, so its error slowly grows, like walking with
  your eyes closed (about 3 cm after 10 s).
- **Delay**: each flight gets a delay of 0, 1 or 2 steps (20 ms each). The brain sees the past. Its commands can
  arrive late too.

Rewards and crashes still use the *true* state. The brain is judged on where the drone really is, not where it
thinks it is.

These numbers are typical values from Bitcraze's docs and Crazyflie papers, rounded. They are marked ASSUMED.
Logs from the real drone will replace them.

## Every drone is a little different (`real_world=true`)

| Each flight, at random | Range | Real cause |
|---|---|---|
| mass | ±8 % | battery size, decks, tape |
| prop strength | ±8 % | worn or chipped props, motor wear |
| motor lag | 0.7–1.5 × | motor and battery differences |
| centre of mass | up to 3 mm off-centre | battery not quite in the middle |
| battery sag | up to 6 % weaker over 10 s | the battery drains as it flies |
| rotor drag, ground effect | on | real air |

An off-centre battery twists the drone a little, all the time. The onboard controller's integral learns to
cancel it, just as on a real drone.

## Why 15 cm for the Flow deck?

A drone can only hold the spot it *thinks* it's at. With a Flow deck, its own sense of position wanders a few cm.
So we measured the sensor limit: a perfect controller that sits exactly where its estimate says.

| Sensors | Perfect controller, within 10 cm | within 15 cm |
|---|---|---|
| Lighthouse | 100 % | 100 % |
| Flow deck | **54 %** | 98.7 % |

No brain can beat 54 % at 10 cm with a Flow deck. Ours gets 50.8 %, close to that wall. At 15 cm the limit is
98.7 % and ours gets 95.3 %. If you want 10 cm accuracy on the real drone, buy the Lighthouse kit.

## What went wrong first

The first noise model drew fresh noise every 20 ms. Hover looked fine (80 %), because fresh noise averages out.
But a real Crazyflie's estimate comes out of a filter, so its errors *wander smoothly*, and the drone follows them.
With realistic wandering errors, the 10 cm score fell to 48 %. That isn't a worse brain, it's a more honest test.
The sensor-limit measurement above is how we told the two apart.

## The lesson

- **The waypoint brain trained on perfect data crashed up to 16 % of the time** on realistic drones. Same brain,
  same task. Only the sensors and the drone changed. That's what usually happens on the first real flight, and
  why we train this way.
- **Measure the limit before you blame the brain.** When results stall, check what a perfect controller would get.

## Tests (`tests/test_sensing.py`)

| Test | Checks |
|---|---|
| ideal is the truth | no noise, no delay |
| noise matches the profile | measured spread = profile (within 5 %), quaternions stay unit length |
| flow-deck drift | grows like a random walk: σ√t |
| observation delay | each drone sees exactly its number of steps back; a reset clears the history |
| action delay | late commands arrive in order |
| rewards use the truth | observations noisy, distance exact |
| real-world drones | ranges, drag on, battery only weakens |
| heavy nose pitches down (`test_aero.py`) | centre-of-mass offset physics |
