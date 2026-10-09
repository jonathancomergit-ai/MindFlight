# Phase 2 · P3: sensors, delays and real-world drones

**Why:** in the simulator we know exactly where each drone is. A real Crazyflie only has an *estimate*. It's noisy,
it drifts, and it reaches the PC a little late over the radio. Real drones also differ from each other and from
the published numbers. A brain trained on perfect data can fall apart on real data. So we train on realistic data.

**Result:** RESULT_TABLE

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

FINDINGS

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
