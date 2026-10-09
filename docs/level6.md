# Level 6: Race

**Goal:** the best policies race each other, live, with a leaderboard.

```
python -m mindflight.race          # then open http://127.0.0.1:8000/
```

Each policy flies 2 drones on the same random 6-gate course (hardest setting). One try per round: finish or crash.
The fastest lap wins the round, and a new course appears. The table in the corner keeps score.

**Result:** three policies race live with a leaderboard. Over a 100-round tournament
(`python scripts/race_tournament.py 100`) the **racer won 96 rounds**, gates 4, daredevil 0:

| Policy | Round wins | Finishes (of 200) | Crashes | Best lap | Average lap |
|---|---|---|---|---|---|
| racer | **96** | 192 | 8 | 2.94 s | **3.36 s** |
| gates | 4 | 188 | 12 | 3.16 s | 3.68 s |
| daredevil | 0 | 0 | 200 | – | – |

---

## The racers

| Racer | Brain | What it is |
|---|---|---|
| **racer** | `policies/race.pt` | the gates brain, then 1,450 more iterations on the race reward (below) |
| **gates** | `policies/gates.pt` | the Level 3 brain: careful |
| **daredevil** | `policies/daredevil.pt` | an early gates brain that only ever saw easy courses (difficulty 0.45): quick when it survives, which is rarely |

Want different racers? `python -m mindflight.race --entrant fast=policies/race.pt --entrant mine=runs/race/<run>/snapshots/it00300.pt`.
Every training run keeps a snapshot at each test, so you can race a brain against its younger self.

## Teaching a drone to hurry

The Level 3 gates policy is careful: it was paid for progress and punished hard for crashing.
The **race** task starts from it and changes the reward a little:

| | Gates | Race |
|---|---|---|
| progress | +10 per metre | +15 per metre |
| time | free | −0.05 every 1/50 s |
| gates, crash | same | same |

The time cost is small on purpose. A big one would make crashing early look good ("stop the clock!").
At 1 m/s the progress reward (0.3 a step) is already 6 × the time cost, so only flying wins.

## The lesson: evaluation

Training scores are noisy and flattering: the drone explores (random actions), every episode is different,
and the course difficulty keeps changing. To compare policies fairly we:

- use the **mean** action (no exploration noise),
- use the **same 1,000 starts** every time (a fixed seed),
- use the **hardest** setting,
- count the **lap time only for laps that finished** (an average that includes crashes means nothing).

`python -m mindflight.evaluate <policies...>` does exactly that and prints one row per policy.
The race view is the fun version of the same idea.

The three racers on the same 1,000 race courses:

| Policy | Finished | Crashed | Average lap (finished laps only) |
|---|---|---|---|
| racer | 95.1 % | 4.6 % | **3.37 s** |
| gates | 95.8 % | 4.1 % | 3.72 s |
| daredevil | 0.2 % | 59.7 % | 2.97 s (2 laps) |

The racer is about 10 % faster for almost the same finish rate. Racing snapshots during training showed the trade:
laps fell steadily (3.68 s → 3.37 s) while the finish rate stayed between 92 and 96 %. The daredevil's rare laps
are the fastest of all, and that is the point of the "finished laps only" rule: an average that ignored crashes
would crown it.

## One more experiment: downwash

`python -m mindflight.race --downwash` turns on the last physics upgrade from the plan: a drone flying above
another pushes it down with its prop wash (`physics/aero.py -> downwash_world`, the empirical model from
gym-pybullet-drones, capped at one drone-weight because the raw formula reaches 28 × the weight at 10 cm).

Every drone was trained alone in its own world, so none of them has ever felt another drone's air. 30 rounds:

| Policy | Without downwash (100 rounds) | With downwash (30 rounds) |
|---|---|---|
| racer | 96 % of drones finish | 27 % finish (16 of 60) |
| gates | 94 % finish | 32 % finish (19 of 60) |

Six drones squeezing through the same 0.5 m gate fly straight through each other's wash, and the brains have
no idea what is pushing them. This is Level 5's lesson from the other side: **a policy only copes with what it
was trained on.** The fix would be to train the racers in shared worlds with downwash on (groups of drones per
course), which is the natural next step after this project.

## Putting it together

| Level | What it added | Policy file |
|---|---|---|
| 0 | physics you can trust (tests) and see (viewer) | – |
| 1 | PPO: reward, policy, value | `hover.pt` |
| 2 | observations and normalization | `waypoint.pt` |
| 3 | curriculum learning | `gates.pt` |
| 4 | sensors and raycasting | `obstacles.pt` |
| 5 | domain randomization and air | `sturdy.pt` |
| 6 | evaluation, racing | `race.pt` |
