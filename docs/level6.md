# Level 6: Race

**Goal:** the best policies race each other, live, with a leaderboard.

```
python -m mindflight.race          # then open http://127.0.0.1:8000/
```

Each policy flies 2 drones on the same random 6-gate course (hardest setting). One try per round: finish or crash.
The fastest lap wins the round, and a new course appears. The table in the corner keeps score.

**Result:** RESULT_LINE

---

## The racers

RACERS_TABLE

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

EVAL_TABLE

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
