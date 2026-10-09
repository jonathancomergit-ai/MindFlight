# Level 2: Fly to a point

**Goal:** fly to a random target, stop there, then do it again. A new target comes every 2.5 s, anywhere in a
3 × 3 × 1.5 m box. That is 4 targets per 10 s flight.

**Done when:** at least 95 % of targets are reached: within 10 cm, and still there for the last 0.5 s.

**Result:** RESULT_LINE

```
python -m mindflight.play --policy policies/waypoint.pt       # watch: yellow dots are the targets
python -m mindflight.train --task waypoint --view             # train it yourself
python -m mindflight.evaluate policies/waypoint.pt            # 1,000 test flights = 4,000 targets
```

---

## What changed from hover

| | Hover | Fly to a point |
|---|---|---|
| target | fixed, 1 m up | new random one every 2.5 s |
| distance in the reward | −1 per metre, max −1 | −0.5 per metre, max −1 (targets can be 3 m away) |
| crash if | > 3 m from the target | outside the 6 × 6 × 3.5 m arena |

The observation is the same 22 numbers. The last 3 ("where is the target from here") now jump every 2.5 s.

## The lesson: observation normalization

The policy sees numbers of very different sizes:

| Observation | Typical size |
|---|---|
| target distance | up to 3 m |
| velocity | up to a few m/s |
| rotation matrix | between −1 and 1 |
| spin | up to 10+ rad/s |
| last action | between −1 and 1 |

A neural network multiplies each input by a weight and adds them up.
A big input shouts. A small one whispers. The network has to learn very different weights just to hear them
equally, and one learning rate can't suit them all.

**Normalization** fixes that before the network sees anything:

```
normalized = (x − running mean) / running spread         then clipped to ±5
```

The mean and spread are measured from everything the drones have seen so far, and keep updating.
After that, every input is "about 0, give or take 1".

### The experiment

Two runs, the same in every way except one switch (`--set normalize_obs=false`):

![Normalization on vs off](img/level2_normalization.png)

NORMALIZATION_FINDINGS

## Tests

| Test | Checks |
|---|---|
| `test_waypoint_targets_change_every_2_5_s_and_count_when_settled` | new targets on time; a target only counts when the drone settled on it |
| `test_waypoint_leaving_the_arena_is_a_crash` | the arena walls |
| `test_running_norm_matches_all_data_at_once` (Level 1) | the running mean and spread are exact |
