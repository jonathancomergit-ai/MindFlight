# Phase 2 · P6: follow me, circle me, fly where I point

**Goal:** one brain, three jobs, around a person who walks about:

| Mode | The drone should |
|---|---|
| **follow** | stay 2 m behind the person and 1.5 m up, as they walk, stop, turn and jog |
| **orbit** | circle the person at 2 m, 1.5 m up, at a set speed (0.3–0.7 rad/s, either way round) |
| **point** | when the person points, fly to that spot (2–4 m away, 1.5 m up) and hold it |

The brain is told the mode as three numbers (a "one-hot": `1 0 0` = follow). The mode can change mid-flight.

**Result (camera on the drone, 1,000 test flights per mode):**

| Mode | Score | Bar | Passed? |
|---|---|---|---|
| follow | **76 %** of the time within 0.3 m | ≥ 95 % | ❌ no (see "Why follow misses its bar") |
| orbit | **97 %** of the time within 0.3 m of the 2 m radius | ≥ 90 % | ✅ |
| point | **99 %** of pointed spots reached (15 cm) | ≥ 90 % | ✅ |
| crashes | **0–0.4 %** per mode; **1.0 %** with modes switching | ≤ 1 % | ✅ (just) |

The shipped brain, `policies/follow.pt`, is **copied from a hand-written controller** (behaviour cloning + DAgger),
not learned by PPO. Every PPO run we tried was worse (below). That is the main lesson of this task.

```
python -m mindflight.play --policy policies/follow.pt            # watch: blue person, yellow line = pointing
python scripts/eval_follow.py policies/follow.pt --sensor-limit  # the tables below, 1,000 flights per mode
python scripts/teach_follow.py --score                           # the hand-written teacher
python scripts/teach_follow.py --out policies/follow.pt --seed 1 # make the shipped brain (about 5 minutes)
```

---

## Results

All on the full task (difficulty 1), 1,000 flights of 20 s per row, fixed starts, the mean action.
"all" = a random mode that switches every 4–10 s.

**Two clone seeds, two test seeds (camera):**

| Brain | Test seed | Modes | follow | orbit | point | crashed |
|---|---|---|---|---|---|---|
| clone seed 0 | 12345 | one at a time | 76.3 % | 97.2 % | 98.8 % | 0.0 / 0.0 / 0.5 % |
| clone seed 0 | 12345 | all | 65.8 % | 96.5 % | 98.3 % | 1.4 % |
| clone seed 0 | 777 | one at a time | 76.5 % | 97.2 % | 98.7 % | 0.0 / 0.0 / 0.5 % |
| clone seed 0 | 777 | all | 67.1 % | 96.2 % | 99.3 % | 0.8 % |
| **clone seed 1 (shipped)** | 12345 | one at a time | **76.1 %** | **97.3 %** | **98.7 %** | 0.0 / 0.0 / 0.4 % |
| **clone seed 1 (shipped)** | 12345 | all | 67.0 % | 96.5 % | 99.1 % | **1.0 %** |
| clone seed 1 (shipped) | 777 | one at a time | 76.4 % | 97.3 % | 98.5 % | 0.0 / 0.0 / 0.3 % |
| clone seed 1 (shipped) | 777 | all | 66.9 % | 96.1 % | 99.1 % | 1.0 % |

With modes switching, follow scores lower (67 %): after a switch the drone may still be on the far side of the
person, and the 2 s settle is not always enough to get round them.

**The shipped brain with each person sensor** (trained only on the camera):

| Person sensor | follow | orbit | point | crashed (one at a time / all) |
|---|---|---|---|---|
| perfect | 88.0 % | 99.6 % | 98.2 % | 0–0.4 % / 0.6 % |
| **camera** | **76.1 %** | **97.3 %** | **98.7 %** | 0–0.4 % / 1.0 % |
| gps_rtk | 74.9 % | 98.0 % | 98.7 % | 0–0.4 % / 1.0 % |
| gps (phone) | 0.7 % | 11.3 % | 28.9 % | **84–87 %** / 81 % |
| camera, drone on Lighthouse sensors + real-world drone randomization (P3) | 73.2 % | 97.5 % | 83.7 % | 0–0.7 % / 0.9 % |

**Sensor limits.** How good could *any* brain be with each sensor? A "perfect follower" is teleported every step to
exactly where the sensor says the target is (no flying at all), so only the sensor's error is left:

| Person sensor | trusts the measurement: follow | orbit | moves it forward by its age first: follow | orbit |
|---|---|---|---|---|
| perfect | 100 % | 100 % | 100 % | 100 % |
| camera | 98.6 % | 99.9 % | 100.0 % | 100.0 % |
| gps_rtk | 86.1 % | 99.4 % | 99.5 % | 100.0 % |
| gps (phone) | **0.7 %** | **11.1 %** | **0.7 %** | **11.2 %** |

- **Camera and RTK GPS** are good enough for the 0.3 m bar. What limits follow is the *flying*, not the sensor
  (next section).
- **Phone GPS can't do it.** Its error is 2–3 m and wanders slowly, so even a perfect follower is within 0.3 m only
  0.7 % of the time. Worse, it is **dangerous**: the brain thinks the person is somewhere they aren't, and flies
  into them (84 % of flights crash; the hand-written teacher crashes just as often, 82–88 %). Rough rule: to stay 1 m clear 99 % of the time with a 2.5 m error,
  the drone would have to keep about 1 + 3 × 2.5 ≈ **8–9 m** away. Phone GPS is for "follow me from far away",
  not for 2 m.

### Why follow misses its bar

| Who flies | Person sensor | follow |
|---|---|---|
| the hand-written teacher | perfect | 88.7 % |
| the teacher | camera | 76.1 % |
| the shipped brain (a copy of the teacher) | camera | 76.1 % |
| a perfect follower (teleported) | camera | 98.6 % |

The camera's error alone would allow 98.6 %. Even with *perfect* sensing, our best flier reaches 89 %. The follow
point is 2 m behind the person **relative to which way they face**, so when they turn, it swings sideways on a 2 m
arm; when they jog off or stop, it accelerates at up to 2.5 m/s². A drone limited to 30° of tilt (about 5.7 m/s²),
reacting to a measurement that is 60–100 ms late and comes 30 times a second, lags behind those swings. Removing
the camera's delay alone took a simple controller from 76.5 % to 84.3 % (we tested each camera property one at a
time: delay mattered most, then the 30 Hz rate; noise and the view cone hardly mattered).

To reach 95 %: a smarter flier (one that predicts the person's turns), a gentler follow rule (follow the walking
direction smoothed over a second, as real follow-me drones do), or a looser bar (0.5 m). We kept the bar and report
the miss.

## The person

Every drone has its own world with one simulated person in it. The person is an upright cylinder: 0.25 m radius,
1.75 m tall. In the viewer it is a blue cylinder with a head.

| Thing | Rule |
|---|---|
| arena | stays inside 10 × 10 m (\|x\|, \|y\| ≤ 5 m) |
| gaits | every 2–5 s a new one: stop (15 %), walk 0.5–1.5 m/s (70 %), jog 2.5–3 m/s (15 %) |
| speeding up / slowing down | at most 1.5 m/s² / 2.5 m/s² |
| turning | a smooth random turn (time constant 2 s, typical 0.3 rad/s), at most 1.2 rad/s, less when fast (speed × turn ≤ 3 m/s²), and the turn rate changes by at most 2 rad/s² |
| near the edge (2 m) | no jogging, and turns back toward the middle |
| point mode | stands still to point |

## What the brain sees: the person sensor

The brain never sees the true person. It sees the latest **measurement** from a person sensor, which is late,
noisy and comes only now and then. Pick one with `--env person_sensor=...` (code: `envs/person_sensor.py`):

| Sensor | What it is | Rate | Delay | Position noise | Other |
|---|---|---|---|---|---|
| **camera** (default) | camera on the drone's nose + pose detector | 30 Hz | 60–100 ms | 3 cm + 2 % of the range | sees only a 60–80° cone ahead of the nose, up to 10 m |
| gps_rtk | RTK GPS on the person | 10 Hz | 100 ms | 2–3 cm | – |
| gps | the person's phone | 1–5 Hz | 100–200 ms | **2–3 m, wandering slowly** (+ 0.3 m jitter) | – |
| perfect | the truth | 50 Hz | 0 | 0 | for tests |

Each measurement also has a velocity (noise 0.2 / 0.05 / 0.3 m/s) and a heading (noise 3° / 3° / 11°).
"60–100 ms" means: drawn once per flight. The camera's detections go through a simple **tracker** (each new one is
blended in at 30 %, after moving the old estimate forward by its velocity), as any real system would do.

**Camera view.** Out of view, the drone keeps the last measurement, and a "how old is it" input keeps growing. So
the brain learns to **turn its nose toward the person** (yaw) to keep them in sight. Keeping them in view also pays
+0.3 a step. Flights start with the nose within ±30° of the person.

The target the brain is shown is worked out from the measurement. The reward and the scores use the **true**
target. So the brain has to cope with a target that is late and shaky: for example, lead a jogging person a little.

Observation (35 numbers): the usual 19 (velocity, rotation, spin, last action), then 16 task numbers:

| Columns | What |
|---|---|
| 19–21 | target − drone position (m): the same 3 numbers as waypoint |
| 22–24 | target velocity (m/s): person velocity + the swing from turning (follow) or circling (orbit); 0 for a spot |
| 25–27 | person (hips, 0.9 m up) − drone position (m) |
| 28–30 | person velocity − drone velocity (m/s) |
| 31–33 | mode one-hot: follow, orbit, point |
| 34 | age of the measurement (s, up to 5) |

Actions are **attitude** setpoints (thrust, roll, pitch, yaw rate), the same as a real Crazyflie takes (P2).

## Safety

The drone must keep **at least 1 m** from the person's body (the cylinder's surface).

| Rule | What happens |
|---|---|
| closer than 0.7 m | crash: −30, flight over |
| 0.7 – 1.0 m | soft penalty, 0 at 1 m up to 1 point a step at 0.7 m |
| within 1.5 m, flying *toward* the person | 0.3 points a step per m/s of closing speed (full inside 1 m) |
| below 0.5 m height, within 2 m sideways of the person | crash |

## Reward

Like waypoint (fly to a point), with d = distance to the true target:

| Part | Points per step |
|---|---|
| alive | +3 |
| distance | −0.5 per metre, up to 4 m |
| near the target | +0.5 · (1 − d / 30 cm) inside 30 cm, and +0.5 inside 10 cm |
| tilt, spin, twitch | small, as in hover |
| safety | the three rules above |
| camera view | +0.3 while the person is in the camera's view |
| floor | the step reward never goes below 0 (crashing must never pay) |
| crash | −30 once |

## How it was trained (and what didn't work)

We measured *why* each attempt failed before changing anything (docs/level1-6 lessons).

| Try | What happened | Why (measured) | Fix |
|---|---|---|---|
| 1. PPO from the waypoint brain | 55–65 % of flights hit the person, for 300+ iterations | it flew straight at pointed spots at 3.4 m/s, through the person, 0.9 s after the spot came; the 0.7–1.0 m soft zone is crossed in < 0.1 s, too short to learn from | a "closing speed" cost |
| 2. + closing cost (first within 2 m) | still crashing, and following got worse | the follow and orbit points are 1.75 m from the body, so the cost charged the drone for doing its job whenever the person slowed | closing cost only within 1.5 m |
| 3. + curriculum | crashes fell slowly; follow fell to 1–10 % | "be safe" was easier to learn than "chase": it hung back 1.5 m behind the follow point | – |
| 4. PPO from scratch | lazy too: follow 1 %, sat 1.5 m off | same | – |
| 5. orbit crashes | a jogger ran into a drone placed in front of them | starting still, the drone had 0.35 s to reach 3 m/s backwards: impossible | flights start moving with the target |
| 6. pointed spots at the arena edge | flew through the person | spots that left the arena were flipped *through* the person | pick spots on the drone's side, pulled inside |
| 7. **a hand-written teacher** | follow 76 %, orbit 97 %, point 99 %, crashes ≤ 1 % | a PD controller + "never close in inside 1.6 m" + "slide round the person if the path is blocked" | copy it into a brain |
| 8. PPO on top of the copy (4 runs, normal and gentle) | always worse: test crashes rose from 1–2 % to 5–25 % (gentle: to 6–8 % by iteration 450), scores never improved | not found. Not the observation scaling (the copy with PPO's scaling still flew fine). Our guess: the copy sits close to its safety margins, and PPO's noisy updates push it over | ship the copy |

**Behaviour cloning** = record the teacher flying (with random wobbles), train the network to give the same action.
**DAgger** = then let the *student* fly, ask the teacher what it would have done in each moment, add those examples,
and train again (2 rounds). The value network is trained too, so `--init policies/follow.pt` can carry on with PPO.

The curriculum (`update_curriculum`, difficulty 0 → 1) is still in the env, for PPO runs: difficulty adds jogging,
faster walking, faster orbits and spots on the far side of the person. It climbs when training crashes fall below
10 %, and is at 1 by iteration 800 regardless. All tests and scores use difficulty 1.

## How each mode is scored

| Mode | Score | Bar ("done when") |
|---|---|---|
| follow | % of follow time within 0.3 m of the follow point, after a 2 s settle | **≥ 95 %** |
| orbit | % of orbit time with radius error \|distance to person − 2 m\| < 0.3 m, after a 2 s settle | **≥ 90 %** |
| point | % of pointed spots reached: within 15 cm, held 0.5 s, before the next spot (4–6 s) | **≥ 90 %** |
| all modes | crashes (any reason) and person hits, per flight | **≤ 1 %** |

Each mode is tested alone (1,000 flights of 20 s, fixed starts, mean action), plus "all": a random mode that
switches every 4–10 s. A spot that a mode switch cuts short is not counted.

## The real-world front end (not built yet)

Everything above is simulated. On the real drone, the person's position will come from a **camera on the drone**
(with a pose detector) or from **GPS** on the person. This is how each would work, and what will be hard.

### Camera on the drone

The camera looks forward along the drone's nose. A pose detector finds the person in each frame. Its distance comes
from the person's size in the image (or a depth camera); its direction from where they are in the image. Turned
into the world frame with the drone's own position and heading, that is the measurement the sim models.

| Good | Hard |
|---|---|
| sees the arm, so **pointing works** | only sees ahead: the drone must keep turning toward the person (the sim's view cone) |
| no gear on the person | error grows with distance (the sim: 2 % of range) |
| works indoors | light, blur, other people, needs compute on board or a video link to a PC |

### GPS on the person

| | RTK GPS | Phone GPS |
|---|---|---|
| how | a base station + RTK receivers on the person and the drone | the phone's own GPS, sent over Wi-Fi / radio |
| accuracy | 2–3 cm | **2–5 m**, and the error drifts slowly (it can sit 2 m off for many seconds) |
| rate | 5–20 Hz | 1–5 Hz |
| indoors | no | no |
| pointing | no (no arm) | no |

GPS only works outdoors and never sees an arm, so **point mode needs the camera** (or a phone app: "fly where my
phone points", using its compass).

### Pose estimation

A pose estimator finds body **keypoints** (nose, shoulders, elbows, wrists, hips, knees, ankles) in each image.

| Model | Notes |
|---|---|
| MediaPipe Pose (BlazePose) | 33 keypoints, runs on a CPU at 30+ fps, gives rough 3-D ("world") keypoints from one camera |
| YOLO-pose (e.g. YOLOv8/YOLO11-pose) | 17 keypoints, many people at once, needs a GPU for speed, 2-D only |
| + depth | a depth camera (RealSense, OAK-D) or a second camera turns 2-D keypoints into metres |

From the keypoints:
- **person position** = the middle of the two hips (in metres, after depth / camera calibration),
- **person velocity** = from a tracker (a Kalman filter on the position), which also smooths the jitter,
- **person heading** = the line between the shoulders, turned 90° (or the walking direction when moving).

### Turning an arm into a pointed spot

```
 shoulder S ●───────────● wrist W            direction u = (W − S) / |W − S|
                         ╲
                          ╲  extend the ray from the shoulder:  P(t) = S + t · u
                           ╲
                            ● spot = where the ray reaches 1.5 m height, or a set distance (2–4 m)
```

1. Take the 3-D shoulder S and wrist W of the pointing arm (the arm that is raised and straight: elbow angle near
   180°, wrist well away from the hip).
2. Direction u = (W − S) / |W − S|. Many systems use eye → fingertip instead; it matches where people *think* they
   point a bit better.
3. A pointing ray along the floor never "ends", so pick the spot: where the ray meets the 1.5 m flight height, or a
   fixed distance along it (this sim uses 2–4 m), clamped to the safe flying area.
4. Only accept a point when the arm is held still for ~0.5 s, so waving is not a command.

### Camera limits (be honest)

| Limit | Size | What it does |
|---|---|---|
| latency | camera 30 fps (33 ms) + detector 10–50 ms + radio ~10 ms ≈ **50–100 ms** | the sim's 60–100 ms delay at 30 Hz |
| field of view | a typical small camera sees 60–90° across | the sim's view cone; out of view = old measurement |
| keypoint jitter | a few pixels: roughly **5–10 cm** at 3 m, worse further away | the sim's 3 cm + 2 % of range |
| pointing accuracy | people point roughly: say **5–15°** off (a rough figure); at 3 m that is **25–80 cm** | not simulated: the sim flies to the spot the detector gave, exactly |
| range | small models lose keypoints past ~5–8 m, or when the person is small in the image | the person must stay near |
| lighting | dark rooms, backlight, motion blur ruin keypoints | indoors with good light first |
| occlusion | the drone itself, furniture, other people | lost track = drone must hold / land |
| AI deck camera | about 320 × 320 pixels, greyscale; GAP8 chip with 512 KB fast RAM + 8 MB slower RAM | full pose models don't fit; tiny detectors (a person box, not 17 keypoints) at a few fps are the realistic limit; pointing from it is not realistic |

### What is simulated here, and what is not

| Simulated | Not simulated |
|---|---|
| a person walking, stopping, jogging, turning | real human motion (sudden sidesteps, sitting, waving) |
| a late, noisy, now-and-then position / velocity / heading; a camera view cone; a wandering phone-GPS error | lost tracks, wrong person, false detections, walls blocking the view, GPS dropouts |
| a pointed spot appears every 4–6 s | the arm, the gesture, pointing errors |
| a 1 m safety rule round a cylinder | arms, the real body shape, other people |

So the brain trained here is the **flying half**. The vision half has to be built and measured on real images before
the two meet: record real camera / GPS tracks of a walking person, measure their noise, delay and rate, and check
them against the numbers in `person_sensor.py`.

## Starting from another brain: `scripts/widen_policy.py`

The follow observation keeps waypoint's 22 numbers first and adds 13 more. `scripts/widen_policy.py` gives an old
brain the extra inputs with **zero** weights, so it starts out flying exactly like before, and `--init` works:

```
python scripts/widen_policy.py policies/waypoint_attitude.pt --task follow --out runs/follow/init.pt
python -m mindflight.train --task follow --init runs/follow/init.pt --reset-std 0.15
```

That was try 1 above. It works (a test checks the actions don't change), but here it didn't lead to a good brain.

## Tests

`tests/test_follow.py` (21) and `tests/test_person_sensor.py` (6):

| Test | Checks |
|---|---|
| shapes and observation columns | 35 numbers; first task columns = seen target − position; one-hot; measurement age |
| person stays in arena and moves like a person | 30 s × 512 people: inside 10 × 10 m, ≤ 3 m/s, turns ≤ 1.2 rad/s, speed × turn ≤ 3 m/s², turn rate changes ≤ 2 rad/s², speed changes within limits, many stop, some jog |
| mode targets by hand | follow point behind for 3 headings and its swing when turning; orbit points and speed; spot vs. waiting |
| orbit goes round at the set rate | angle after 1 s = ω · 1 s, radius 2 m |
| pointing waits for the person to stop | no spot while walking; spots 2–4 m away, inside 6 m, 1.5 m up |
| easy curriculum / spots at the arena edge | easy spots stay on the drone's side; at the edge no spot is behind the person |
| curriculum climbs | climbs when crashes are rare; reaches 1 by iteration 800 anyway |
| person distance is to the body surface | beside, above, inside, diagonal |
| too close = crash, near = soft penalty | 0.6 m crashes (−30), 0.85 m costs 0.5, 1.2 m costs 0; also through `step` |
| flying at the person costs more | 2 m/s toward at 1.25 m costs 0.3; flying past costs nothing |
| low near the person = crash | 0.4 m high at 1.5 m: crash; at 3 m: fine |
| reward never negative before a crash | 4 m off and at 0.7 m: ≥ 0 |
| perfect sensor = truth; flights start looking at the person; view bonus | |
| scores | a perfect follower scores 100 % after the 2 s settle; a perfect pointer reaches every spot |
| modes switch only when asked | all switch within 10 s; a single-mode env never does |
| viewer shapes and scene | person cylinder + head, camera guess, pointing line ends at the spot; scene has them all |
| widened policy flies like the old one | `scripts/widen_policy.py` keeps actions and values the same |
| teacher does the job | follow > 50 %, orbit > 85 %, spots > 90 %, no crashes (short CPU run) |
| sensor: delay | a 10 Hz, 5-step-late sensor gives the person exactly 5 steps back, every 5th step, and the age is right |
| sensor: rates | 30 / 10 / 50 Hz measured; phone GPS 1–5 Hz |
| sensor: camera noise grows with range | σ = 7 cm at 2 m, 13 cm at 5 m; velocity σ 0.2 m/s |
| sensor: RTK noise, phone GPS wanders | RTK 2–3 cm; GPS error 2–3 m, barely moves in 1 s, mostly new after 1 minute |
| sensor: field of view | inside / outside the cone, behind, too far; turning the nose brings the person in |
| sensor: out of view | the camera keeps the old measurement and its age grows |
