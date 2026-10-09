# Phase 2 · P6: follow me, circle me, fly where I point

**Goal:** one brain, three jobs, around a person who walks about:

| Mode | The drone should |
|---|---|
| **follow** | stay 2 m behind the person and 1.5 m up, as they walk, stop, turn and jog |
| **orbit** | circle the person at 2 m, 1.5 m up, at a set speed (0.3–0.7 rad/s, either way round) |
| **point** | when the person points, fly to that spot (2–4 m away, 1.5 m up) and hold it |

The brain is told the mode as three numbers (a "one-hot": `1 0 0` = follow). The mode can change mid-flight.

**Result:** RESULTS_SUMMARY

```
python -m mindflight.play --policy policies/follow.pt          # watch: blue person, yellow line = pointing
python scripts/eval_follow.py policies/follow.pt               # the table below, 1,000 flights per mode
python -m mindflight.train --task follow --iters 1500          # train it yourself (attitude actions)
```

---

## Results

RESULTS_TABLE

## The person

Every drone has its own world with one simulated person in it. The person is an upright cylinder: 0.25 m radius,
1.75 m tall. In the viewer it is a blue cylinder with a head.

| Thing | Rule |
|---|---|
| arena | stays inside 10 × 10 m (\|x\|, \|y\| ≤ 5 m) |
| gaits | every 2–5 s a new one: stop (15 %), walk 0.5–1.5 m/s (70 %), jog 2.5–3 m/s (15 %) |
| speeding up / slowing down | at most 1.5 m/s² / 2.5 m/s² |
| turning | a smooth random turn (time constant 1 s), at most 1.2 rad/s, and less when fast (speed × turn ≤ 3 m/s²) |
| near the edge (2 m) | no jogging, and turns back toward the middle |
| point mode | stands still to point |

## What the brain sees: the camera

A real drone would find the person with a camera and a pose detector. Those are **noisy** and **late**, so the brain
never sees the true person. It sees an estimate:

| | Simulated as | Why |
|---|---|---|
| delay | 3–5 steps = 60–100 ms (random per flight) | camera exposure + detector + radio |
| position noise | σ = 5–10 cm (random per flight), fresh each step | keypoints jitter frame to frame |
| velocity noise | σ = 0.2 m/s | a tracker's velocity estimate |
| heading noise | σ = 0.05 rad (3°) | which way the person faces |

The target it is shown is worked out from that estimate. The reward and the scores use the **true** target. So the
brain has to cope with a target that is late and shaky: for example, lead a jogging person a little.

Observation (34 numbers): the usual 19 (velocity, rotation, spin, last action), then 15 task numbers:

| Columns | What |
|---|---|
| 19–21 | target − drone position (m) — the same 3 numbers as waypoint |
| 22–24 | target velocity (m/s): person velocity + the swing from turning (follow) or circling (orbit); 0 for a spot |
| 25–27 | person (hips, 0.9 m up) − drone position (m) |
| 28–30 | person velocity − drone velocity (m/s) |
| 31–33 | mode one-hot: follow, orbit, point |

Actions are **attitude** setpoints (thrust, roll, pitch, yaw rate), the same as a real Crazyflie takes (P2).

## Safety

The drone must keep **at least 1 m** from the person's body (the cylinder's surface).

| Rule | What happens |
|---|---|
| closer than 0.7 m | crash: −30, flight over |
| 0.7 – 1.0 m | soft penalty, 0 at 1 m up to 1 point a step at 0.7 m |
| within 2 m, flying *toward* the person | 0.3 points a step per m/s of closing speed (full inside 1 m) |
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
| floor | the step reward never goes below 0 (crashing must never pay) |
| crash | −30 once |

LESSONS

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

Everything above is simulated. On a real drone, the person's position and the pointing direction would come from a
**camera + pose estimator**. This is how it would work, and what will be hard.

### Where the camera goes

| Option | Good | Bad |
|---|---|---|
| on the drone (e.g. Crazyflie AI deck) | goes wherever the drone goes | tiny camera, little compute, shaky, sees the person from above and close |
| on the ground (webcam / phone on a tripod, PC does the work) | big camera, big GPU, steady | only sees one area; must know where the drone is too (Lighthouse / Flow deck) |

For a first real test, **the ground camera is far easier**: a PC runs the detector, and the Crazyflie's own
position (from a Lighthouse or Flow deck) puts both in the same frame.

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

### Limits (be honest)

| Limit | Size | What it does |
|---|---|---|
| latency | camera 30 fps (33 ms) + detector 10–50 ms + radio ~10 ms ≈ **50–100 ms** | the sim's 60–100 ms delay |
| keypoint jitter | a few pixels: **5–10 cm** at 3 m, worse further away | the sim's position noise |
| pointing accuracy | people point roughly: say **5–15°** off (a rough figure); at 3 m that is **25–80 cm** | not simulated: the sim flies to the spot the detector gave, exactly |
| range | small models lose keypoints past ~5–8 m, or when the person is small in the image | the person must stay near |
| lighting | dark rooms, backlight, motion blur ruin keypoints | indoors with good light first |
| occlusion | the drone itself, furniture, other people | lost track = drone must hold / land |
| AI deck camera | about 320 × 320 pixels, greyscale; GAP8 chip with 512 KB fast RAM + 8 MB slower RAM | full pose models don't fit; tiny detectors (a person box, not 17 keypoints) at a few fps are the realistic limit; pointing from it is not realistic |

### What is simulated here, and what is not

| Simulated | Not simulated |
|---|---|
| a person walking, stopping, jogging, turning | real human motion (sudden sidesteps, sitting, waving) |
| a late, noisy position / velocity / heading | lost tracks, wrong person, false detections |
| a pointed spot appears every 4–6 s | the arm, the gesture, pointing errors |
| a 1 m safety rule round a cylinder | arms, the real body shape, other people |

So the brain trained here is the **flying half**. The vision half has to be built and measured on real images before
the two meet: first on the ground camera, recorded, then compared against the sim's noise and delay numbers.

## Starting from the waypoint brain

STARTING

## Tests (`tests/test_follow.py`)

| Test | Checks |
|---|---|
| shapes and observation columns | 34 numbers; first task columns = seen target − position; one-hot right |
| person stays in arena and moves like a person | 30 s × 512 people: inside 10 × 10 m, ≤ 3 m/s, turns ≤ 1.2 rad/s, speed × turn ≤ 3 m/s², acceleration limits, some stop and some jog |
| mode targets by hand | follow point behind for 3 headings, its swing when turning; orbit points and speed; spot vs. waiting |
| orbit goes round at the set rate | angle after 1 s = ω · 1 s, radius 2 m |
| pointing waits for the person to stop | no spot while walking; spots 2–4 m from the person, inside 6 m, 1.5 m up |
| person distance is to the body surface | beside, above, inside, diagonal |
| too close = crash, near = soft penalty | 0.6 m crashes (−30), 0.85 m costs 0.5, 1.2 m costs 0; also through `step` |
| flying at the person costs more than flying away | 2 m/s toward at 1.5 m costs 0.3 |
| low near the person = crash | 0.4 m high at 1.5 m: crash; at 3 m: fine |
| reward never negative before a crash | 4 m off and at 0.7 m: ≥ 0 |
| camera delay | the estimate equals the true state exactly 3, 4 or 5 steps back |
| camera noise | σ of position, velocity, heading errors match (20,000 samples); ranges per flight |
| camera off = truth | |
| scores | a perfect follower scores 100 % after the 2 s settle; a perfect pointer reaches every spot |
| modes switch only when asked | all switch within 10 s; a single-mode env never does |
| viewer shapes and scene | person cylinder + head, camera guess, pointing line ends at the spot; scene has them all |
| widened policy flies like the old one | `scripts/widen_policy.py` keeps the actions and values the same |
