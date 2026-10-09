# MindFlight 🛸🧠

Thousands of simulated Crazyflie drones learning to fly at once, on the GPU, watched live in your browser.

A learning project and proof of concept for physics simulation + reinforcement learning.

- **Physics:** our own batched quadcopter model in PyTorch, using published Crazyflie 2.x numbers
- **Learning:** PPO, written as one readable file
- **Viewer:** Three.js in the browser, streamed live from Python

See [PLAN.md](PLAN.md) for how it fits together and the levels, and [AGENTS.md](AGENTS.md) for contributor rules.

## Where it stands

All seven levels are done. Each score is from 1,000 fixed test flights with the policy's mean action, on the hardest setting:

| Level | Done when | Result |
|---|---|---|
| 0 Physics | tests prove free fall, hover, torque signs, unit quaternions; 4,096 drones on the GPU | 567 M drone-steps/s at 16,384 drones (CUDA graph + compile) |
| 1 Hover | holds 1 m for 10 s from random starts | 100.0 %, 0 crashes |
| 2 Fly to a point | ≥ 95 % of targets reached within 10 cm, settled | 99.3 % |
| 3 Gates | ≥ 90 % finish 6 gates in order | 95.8 %, 3.7 s laps |
| 4 Obstacles | ≥ 90 % cross the densest pillar field by lidar | 100.0 % (99.6 % on 2,000 more) |
| 5 Sturdy | survives gusts and ±20 % mass; beats the non-randomized policy | 97.4 % vs 84.6 % (77.8 % vs 56.0 % in 2× wind) |
| 6 Race | ≥ 3 policies race live with a leaderboard | racer won 96 of 100 rounds (3.36 s laps vs 3.68 s) |

## Run the tests

```
python -m venv .venv --system-site-packages
.venv\Scripts\activate
pip install -e .[dev]
python -m pytest -q
```

## Watch the drones

```
python -m mindflight.demo --n 64
```

Then open http://127.0.0.1:8000/. The real physics runs live: four groups of drones hover, fall, tip over and spin.
Press **R** to start over. [docs/level0.md](docs/level0.md) explains what you see. Drag to orbit, scroll to zoom, and
tick **size ×10** to see real-scale drones from far away. The server only listens on this machine.

`python -m mindflight.viewer --fake --n 64` streams a grid of fake bobbing drones instead (no physics).

## Train and watch a brain learn

| Level | Watch the trained policy | Train it yourself | Explained |
|---|---|---|---|
| 1 Hover | `python -m mindflight.play --policy policies/hover.pt` | `python -m mindflight.train --task hover --view` | [docs/level1.md](docs/level1.md) |
| 2 Fly to a point | `python -m mindflight.play --policy policies/waypoint.pt` | `python -m mindflight.train --task waypoint --init policies/hover.pt --reset-std 0.15 --view` | [docs/level2.md](docs/level2.md) |
| 3 Gates | `python -m mindflight.play --policy policies/gates.pt` | `python -m mindflight.train --task gates --iters 3000 --view` | [docs/level3.md](docs/level3.md) |
| 4 Obstacles | `python -m mindflight.play --policy policies/obstacles.pt` | `python -m mindflight.train --task obstacles --iters 3000 --view` | [docs/level4.md](docs/level4.md) |
| 5 Sturdy | `python -m mindflight.play --policy policies/sturdy.pt` | `python -m mindflight.train --task sturdy --init policies/waypoint.pt --reset-std 0.15 --view` | [docs/level5.md](docs/level5.md) |
| 6 Race | `python -m mindflight.race` | `python -m mindflight.train --task race --init policies/gates.pt` | [docs/level6.md](docs/level6.md) |

- `--view` opens the live viewer while training: the newest policy flies, and learning curves grow in the corner.
- Each training run writes `runs/<task>/<date-time>/` with `metrics.jsonl`, `latest.pt` and `best.pt`.
- `python -m mindflight.evaluate policies/hover.pt` scores a policy on 1,000 fixed test starts.
- `python scripts/plot_run.py runs/hover/<run> curves.png` draws a run's learning curves.
