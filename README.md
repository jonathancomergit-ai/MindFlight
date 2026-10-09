# MindFlight 🛸🧠

Thousands of simulated Crazyflie drones learning to fly at once, on the GPU, watched live in your browser.

A learning project and proof of concept for physics simulation + reinforcement learning.

- **Physics:** our own batched quadcopter model in PyTorch, using published Crazyflie 2.x numbers
- **Learning:** PPO, written as one readable file
- **Viewer:** Three.js in the browser, streamed live from Python

See [PLAN.md](PLAN.md) for how it fits together and the levels, and [AGENTS.md](AGENTS.md) for contributor rules.

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

- `--view` opens the live viewer while training: the newest policy flies, and learning curves grow in the corner.
- Each training run writes `runs/<task>/<date-time>/` with `metrics.jsonl`, `latest.pt` and `best.pt`.
- `python -m mindflight.evaluate policies/hover.pt` scores a policy on 1,000 fixed test starts.
- `python scripts/plot_run.py runs/hover/<run> curves.png` draws a run's learning curves.
