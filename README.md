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
