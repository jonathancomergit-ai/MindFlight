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
