# Rules for agents working on MindFlight

Read `PLAN.md` first. Its **Conventions** section is the contract: units, frames, quaternion order, state layout,
motor layout. Don't change a convention in a feature PR; propose it in its own issue.

## Git
- Never commit to `main`. Work on your own branch, open a PR that closes the issue.
- One issue per PR. Keep PRs small.

## Code
- Python 3.11+, PyTorch, `float32`. Physics functions are **batched**: tensors shaped `(N, ...)`, no Python loops
  over drones, and they run on whatever device the inputs are on.
- This is a learning project: write it so a beginner can read it. A short docstring on each function saying what
  it computes, with the equation and units. Name things in words (`thrust_n`, not `t`).
- Physical constants live only in `mindflight/physics/params.py`. Anything not from the cited source is marked
  `# ASSUMED`.
- The viewer is plain HTML + ES modules from a pinned CDN version. No npm, no build step.

## Tests
- Every physics change comes with a pytest that checks real physics (free fall rate, hover balance, torque signs,
  quaternion stays unit length), not only that the code runs.
- Run before opening a PR: `python -m pytest -q`
- Tests must pass on CPU. GPU-only tests are skipped with `pytest.mark.skipif(not torch.cuda.is_available(), ...)`.

## Setup
```
python -m venv .venv --system-site-packages
.venv\Scripts\activate
pip install -e .[dev]
```
`--system-site-packages` reuses the machine's CUDA build of PyTorch (RTX 5090, cu128) instead of downloading it again.
