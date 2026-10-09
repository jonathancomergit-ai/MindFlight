"""Phase 2 · P4: fly a trained brain on a real Crazyflie, and on a fake one first (docs/phase2-deploy.md).

| File | Job |
|---|---|
| `export.py` | brain (.pt) -> ONNX file, checked against PyTorch |
| `policy.py` | load a brain from .pt or .onnx |
| `interface.py` | the ONE place our units and signs become the firmware's (and log variables become ours) |
| `link.py` | the API every link has; the shared part (estimate, unlock, motors off) |
| `sitl.py` | a fake Crazyflie: our simulator dressed up as the firmware (delay, watchdog, unlock, noise, battery) |
| `crazyflie.py` | the real Crazyflie on cflib (thin) |
| `runner.py` | the 50 Hz flight loop: mission, safety, flight log |
| `__main__.py` | `python -m mindflight.deploy --policy ... --sitl` (or `--uri radio://...`) |
"""
