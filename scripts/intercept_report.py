"""Score an intercept policy: `python scripts/intercept_report.py policies/intercept.pt [--sweep]`.

One 15 s episode (about 5 throws) per drone on fixed test starts (seed 12345), mean action, like
`python -m mindflight.evaluate`. Prints, for the hardest training setting (difficulty 1) and, with --sweep, for
other throw arcs and flight times:

    caught      interceptions / throws, over all throws (a crash with a ball in the air counts as a miss)
    crashed     episodes that ended in a crash
    too close   episodes that ended because the drone's edge came within 0.5 m of the body (the safety rule)
    closest     how close the drone's edge came to the body, on average over episodes (m)
"""

import argparse

import torch

from mindflight.envs import make_env
from mindflight.envs.intercept import DRONE_RADIUS_M, HARD_ARC_DEG, HARD_FLIGHT_S
from mindflight.rl.train import evaluate, load_policy

SWEEP_ARCS_DEG = (10, 30, 45, 60, 90)
SWEEP_FLIGHT_S = (1.4, 1.2, 1.1, 1.0, 0.9, 0.8)


def score(model, env_kwargs: dict, n: int, seed: int, device: str) -> dict[str, float]:
    env = make_env("intercept", n, device=device, seed=seed, **env_kwargs)
    return evaluate(model, env, seed=seed)


def row(label: str, r: dict[str, float]) -> str:
    caught = r["intercepts"] / max(r["throws"], 1e-9)
    edge = r["closest_to_body_m"] - DRONE_RADIUS_M
    return (
        f"{label:<28} {100 * caught:6.1f}% {100 * r['crashed']:7.1f}% {100 * r['crashed_too_close']:9.1f}% "
        f"{edge:8.2f} m {r['throws']:6.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--sweep", action="store_true", help="also try other arcs and flight times")
    parser.add_argument("--env", nargs="*", default=[], metavar="KEY=VALUE", help="extra task options, e.g. max_tilt_deg=45")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_policy(args.policy, device)
    base = dict(checkpoint.get("env_kwargs", {}))
    for item in args.env:
        key, value = item.split("=", 1)
        base[key] = float(value)

    print(f"{'setting':<28} {'caught':>7} {'crashed':>8} {'too close':>10} {'closest':>10} {'throws':>6}")
    hard = f"difficulty 1 (±{HARD_ARC_DEG:.0f}°, {HARD_FLIGHT_S[0]}–{HARD_FLIGHT_S[1]} s)"
    print(row(hard, score(model, {**base, "difficulty": 1.0}, args.n, args.seed, device)))
    if args.sweep:
        for flight_s in SWEEP_FLIGHT_S:
            for arc in SWEEP_ARCS_DEG:
                kwargs = {**base, "difficulty": 1.0, "arc_deg": arc, "flight_s": flight_s}
                r = score(model, kwargs, args.n, args.seed, device)
                print(row(f"±{arc}°, {flight_s - 0.1:.1f}–{flight_s + 0.1:.1f} s", r))


if __name__ == "__main__":
    main()
