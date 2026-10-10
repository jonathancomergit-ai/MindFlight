"""Score a follow policy mode by mode: `python scripts/eval_follow.py policies/follow.pt [more.pt ...]`.

For each mode (follow, orbit, point, and "all" = random modes switching mid-episode) it flies 1,000 episodes of 20 s
on fixed starts (the same every time), with the mean action, and prints one table row (markdown) per policy and mode:

| Column | Means | Bar (docs/phase2-follow.md) |
|---|---|---|
| follow | % of follow time within 0.3 m of the follow point (after a 2 s settle) | ≥ 95 % |
| orbit | % of orbit time with radius error < 0.3 m (after a 2 s settle) | ≥ 90 % |
| point | % of pointed spots reached (within 15 cm, held 0.5 s) | ≥ 90 % |
| crashed | % of episodes that crashed (any reason) | ≤ 1 % |
| hit person | % of episodes that ended by getting closer than 0.7 m (or low near the person) | ≤ 1 % |
| < 1 m | average seconds per episode spent inside the 1 m comfort distance | – |
"""

import argparse

import torch

from mindflight.envs import make_env
from mindflight.physics.dynamics import POS, QUAT, VEL
from mindflight.physics.rotations import quat_from_axis_angle
from mindflight.rl.train import evaluate, load_policy

MODES = ("follow", "orbit", "point", "all")


def score_modes(path: str, n: int = 1000, seed: int = 12345, device: str | None = None, **env_kwargs) -> dict:
    """{mode: {"follow": %, "orbit": %, "point": %, "crashed": %, "hit_person": %, "close_s": s}} for one policy."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_policy(path, device)
    results = {}
    for mode in MODES:
        kwargs = dict(checkpoint.get("env_kwargs", {}))
        kwargs.update(env_kwargs)
        kwargs["mode"] = mode
        kwargs["difficulty"] = 1.0  # the full task
        env = make_env("follow", n, device=device, seed=seed, **kwargs)
        r = evaluate(model, env, seed=seed)

        def ratio(good: str, total: str) -> float | None:
            return 100 * r[good] / r[total] if r.get(total, 0) > 0 else None

        results[mode] = {
            "follow": ratio("follow_good_s", "follow_scored_s"),
            "orbit": ratio("orbit_good_s", "orbit_scored_s"),
            "point": ratio("spots_reached", "spots"),
            "crashed": 100 * r["crashed"],
            "hit_person": 100 * r["hit_person"],
            "close_s": r["close_s"],
            "min_person_m": r["min_person_m"],
        }
    return results


@torch.no_grad()
def sensor_limit(
    person_sensor: str, n: int = 1000, seed: int = 12345, device: str | None = None, predict: bool = False
) -> dict:
    """A "perfect follower" for this sensor: always exactly where the sensor says the target is (teleported there
    every step, nose toward where it sees the person). No flying at all, so what is left is only the sensor's error.
    predict=True: it first moves the measurement forward by its age (position + velocity × age, heading + turn rate
    × age), the simplest way to make up for the delay. Returns follow % and orbit % (same scoring as the policies)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for mode in ("follow", "orbit"):
        env = make_env("follow", n, device=device, seed=seed, mode=mode, difficulty=1.0, person_sensor=person_sensor)
        env.reset()
        up = torch.zeros(n, 3, device=env.device)
        up[:, 2] = 1.0
        for _ in range(env.max_steps - 1):
            target = env.seen_target
            if predict:
                age = env.tracker.age_s[:, None]
                target, _ = env.mode_target(env.seen_xy + env.seen_vel * age, env.seen_vel,
                                            env.seen_heading + env.seen_turn_rate * age[:, 0], env.seen_turn_rate)
            env.state[:, POS] = target
            env.state[:, VEL] = 0.0
            rel = env.seen_xy - env.seen_target[:, :2]
            env.state[:, QUAT] = quat_from_axis_angle(up, torch.atan2(rel[:, 1], rel[:, 0]))
            env._post_step()
        good, scored = (env.follow_good, env.follow_scored) if mode == "follow" else (env.orbit_good, env.orbit_scored)
        out[mode] = 100 * good.sum().item() / max(scored.sum().item(), 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("policies", nargs="+")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--person-sensor", default=None, help="camera, gps_rtk, gps or perfect (default: as trained)")
    parser.add_argument("--env", nargs="*", default=[], metavar="KEY=VALUE", help="e.g. sensors=lighthouse real_world=true")
    parser.add_argument("--sensor-limit", action="store_true", help="also print the perfect-follower limit per sensor")
    args = parser.parse_args()
    env_kwargs = {}
    for item in args.env:
        key, value = item.split("=", 1)
        env_kwargs[key] = value.lower() == "true" if value.lower() in ("true", "false") else value
    if args.person_sensor:
        env_kwargs["person_sensor"] = args.person_sensor

    def pct(v):
        return "-" if v is None else f"{v:.1f} %"

    print("| policy | modes | follow | orbit | point | crashed | hit person | < 1 m (s) |")
    print("|---|---|---|---|---|---|---|---|")
    for path in args.policies:
        results = score_modes(path, n=args.n, seed=args.seed, **env_kwargs)
        for mode, s in results.items():
            print(
                f"| {path} | {mode} | {pct(s['follow'])} | {pct(s['orbit'])} | {pct(s['point'])} | "
                f"{s['crashed']:.1f} % | {s['hit_person']:.1f} % | {s['close_s']:.2f} |"
            )

    if args.sensor_limit:
        print()
        print("| person sensor | trusts it: follow | orbit | predicts: follow | orbit |")
        print("|---|---|---|---|---|")
        for sensor in ("perfect", "camera", "gps_rtk", "gps"):
            trust = sensor_limit(sensor, n=args.n, seed=args.seed)
            ahead = sensor_limit(sensor, n=args.n, seed=args.seed, predict=True)
            print(f"| {sensor} | {trust['follow']:.1f} % | {trust['orbit']:.1f} % | {ahead['follow']:.1f} % | "
                  f"{ahead['orbit']:.1f} % |")


if __name__ == "__main__":
    main()
