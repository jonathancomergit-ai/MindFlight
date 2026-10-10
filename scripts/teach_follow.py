"""A hand-written "teacher" for the follow task, and a first brain copied from it (behaviour cloning).

    python scripts/teach_follow.py --score                       # how well the teacher itself flies
    python scripts/teach_follow.py --out policies/follow.pt      # copy it into a brain (+ 2 DAgger rounds)
    python -m mindflight.train --task follow --init policies/follow.pt --reset-std 0.03 --iters 300         --env start_difficulty=1.0 --set learning_rate=3e-5 target_kl=0.003   # PPO on top (didn't help, see doc)

Why: PPO started from the waypoint brain (or from scratch) learned to be safe by hanging back 1.5 m behind the
target (follow scores of 1–10 %). Chasing a moving point is a skill a plain controller already has, so we wrote one
and copied it into a brain. PPO on top of the copy didn't improve it (every run we tried slowly lost safety), so the
shipped policies/follow.pt is the copy (docs/phase2-follow.md).

The teacher is a PD controller, using only what the brain sees (the person measurement and its age):
    1. target: the follow / orbit / spot point, worked out from the measurement moved forward by its age
       (position + velocity × age, heading + turn rate × age);
    2. wanted velocity = target velocity + 1.6 /s × (target − position), the second part at most 2.5 m/s;
    3. keep clear of the person: inside 1.6 m of the body, never closing in on them (even if they walk at the
       drone) and back off at 3 m/s per metre too
       close; and if the straight path to the target passes within 1.3 m of them, slide round (sideways, up to
       2.5 m/s inside 3 m);
       acceleration a = 5 /s · (wanted velocity − velocity), at most 5 m/s² sideways (near the target that is the
       usual PD: a = 8 · (target − position) + 5 · (target velocity − velocity));
    4. turn the acceleration into the attitude action: pitch = atan(a_forward / g), roll = atan(−a_left / g),
       thrust = (1 + a_up / g) / cos(tilt); and yaw toward the person to keep them in the camera's view.

Behaviour cloning: fly the teacher (with random wobbles added, so it also sees how to recover), record (observation,
teacher action, reward), and train the policy network to give the same action (mean squared error). The value
network is trained too, on what each moment turned out to be worth, so PPO's first updates aren't steered by a
value network that knows nothing (our first try without it lost a third of the teacher's skill in 100 iterations).
"""

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import torch

from mindflight.envs import make_env
from mindflight.envs.follow import PERSON_RADIUS_M, FollowEnv
from mindflight.physics.dynamics import POS, VEL
from mindflight.rl.ppo import ActorCritic, PPOConfig
from mindflight.rl.train import evaluate

G = 9.81
MAX_TILT_RAD = math.radians(30)


class Teacher:
    """The PD + keep-clear controller above, as an object with the policy's `act(obs)` so `evaluate` can fly it."""

    def __init__(self, env: FollowEnv, kp: float = 8.0, kd: float = 5.0, clear_m: float = 1.6, slide_m: float = 3.0):
        self.env, self.kp, self.kd, self.clear_m, self.slide_m = env, kp, kd, clear_m, slide_m

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        e = self.env
        pos, vel = e.state[:, POS], e.state[:, VEL]
        age = e.tracker.age_s[:, None]
        target, target_vel = e.mode_target(
            e.seen_xy + e.seen_vel * age, e.seen_vel, e.seen_heading + e.seen_turn_rate * age[:, 0], e.seen_turn_rate
        )
        # Wanted velocity: move with the target, plus close the gap (at most 2.5 m/s extra).
        closing = self.kp / self.kd * (target - pos)
        closing = closing * (2.5 / closing.norm(dim=1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
        wanted = target_vel + closing

        # Keep clear of the person.
        away = pos[:, :2] - e.seen_xy
        dist = away.norm(dim=1, keepdim=True).clamp_min(1e-6)
        away = away / dist
        gap = dist - PERSON_RADIUS_M
        toward = ((wanted[:, :2] - e.seen_vel) * -away).sum(dim=1, keepdim=True).clamp_min(0.0)  # closing speed
        too_near = (self.clear_m - gap).clamp_min(0.0)
        wanted[:, :2] += (too_near > 0) * toward * away + 3.0 * too_near * away  # never closing in inside 1.6 m
        # Path blocked? The straight line to the target passes within 1.3 m of the body: slide round the person.
        to_target = target[:, :2] - pos[:, :2]
        length = to_target.norm(dim=1, keepdim=True).clamp_min(1e-6)
        along = ((e.seen_xy - pos[:, :2]) * to_target).sum(dim=1, keepdim=True) / length  # how far ahead the person is
        closest = pos[:, :2] + to_target / length * along.clamp(min=0.0) - e.seen_xy
        blocked = ((along > 0) & (along < length) & (closest.norm(dim=1, keepdim=True) - PERSON_RADIUS_M < 1.3)).float()
        side = torch.stack((-away[:, 1], away[:, 0]), dim=1)
        side = side * torch.sign((side * to_target).sum(dim=1, keepdim=True) + 1e-6)  # slide toward the target's side
        wanted[:, :2] += blocked * 2.5 * (self.slide_m - gap).clamp(0.0, 1.0) * side
        accel = self.kd * (wanted - vel)

        horizontal = accel[:, :2]
        horizontal = horizontal * (5.0 / horizontal.norm(dim=1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
        yaw = e.drone_yaw()
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        forward = horizontal[:, 0] * cos_y + horizontal[:, 1] * sin_y
        left = -horizontal[:, 0] * sin_y + horizontal[:, 1] * cos_y
        pitch, roll = torch.atan(forward / G), torch.atan(-left / G)
        tilt = torch.sqrt(pitch**2 + roll**2).clamp(max=MAX_TILT_RAD)
        thrust = (1.0 + accel[:, 2].clamp(-5.0, 5.0) / G) / torch.cos(tilt)
        look = e.seen_xy - pos[:, :2]
        yaw_error = torch.remainder(torch.atan2(look[:, 1], look[:, 0]) - yaw + math.pi, 2 * math.pi) - math.pi
        action = torch.stack(
            ((thrust - 1.0) / 1.25, roll / MAX_TILT_RAD, pitch / MAX_TILT_RAD, 3.0 * yaw_error / math.pi), dim=1
        )
        return action.clamp(-1.0, 1.0)


def score_teacher(n: int = 1000, seed: int = 12345, device: str = "cuda", **env_kwargs) -> dict:
    """The teacher on the same 1,000 fixed flights per mode as `scripts/eval_follow.py`."""
    out = {}
    for mode in ("follow", "orbit", "point", "all"):
        env = make_env("follow", n, device=device, seed=seed, mode=mode, difficulty=1.0, **env_kwargs)
        r = evaluate(Teacher(env), env, seed=seed)
        out[mode] = {
            "follow": 100 * r["follow_good_s"] / r["follow_scored_s"] if r["follow_scored_s"] > 0 else None,
            "orbit": 100 * r["orbit_good_s"] / r["orbit_scored_s"] if r["orbit_scored_s"] > 0 else None,
            "point": 100 * r["spots_reached"] / r["spots"] if r["spots"] > 0 else None,
            "crashed": 100 * r["crashed"],
        }
    return out


@torch.no_grad()
def record(env: FollowEnv, teacher: Teacher, pilot, steps: int, wobble: float, gamma: float) -> tuple:
    """Fly `pilot` (the teacher, or the student so far) with random wobbles for `steps` steps, and record what it saw,
    what the *teacher* would have done there, and what each moment turned out to be worth (the critic's target):
    G_t = r_t + γ · G_{t+1}, cut at crashes. The last 200 steps don't have enough future recorded (γ^200 ≈ 0.13), so
    they get no value target (NaN)."""
    obs = env.reset(stagger=True)
    observations, labels, rewards, ends = [], [], [], []
    for _ in range(steps):
        label = teacher.act(obs)
        flown = label if pilot is teacher else pilot.act(obs, deterministic=True)
        observations.append(obs)
        labels.append(label)
        obs, reward, _, info = env.step((flown + wobble * torch.randn_like(flown)).clamp(-1.0, 1.0))
        rewards.append(reward)
        ends.append(info["terminated"].float())  # crashes cut the future; running out of time doesn't
    returns = torch.full((steps, env.n_envs), float("nan"), device=env.device)
    future = torch.zeros(env.n_envs, device=env.device)
    for t in reversed(range(steps)):
        future = rewards[t] + gamma * (1.0 - ends[t]) * future
        if t < steps - 200:
            returns[t] = future
    return torch.cat(observations), torch.cat(labels), returns.flatten()


def clone(out: str, n: int = 4096, steps: int = 600, epochs: int = 30, device: str = "cuda", seed: int = 0,
          wobble: float = 0.15, dagger_rounds: int = 2) -> None:
    """Record the teacher and train a policy network to copy it; save a checkpoint `--init` can start from.

    DAgger ("dataset aggregation"): a copy is never perfect, so the student drifts into situations the teacher never
    flew into, and doesn't know what to do there. So after the first round, the *student* flies, the teacher says
    what it would have done in each moment, those examples are added, and the student trains again."""
    torch.manual_seed(seed)
    env = make_env("follow", n, device=device, seed=seed, difficulty=1.0)
    teacher = Teacher(env)
    config = PPOConfig()
    model = ActorCritic(env.obs_size, env.action_size, config.hidden, config.init_std, config.normalize_obs).to(device)
    optimizer = torch.optim.Adam(list(model.actor.parameters()) + list(model.critic.parameters()), lr=1e-3)
    data = []
    for round_ in range(1 + dagger_rounds):
        data.append(record(env, teacher, teacher if round_ == 0 else model, steps, wobble, config.gamma))
        observations, labels, returns = (torch.cat(parts) for parts in zip(*data))
        if round_ == 0:
            model.obs_norm.update(observations)
            model.obs_norm.count = torch.tensor(1e6, device=device)  # let PPO's statistics take over soon
        batch = 8192
        for epoch in range(epochs):
            order = torch.randperm(len(observations), device=device)
            total = value_total = 0.0
            for start in range(0, len(order), batch):
                idx = order[start : start + batch]
                copy_loss = ((model.actor(model.obs_norm(observations[idx])) - labels[idx]) ** 2).mean()
                valued = idx[~torch.isnan(returns[idx])]
                value = model.critic(model.obs_norm(observations[valued])).squeeze(-1)
                value_loss = ((value - returns[valued]) ** 2).mean()
                optimizer.zero_grad()
                (copy_loss + 1e-3 * value_loss).backward()  # (values are in the hundreds: scale so both matter)
                optimizer.step()
                total += copy_loss.item() * len(idx)
                value_total += value_loss.item() * len(idx)
        print(f"round {round_}: {len(observations) / 1e6:.1f} M examples, copy error {total / len(order):.4f}, "
              f"value error {value_total / len(order):.0f}")
    test_env = make_env("follow", 1000, device=device, seed=12345, difficulty=1.0)
    scores = evaluate(model, test_env, seed=12345)
    print(f"test (1,000 flights, all modes): success {100 * scores['success']:.1f} %, crashed {100 * scores['crashed']:.1f} %")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"task": "follow", "env_kwargs": {}, "iteration": 0, "config": asdict(config), "model": model.state_dict(),
         "cloned_from": "scripts/teach_follow.py Teacher", "clone_seed": seed, "dagger_rounds": dagger_rounds,
         "eval": {f"eval_{k}": v for k, v in scores.items()}},
        out,
    )
    print(f"saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score", action="store_true", help="score the teacher itself")
    parser.add_argument("--out", default=None, help="clone the teacher into this checkpoint")
    parser.add_argument("--person-sensor", default="camera")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0, help="clone seed")
    parser.add_argument("--dagger", type=int, default=2, help="DAgger rounds after the first copy")
    args = parser.parse_args()
    if args.score:
        print("| modes | follow | orbit | point | crashed |")
        print("|---|---|---|---|---|")
        for mode, s in score_teacher(args.n, device=args.device, person_sensor=args.person_sensor).items():
            cells = ["-" if s[k] is None else f"{s[k]:.1f} %" for k in ("follow", "orbit", "point")]
            print(f"| {mode} | {' | '.join(cells)} | {s['crashed']:.1f} % |")
    if args.out:
        clone(args.out, device=args.device, seed=args.seed, dagger_rounds=args.dagger)


if __name__ == "__main__":
    main()
