"""The training loop, evaluation and checkpoints. `python -m mindflight.train` is the command-line front end.

One training **iteration**:
1. Rollout: every drone flies `rollout_steps` policy steps with the current (slightly random) policy.
2. Advantages: `compute_gae` works out which actions turned out better or worse than expected.
3. Update: `PPO.update` nudges the networks.
4. Observation statistics are refreshed with what this rollout saw (after the update, so the rollout and the update
   both used the same scaling).
5. Log a line to the console and to `metrics.jsonl`; save a checkpoint now and then.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch

from mindflight.envs import make_env
from mindflight.envs.base import DroneEnv
from mindflight.rl.ppo import PPO, ActorCritic, PPOConfig, RolloutBuffer, compute_gae

RUNS_DIR = Path("runs")


class EpisodeStats:
    """Averages of the per-episode numbers (return, length, success, ...) of every episode that ended recently."""

    def __init__(self):
        self.sums: dict[str, float] = {}
        self.count = 0

    def add(self, episodes: dict[str, torch.Tensor]) -> None:
        if not episodes:
            return
        n = len(next(iter(episodes.values())))
        for key, values in episodes.items():
            self.sums[key] = self.sums.get(key, 0.0) + values.sum().item()
        self.count += n

    def means(self) -> dict[str, float]:
        return {f"episode_{k}": v / self.count for k, v in self.sums.items()} if self.count else {}


def save_checkpoint(path: Path, ppo: PPO, task: str, env_kwargs: dict, iteration: int, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task": task,
            "env_kwargs": env_kwargs,
            "iteration": iteration,
            "config": asdict(ppo.config),
            "model": ppo.model.state_dict(),
            **extra,
        },
        path,
    )


def load_policy(path: str | Path, device: str | torch.device) -> tuple[ActorCritic, dict]:
    """Load a checkpoint -> (model in eval mode, the checkpoint dict). Works for files saved by `save_checkpoint`."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = PPOConfig(**checkpoint["config"])
    state = checkpoint["model"]
    obs_size = state["obs_norm.mean"].shape[0]
    action_size = state["log_std"].shape[0]
    model = ActorCritic(obs_size, action_size, tuple(config.hidden), config.init_std, config.normalize_obs).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def evaluate(model: ActorCritic, env: DroneEnv, deterministic: bool = True, seed: int = 12345) -> dict[str, float]:
    """Fly exactly one full episode per drone (env.n_envs episodes) and average the per-episode numbers.

    The env is re-seeded first, so every evaluation sees the same starts and scores can be compared fairly.
    Drones that finish early are reset by the env and keep flying, but only their *first* episode is counted.
    """
    env.generator.manual_seed(seed)
    obs = env.reset()
    counted = torch.zeros(env.n_envs, dtype=torch.bool, device=env.device)
    sums: dict[str, torch.Tensor] = {}
    for _ in range(env.max_steps + 1):
        obs, _, done, info = env.step(model.act(obs, deterministic=deterministic))
        first = done & ~counted
        if first.any():
            ended = done.nonzero(as_tuple=True)[0]
            keep = first[ended]  # which of the drones that just ended are on their first episode
            for key, values in info["episodes"].items():
                sums[key] = sums.get(key, torch.zeros((), device=env.device)) + values[keep].sum()
            counted |= first
        if counted.all():
            break
    return {key: (value / env.n_envs).item() for key, value in sums.items()}


def train(
    task: str,
    n_envs: int = 4096,
    iterations: int = 300,
    seed: int = 0,
    device: str | None = None,
    config: PPOConfig | None = None,
    env_kwargs: dict | None = None,
    run_dir: Path | None = None,
    init_checkpoint: str | Path | None = None,
    eval_every: int = 50,
    on_iteration: Callable[[DroneEnv, dict], None] | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Train a policy for `task`. Returns the run folder, which holds metrics.jsonl, latest.pt and best.pt.

    on_iteration(env, metrics) is called after every iteration (the live viewer uses it).
    init_checkpoint: start from an existing policy instead of from scratch (fine-tuning, curricula).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    config = config or PPOConfig()
    env_kwargs = env_kwargs or {}
    torch.manual_seed(seed)
    env = make_env(task, n_envs, device=device, seed=seed, **env_kwargs)
    ppo = PPO(env.obs_size, env.action_size, config, device)
    if init_checkpoint is not None:
        model, _ = load_policy(init_checkpoint, device)
        ppo.model.load_state_dict(model.state_dict())
    # Evaluation always uses the task's hardest setting, so the score means the same thing all through training.
    eval_kwargs = dict(env_kwargs)
    if getattr(env, "has_curriculum", False):
        eval_kwargs.setdefault("difficulty", 1.0)
    eval_env = make_env(task, 1000, device=device, seed=seed + 12345, **eval_kwargs)

    run_dir = run_dir or RUNS_DIR / task / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"task": task, "n_envs": n_envs, "seed": seed, "env_kwargs": env_kwargs, **asdict(config)}, indent=2)
    )
    metrics_file = open(run_dir / "metrics.jsonl", "a", encoding="utf-8")
    log(f"run folder: {run_dir}   task {task}, {n_envs} drones on {device}, obs {env.obs_size}")

    buffer = RolloutBuffer(config.rollout_steps, n_envs, env.obs_size, env.action_size, device)
    obs = env.reset()
    samples = 0
    best_success = -1.0
    start_s = time.perf_counter()
    for iteration in range(1, iterations + 1):
        # 1. Rollout.
        episode_stats = EpisodeStats()
        rollout_start_s = time.perf_counter()
        with torch.no_grad():
            for t in range(config.rollout_steps):
                dist = ppo.model.distribution(obs)
                action = dist.sample()
                buffer.obs[t] = obs
                buffer.actions[t] = action
                buffer.log_probs[t] = dist.log_prob(action).sum(-1)
                buffer.values[t] = ppo.model.value(obs)

                obs, reward, done, info = env.step(action)
                # Out of time is not a crash: add what the final state was still worth (bootstrapping).
                truncated = info["truncated"]
                if truncated.any():
                    reward = reward + config.gamma * ppo.model.value(info["final_obs"]) * truncated
                buffer.rewards[t] = reward
                buffer.dones[t] = done.float()
                episode_stats.add(info["episodes"])
            last_value = ppo.model.value(obs)
        rollout_s = time.perf_counter() - rollout_start_s

        # 2–4. Advantages, update, observation statistics.
        compute_gae(buffer, last_value, config.gamma, config.gae_lambda)
        update_start_s = time.perf_counter()
        stats = ppo.update(buffer)
        ppo.model.obs_norm.update(buffer.obs.flatten(0, 1))
        update_s = time.perf_counter() - update_start_s
        samples += config.rollout_steps * n_envs

        # 5. Log and save.
        curriculum = env.update_curriculum(episode_stats.means())
        metrics = {
            "iteration": iteration,
            "samples": samples,
            "time_s": round(time.perf_counter() - start_s, 2),
            "samples_per_s": round(config.rollout_steps * n_envs / (rollout_s + update_s)),
            "reward_per_step": buffer.rewards.mean().item(),
            **episode_stats.means(),
            **curriculum,
            **stats,
        }
        if iteration % eval_every == 0 or iteration == iterations:
            metrics.update({f"eval_{k}": v for k, v in evaluate(ppo.model, eval_env).items()})
            success = metrics.get("eval_success", 0.0)
            extra = {"eval": {k: v for k, v in metrics.items() if k.startswith("eval_")}}
            save_checkpoint(run_dir / "latest.pt", ppo, task, env_kwargs, iteration, extra)
            if success >= best_success:
                best_success = success
                save_checkpoint(run_dir / "best.pt", ppo, task, env_kwargs, iteration, extra)
        metrics_file.write(json.dumps(metrics) + "\n")
        metrics_file.flush()
        log(format_metrics(metrics))
        if on_iteration is not None:
            on_iteration(env, metrics)

    metrics_file.close()
    return run_dir


def format_metrics(m: dict) -> str:
    """One short console line per iteration."""
    parts = [f"it {m['iteration']:4d}", f"{m['samples'] / 1e6:6.1f} M samples", f"{m['samples_per_s'] / 1e3:5.0f} k/s"]
    if "episode_return" in m:
        parts.append(f"return {m['episode_return']:7.1f}")
        parts.append(f"len {m['episode_length_s']:4.1f} s")
        parts.append(f"crash {100 * m['episode_crashed']:3.0f}%")
    if "episode_success" in m:
        parts.append(f"success {100 * m['episode_success']:3.0f}%")
    parts.append(f"std {m['action_std']:.2f}")
    if "eval_success" in m:
        parts.append(f"| EVAL success {100 * m['eval_success']:.1f}% crash {100 * m['eval_crashed']:.1f}%")
    return "  ".join(parts)
