"""PPO (Proximal Policy Optimization) in one file, for thousands of drones at once on the GPU.

The big idea, in five lines:
1. The **policy** is a neural network: observation in, a *range* of good actions out (a Gaussian: mean + spread).
2. Let every drone fly with it for a short while (a **rollout**), trying slightly random actions.
3. The **value** network guesses "how much reward will I get from here?". Comparing what really happened with that
   guess tells us whether each action turned out **better or worse than expected** (the **advantage**).
4. Nudge the policy to make better-than-expected actions more likely and worse ones less likely.
5. ...but only a little each time (the **clip**), because the rollout was collected with the old policy and stops
   being trustworthy if the policy changes a lot. That "proximal" step is what makes PPO stable.

Pieces, top to bottom: `RunningNorm` (scales observations), `ActorCritic` (the two networks), `RolloutBuffer`
(what happened), `compute_gae` (advantages), `PPO.update` (the learning step).
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PPOConfig:
    rollout_steps: int = 24  # policy steps per drone per rollout (24 × 4,096 drones ≈ 98 k samples)
    epochs: int = 5  # passes over each rollout
    minibatches: int = 4  # each pass is split into this many gradient steps
    gamma: float = 0.99  # discount: a reward 1 s (50 steps) away counts 0.99⁵⁰ ≈ 0.6 as much as one now
    gae_lambda: float = 0.95  # GAE: how far to trust real rewards vs. the value guess (1 = all real, 0 = all guess)
    clip: float = 0.2  # the "proximal" limit: the new/old probability ratio is kept within 1 ± 0.2
    learning_rate: float = 3e-4
    target_kl: float = 0.01  # adaptive learning rate aims for this much policy change per update
    value_coef: float = 1.0
    entropy_coef: float = 0.0  # > 0 pays the policy for staying random (exploring); the Gaussian spread already does
    max_grad_norm: float = 1.0
    hidden: tuple[int, ...] = (256, 256, 128)
    init_std: float = 0.2  # starting spread of the actions (0.5 flips the twitchy Crazyflie in ~0.4 s)
    normalize_obs: bool = True  # scale observations with RunningNorm (Level 2 shows why)


class RunningNorm(nn.Module):
    """Keeps a running mean and variance of every observation column and rescales observations to ≈ mean 0, std 1.

    Networks learn badly when one input is in the thousands and another is 0.001: the big one swamps the rest.
    Normalizing puts every column on the same footing. (Level 2's doc shows the before/after.)

        normalized = clamp((x − mean) / sqrt(var + 1e-8), −5, 5)

    Mean and variance are merged batch by batch with the parallel-variance formula (Chan et al.), so it works on
    any number of samples at once and stays exact.
    """

    def __init__(self, size: int, clip: float = 5.0, enabled: bool = True):
        super().__init__()
        self.enabled = enabled  # False: pass observations through untouched (for the Level 2 comparison)
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if not self.enabled:
            return
        batch_mean, batch_var, batch_count = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.var = (self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total) / total
        self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


def mlp(sizes: list[int], out_gain: float) -> nn.Sequential:
    """A plain stack of Linear + ELU layers. The last layer starts small (out_gain) so the first actions are calm."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        last = i == len(sizes) - 2
        nn.init.orthogonal_(linear.weight, gain=out_gain if last else 2**0.5)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if not last:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Two networks that share only the observation scaling.

    actor (the policy):  obs -> mean action μ (4 numbers). The spread σ is a separate learned number per motor.
                         The action is drawn from a Gaussian N(μ, σ²); at test time we just use μ.
    critic (the value):  obs -> V, the expected discounted sum of future rewards from here.
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden: tuple[int, ...] = (256, 256, 128),
        init_std: float = 0.2,
        normalize_obs: bool = True,
    ):
        super().__init__()
        self.obs_norm = RunningNorm(obs_size, enabled=normalize_obs)
        self.actor = mlp([obs_size, *hidden, action_size], out_gain=0.01)
        self.critic = mlp([obs_size, *hidden, 1], out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((action_size,), float(torch.log(torch.tensor(init_std)))))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(self.obs_norm(obs))
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.obs_norm(obs)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """The action to fly with: the mean (deterministic, for testing) or a random draw (for learning)."""
        if deterministic:
            return self.actor(self.obs_norm(obs))
        return self.distribution(obs).sample()


class RolloutBuffer:
    """Everything that happened in one rollout, shaped (T steps, N drones, ...), kept on the GPU."""

    def __init__(self, steps: int, n_envs: int, obs_size: int, action_size: int, device):
        def zeros(*shape):
            return torch.zeros(steps, n_envs, *shape, device=device)

        self.obs = zeros(obs_size)
        self.actions = zeros(action_size)
        self.log_probs = zeros()
        self.rewards = zeros()
        self.dones = zeros()
        self.values = zeros()
        self.advantages = zeros()
        self.returns = zeros()


def compute_gae(buffer: RolloutBuffer, last_value: torch.Tensor, gamma: float, lam: float) -> None:
    """Generalized Advantage Estimation, filled into buffer.advantages and buffer.returns.

    One-step surprise (the TD error):   δ_t = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)
        "the reward I got, plus what the next state is worth, minus what I expected".
    Advantage: a decaying sum of surprises, walking backward in time:
        A_t = δ_t + γλ · (1 − done_t) · A_{t+1}
    Return (the target the value network learns):  R_t = A_t + V(s_t)

    `done` cuts the chain at crashes and resets. Time-limit endings are handled before this (in the trainer): their
    reward already has γ · V(final state) added, so running out of time doesn't look like a crash.
    """
    advantage = torch.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(buffer.rewards.shape[0])):
        not_done = 1.0 - buffer.dones[t]
        delta = buffer.rewards[t] + gamma * next_value * not_done - buffer.values[t]
        advantage = delta + gamma * lam * not_done * advantage
        buffer.advantages[t] = advantage
        next_value = buffer.values[t]
    buffer.returns = buffer.advantages + buffer.values


class PPO:
    """The learner: owns the networks and the optimizer, and turns a full RolloutBuffer into better networks."""

    def __init__(self, obs_size: int, action_size: int, config: PPOConfig, device):
        self.config = config
        self.model = ActorCritic(obs_size, action_size, config.hidden, config.init_std, config.normalize_obs).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.learning_rate = config.learning_rate

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """A few epochs of minibatch gradient steps on the clipped PPO loss. Returns averages for logging.

        For each sample, ratio = π_new(a | s) / π_old(a | s), and A = its advantage (normalized per minibatch).
            policy loss  = −mean( min(ratio · A, clip(ratio, 1 − ε, 1 + ε) · A) )
                The min takes the more pessimistic of "follow the ratio" and "follow the clipped ratio", so pushing
                an action's probability more than ε past the old policy earns nothing extra.
            value loss   = mean( (V(s) − R)² )
            total        = policy loss + value_coef · value loss − entropy_coef · entropy

        Adaptive learning rate: after each minibatch we measure how far the policy moved (KL divergence between the
        old and new Gaussians). Too far (> 2 × target) -> learning rate ÷ 1.5; too timid (< target / 2) -> × 1.5.
        """
        c = self.config
        obs = buffer.obs.flatten(0, 1)
        actions = buffer.actions.flatten(0, 1)
        old_log_probs = buffer.log_probs.flatten(0, 1)
        advantages = buffer.advantages.flatten(0, 1)
        returns = buffer.returns.flatten(0, 1)

        with torch.no_grad():
            old_dist = self.model.distribution(obs)
            old_mean, old_std = old_dist.mean, old_dist.stddev

        n_samples = obs.shape[0]
        batch_size = n_samples // c.minibatches
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clip_fraction": 0.0}
        n_updates = 0
        for _ in range(c.epochs):
            order = torch.randperm(n_samples, device=obs.device)
            for start in range(0, batch_size * c.minibatches, batch_size):
                idx = order[start : start + batch_size]
                dist = self.model.distribution(obs[idx])
                log_prob = dist.log_prob(actions[idx]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                # How far did the policy move? KL(old ‖ new) for diagonal Gaussians, summed over the 4 motors.
                with torch.no_grad():
                    new_std = dist.stddev
                    kl = (
                        torch.log(new_std / old_std[idx])
                        + (old_std[idx] ** 2 + (old_mean[idx] - dist.mean) ** 2) / (2 * new_std**2)
                        - 0.5
                    ).sum(-1).mean()
                    if kl > 2 * c.target_kl:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl < c.target_kl / 2:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate

                adv = advantages[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = torch.exp(log_prob - old_log_probs[idx])
                policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - c.clip, 1 + c.clip) * adv).mean()
                value_loss = ((self.model.value(obs[idx]) - returns[idx]) ** 2).mean()
                loss = policy_loss + c.value_coef * value_loss - c.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), c.max_grad_norm)
                self.optimizer.step()

                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"] += value_loss.item()
                totals["entropy"] += entropy.item()
                totals["kl"] += kl.item()
                totals["clip_fraction"] += ((ratio - 1).abs() > c.clip).float().mean().item()
                n_updates += 1

        stats = {k: v / n_updates for k, v in totals.items()}
        stats["learning_rate"] = self.learning_rate
        stats["action_std"] = self.model.log_std.exp().mean().item()
        return stats
