import pytest
import torch

from mindflight.rl.ppo import PPO, PPOConfig, RolloutBuffer, RunningNorm, compute_gae


def test_gae_matches_hand_calculation():
    """Two steps, one drone, no done: check δ and A against the formulas written out by hand."""
    gamma, lam = 0.9, 0.8
    buffer = RolloutBuffer(2, 1, obs_size=1, action_size=1, device="cpu")
    buffer.rewards[:, 0] = torch.tensor([1.0, 2.0])
    buffer.values[:, 0] = torch.tensor([0.5, 0.25])
    last_value = torch.tensor([4.0])
    compute_gae(buffer, last_value, gamma, lam)
    delta1 = 2.0 + gamma * 4.0 - 0.25
    delta0 = 1.0 + gamma * 0.25 - 0.5
    a1 = delta1
    a0 = delta0 + gamma * lam * a1
    assert buffer.advantages[:, 0].tolist() == pytest.approx([a0, a1])
    assert buffer.returns[:, 0].tolist() == pytest.approx([a0 + 0.5, a1 + 0.25])


def test_gae_stops_at_done():
    buffer = RolloutBuffer(2, 1, obs_size=1, action_size=1, device="cpu")
    buffer.rewards[:, 0] = torch.tensor([1.0, 1.0])
    buffer.dones[0, 0] = 1.0  # the episode ended after step 0
    compute_gae(buffer, torch.tensor([100.0]), gamma=0.99, lam=0.95)
    assert buffer.advantages[0, 0].item() == pytest.approx(1.0)  # nothing leaks back from after the reset


def test_running_norm_matches_all_data_at_once():
    torch.manual_seed(0)
    norm = RunningNorm(3)
    batches = [torch.randn(100, 3) * torch.tensor([1.0, 10.0, 0.1]) + 5 for _ in range(7)]
    for b in batches:
        norm.update(b)
    everything = torch.cat(batches)
    assert torch.allclose(norm.mean, everything.mean(0), atol=1e-4)
    assert torch.allclose(norm.var, everything.var(0, unbiased=False), rtol=1e-3)
    assert norm(everything).std(0).tolist() == pytest.approx([1, 1, 1], abs=0.01)


def test_ppo_learns_a_one_step_task():
    """A one-step game: the reward is −|action − 0.5|² for each of 4 motors. PPO should move its mean to ≈ 0.5."""
    torch.manual_seed(0)
    config = PPOConfig(rollout_steps=8, epochs=4, minibatches=2, hidden=(32, 32), learning_rate=3e-3)
    ppo = PPO(obs_size=2, action_size=4, config=config, device="cpu")
    n = 256
    obs = torch.ones(n, 2)
    for _ in range(40):
        buffer = RolloutBuffer(config.rollout_steps, n, 2, 4, "cpu")
        with torch.no_grad():
            for t in range(config.rollout_steps):
                dist = ppo.model.distribution(obs)
                action = dist.sample()
                buffer.obs[t] = obs
                buffer.actions[t] = action
                buffer.log_probs[t] = dist.log_prob(action).sum(-1)
                buffer.values[t] = ppo.model.value(obs)
                buffer.rewards[t] = -((action - 0.5) ** 2).sum(-1)
                buffer.dones[t] = 1.0  # every step is a whole episode
        compute_gae(buffer, torch.zeros(n), config.gamma, config.gae_lambda)
        ppo.update(buffer)
    mean_action = ppo.model.act(obs[:1], deterministic=True)
    assert mean_action.tolist()[0] == pytest.approx([0.5] * 4, abs=0.1)
