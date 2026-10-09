import torch

from mindflight.envs.race import RaceEnv
from mindflight.race import Entrant, RaceSource
from mindflight.rl.ppo import ActorCritic


def make_race(n_entrants=3):
    entrants = [Entrant(f"p{i}", f"p{i}.pt") for i in range(n_entrants)]
    obs_size = RaceEnv(1, device="cpu").obs_size
    models = [ActorCritic(obs_size, 4) for _ in entrants]
    return RaceSource(entrants, models, drones_each=2)


def test_race_env_is_full_difficulty_and_hurries():
    env = RaceEnv(4, device="cpu")
    assert env.difficulty == 1.0 and env.time_cost > 0
    assert env.gate_opening.tolist() == [0.0] * 4  # set at reset
    env.reset()
    assert env.gate_opening.tolist() == [0.5] * 4


def test_every_drone_shares_the_course_and_flies_its_own_policy():
    race = make_race()
    env = race.env
    assert env.n_envs == 6
    assert torch.equal(env.gate_centre[0], env.gate_centre[5])
    assert race.policy_of_drone.tolist() == [0, 0, 1, 1, 2, 2]


def test_a_round_ends_and_scores_when_every_drone_is_done():
    race = make_race()
    # Untrained policies crash quickly; run until the first round is scored.
    for _ in range(2000):
        race.advance(1)
        if race.round_time_s == 0 or race.pause_s > 0:
            break
    assert race.pause_s > 0  # round over, showing results
    assert race.board["title"].startswith("Race · round 1")
    assert len(race.board["rows"]) == 3
    total = sum(e.finishes + e.crashes for e in race.entrants)
    assert total == 6  # every drone had exactly one try
    poses = race.frame(0.0)
    assert poses.shape == (6, 7)
