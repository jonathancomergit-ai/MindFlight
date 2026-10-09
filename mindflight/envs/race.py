"""Level 6 task: race. The gate course at full difficulty, but now the clock matters.

Same course, same rules and observations as `gates.py`. Two changes to the reward make the drone hurry:

    reward = 15 · (metres closer to the next gate)       (was 10: covering ground is worth more)
           − 0.05 every step                              (new: every 1/50 s on the course costs a little)
           + gate bonuses, − spin/twitch, crash −5        (as before)

The time cost is kept small on purpose. If it were big, a slow drone could "win" by crashing early to stop the
clock. At 1 m/s the progress reward (15 × 0.02 m = 0.3 a step) is already far above the 0.05 time cost.

A race policy starts from a trained gates policy (`--init policies/gates.pt`) and learns to go faster.
"""

from mindflight.envs.gates import GatesEnv


class RaceEnv(GatesEnv):
    name = "race"
    has_curriculum = False
    progress_weight = 15.0
    time_cost = 0.05

    def __init__(
        self,
        n_envs: int,
        episode_s: float = 20.0,
        device: str | None = None,
        seed: int = 0,
        difficulty: float = 1.0,
        shared_course: bool = False,
        **options,
    ):
        super().__init__(
            n_envs, episode_s, device, seed, difficulty=difficulty, shared_course=shared_course, **options
        )
