"""Tasks: observations, reward, reset and done for hover, waypoint, gates, obstacles.

`TASKS` maps a task name (as typed on the command line) to its env class.
"""

from mindflight.envs.base import DroneEnv
from mindflight.envs.gates import GatesEnv
from mindflight.envs.hover import HoverEnv
from mindflight.envs.obstacles import ObstaclesEnv
from mindflight.envs.pack import PackEnv
from mindflight.envs.race import RaceEnv
from mindflight.envs.sturdy import SturdyEnv
from mindflight.envs.waypoint import WaypointEnv

TASKS: dict[str, type[DroneEnv]] = {
    HoverEnv.name: HoverEnv,
    WaypointEnv.name: WaypointEnv,
    GatesEnv.name: GatesEnv,
    ObstaclesEnv.name: ObstaclesEnv,
    RaceEnv.name: RaceEnv,
    SturdyEnv.name: SturdyEnv,
    PackEnv.name: PackEnv,
}


def make_env(task: str, n_envs: int, device: str | None = None, seed: int = 0, **kwargs) -> DroneEnv:
    """Build the env for `task` with `n_envs` drones."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {sorted(TASKS)}")
    return TASKS[task](n_envs, device=device, seed=seed, **kwargs)
