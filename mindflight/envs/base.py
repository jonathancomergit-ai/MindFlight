"""The shared part of every MindFlight task: N drones, each in its own copy of the world, stepped together on the GPU.

An *environment* ("env") is the world a learning agent lives in. Every 1/50 s it:
1. gets an **action** from the policy: 4 numbers in [−1, 1], one per motor,
2. turns them into motor commands and runs the physics for 10 steps of 2 ms (PLAN.md -> Timing),
3. hands back an **observation** (what the drone senses), a **reward** (how good that was) and **done** flags.

All N drones are rows of one tensor, so one env object is really N separate worlds. The drones never touch each
other, and each one's numbers are relative to its own world's origin.

A task (hover, fly to a point, gates, ...) subclasses `DroneEnv` and fills in four hooks:

| Hook | Says |
|---|---|
| `_reset_task(ids)` | where drone `ids` start and what their goal is |
| `_task_obs()` | extra observation columns the task adds (e.g. where the target is) |
| `_reward()` | the reward for the step that just happened |
| `_crashed()` | which drones have failed (episode ends with a penalty) |

and optionally `_finished()` (the drone completed the task early), `_post_step()` (bookkeeping such as switching
targets), `_episode_info(ids)` (task scores), `_cut_short()` (stop an episode early without blaming the drone) and
`update_curriculum(means)` (make the task harder as it improves).

Episodes end in two different ways, and PPO treats them differently:
- **terminated**: the drone crashed, or finished the whole task. Either way the episode is really over, so there is
  no future to count.
- **truncated**: time ran out. The drone was fine, so the learner should still count the future it would have had.
A finished drone is reset straight away inside `step`, so the batch never has holes in it.
"""

import math

import torch

from mindflight.physics.aero import DroneVariation
from mindflight.physics.controller import GAINS, attitude_action_to_setpoint, controller_step
from mindflight.physics.dynamics import OMEGA, PHYSICS_DT_S, POS, QUAT, RPM, STATE_SIZE, VEL, step
from mindflight.physics.fast import CudaGraphStepper
from mindflight.physics.params import CRAZYFLIE, CrazyflieParams, load_drone
from mindflight.physics.rotations import quat_from_axis_angle, quat_multiply, quat_to_matrix

POLICY_HZ = 50
SUBSTEPS = round(1 / (POLICY_HZ * PHYSICS_DT_S))  # 10 physics steps per action
POLICY_DT_S = SUBSTEPS * PHYSICS_DT_S  # 0.02 s

# Action a in [−1, 1] -> motor command rpm = hover_rpm · (1 + 0.5 · a).
# a = 0 is hover, a = +1 is 1.5 × hover RPM = full throttle (thrust ∝ rpm², and 1.5² = 2.25 = thrust-to-weight),
# a = −1 is half the hover RPM (a quarter of the hover thrust).
ACTION_RPM_SPAN = 0.5


def action_to_rpm(action: torch.Tensor, params: CrazyflieParams = CRAZYFLIE) -> torch.Tensor:
    """Policy action (N, 4) in [−1, 1] -> motor command (N, 4) in RPM: hover_rpm · (1 + 0.5 · a)."""
    return params.hover_rpm * (1.0 + ACTION_RPM_SPAN * action.clamp(-1.0, 1.0))


def random_unit_vectors(n: int, generator: torch.Generator, device) -> torch.Tensor:
    """n directions spread evenly over the sphere, (n, 3): a normal sample in 3-D, scaled to length 1."""
    v = torch.randn(n, 3, generator=generator, device=device)
    return v / v.norm(dim=1, keepdim=True).clamp_min(1e-9)


class DroneEnv:
    """N drones, one world each. Subclass it and fill in the hooks listed at the top of this file.

    n_envs: how many drones (and worlds). episode_s: time limit in seconds. device: "cuda" or "cpu".
    seed: makes the random starts repeatable.

    After `step`, these hold the step that just happened (handy for hooks and for the viewer):
        self.state           (N, 17) physics state, PLAN.md -> Conventions
        self.action          (N, 4)  the action just taken, clamped to [−1, 1]
        self.prev_action     (N, 4)  the action before that
        self.step_count      (N,)    policy steps since this drone's episode began
    """

    name = "base"
    base_obs_size = 3 + 9 + 3 + 4  # velocity, rotation matrix, angular velocity, last action: see `observe`
    task_obs_size = 0  # set by each task
    action_size = 4

    # On a GPU the 10 physics substeps run as one recorded CUDA graph with fused kernels (physics/fast.py): same
    # physics, ~50× faster. Set False to use the plain readable `step` loop instead.
    fast_physics = True

    def __init__(
        self,
        n_envs: int,
        episode_s: float = 10.0,
        device: str | None = None,
        seed: int = 0,
        drone: "str | CrazyflieParams | None" = None,
        action_mode: str = "motors",
    ):
        """drone: which drone file to fly (mindflight/drones/<name>.json); None = the Crazyflie 2.1+.

        action_mode: what the 4 action numbers mean.
            "motors"   (Phase 1): each one is a motor's speed, rpm = hover · (1 + 0.5 · a).
            "attitude" (Phase 2): thrust, roll, pitch, yaw rate, handed to the onboard controller
                       (physics/controller.py), exactly the kind of command a real Crazyflie takes over the radio.
        """
        if action_mode not in ("motors", "attitude"):
            raise ValueError(f"action_mode must be 'motors' or 'attitude', not {action_mode!r}")
        self.params = load_drone(drone)
        self.drone = self.params.name
        self.action_mode = action_mode
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_envs = n_envs
        self.device = torch.device(device)
        self.max_steps = round(episode_s / POLICY_DT_S)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

        self.state = torch.zeros(n_envs, STATE_SIZE, device=self.device)
        self.action = torch.zeros(n_envs, self.action_size, device=self.device)
        self.prev_action = torch.zeros_like(self.action)
        self.step_count = torch.zeros(n_envs, dtype=torch.long, device=self.device)
        # Running totals for the episode each drone is in, reported when it ends.
        self.episode_return = torch.zeros(n_envs, device=self.device)
        self.stepper: CudaGraphStepper | None = None  # built on first use (recording takes a few seconds)
        self.controller_integral = torch.zeros(n_envs, 3, device=self.device)  # attitude mode: the controller's memory
        # Level 5: per-drone mass/thrust/motor lag, wind, drag, ground effect (physics/aero.py). None = standard.
        self.variation: DroneVariation | None = None

    @property
    def obs_size(self) -> int:
        return self.base_obs_size + self.task_obs_size

    # ---------- the hooks a task fills in ----------

    def _reset_task(self, ids: torch.Tensor) -> None:
        """Set the start state (self.state[ids]) and the goal for drones `ids`."""
        raise NotImplementedError

    def _task_obs(self) -> torch.Tensor:
        """(N, task_obs_size) extra observation columns."""
        raise NotImplementedError

    def _reward(self) -> torch.Tensor:
        """(N,) reward for the step that just happened."""
        raise NotImplementedError

    def _crashed(self) -> torch.Tensor:
        """(N,) bool: drones that failed this step."""
        raise NotImplementedError

    def _finished(self) -> torch.Tensor:
        """Optional: (N,) bool, drones that completed the task this step (ends the episode, no penalty)."""
        return torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)

    def _cut_short(self) -> torch.Tensor:
        """Optional: (N,) bool, drones whose episode stops now through no fault of their own (packs: a pack-mate
        crashed, so the whole pack starts over). Counted like running out of time: the future still counts."""
        return torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)

    def update_curriculum(self, episode_means: dict[str, float]) -> dict[str, float]:
        """Optional: called by the trainer after each iteration with the averages of the episodes that ended
        (e.g. "episode_success"). A task can make itself harder here. Returns numbers to log (e.g. difficulty)."""
        return {}

    def display_objects(self) -> dict[str, torch.Tensor]:
        """Optional: the task's objects for the viewer, per drone, in that drone's own world. Keys the viewer knows:
        "targets" (N, 3); "gate_centre" (N, G, 3), "gate_heading" (N, G), "gate_opening" (N,), "next_gate" (N,);
        "pillars" (N, P, 3) as x, y, radius (radius 0 = no pillar); and general shapes "spheres" (N, K, 4: x, y, z,
        radius), "cylinders" (N, K, 5: x, y, z_bottom, height, radius), "lines" (N, K, 6: two end points), each with
        optional "<kind>_colors" (K hex strings). See viewer/policy_source.py -> PolicySource.scene."""
        return {}

    def _post_step(self) -> None:
        """Optional: task bookkeeping after the reward, before the next observation (e.g. switch to a new target)."""

    def _episode_info(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Optional: per-task numbers about the episodes of drones `ids` that just ended (e.g. success)."""
        return {}

    # ---------- helpers tasks can use ----------

    def uniform(self, n: int, low: float, high: float, size: int | None = None) -> torch.Tensor:
        """Random numbers in [low, high), shape (n,) or (n, size)."""
        shape = (n,) if size is None else (n, size)
        return low + (high - low) * torch.rand(shape, generator=self.generator, device=self.device)

    def random_tilt(self, n: int, max_tilt_rad: float) -> torch.Tensor:
        """n random orientations, (n, 4): tipped up to `max_tilt_rad` from level about a random horizontal axis,
        then turned to a random heading (yaw). Uniform in tilt angle, not in solid angle; fine for starts."""
        tilt_axis = torch.zeros(n, 3, device=self.device)
        tilt_axis[:, :2] = torch.randn(n, 2, generator=self.generator, device=self.device)
        tilt = quat_from_axis_angle(tilt_axis, self.uniform(n, 0.0, max_tilt_rad))
        up = torch.zeros(n, 3, device=self.device)
        up[:, 2] = 1.0
        heading = quat_from_axis_angle(up, self.uniform(n, -math.pi, math.pi))
        return quat_multiply(tilt, heading)  # yaw first (body frame), then tip

    # ---------- the env loop ----------

    def observe(self) -> torch.Tensor:
        """What the policy sees, (N, obs_size). Columns:

            0:3    linear velocity, world frame (m/s)
            3:12   rotation matrix R (body -> world), row by row. 9 numbers instead of a quaternion because q and −q
                   are the same rotation, which would confuse a network; R has no such double meaning.
            12:15  angular velocity, body frame (rad/s)
            15:19  the previous action (so the policy knows what the motors are still doing)
            19:    task columns from `_task_obs`

        Raw SI numbers; scaling them to a good range is the job of observation normalization in the learner.
        """
        rotation = quat_to_matrix(self.state[:, QUAT]).reshape(-1, 9)
        base = torch.cat((self.state[:, VEL], rotation, self.state[:, OMEGA], self.action), dim=1)
        return torch.cat((base, self._task_obs()), dim=1)

    def reset(self, stagger: bool = False) -> torch.Tensor:
        """Start a fresh episode for every drone and return the first observations.

        stagger=True (training): each drone starts at a random point of its time limit, as if it had already been
        flying a while. Otherwise all 4,096 drones would time out on the same step, and get new targets on the same
        step, forever: some rollouts would be full of endings and target jumps and others would have none, and the
        learner would lurch between them (we saw the value loss spike on that beat and the waypoint policy fall apart).
        """
        self._reset_ids(torch.arange(self.n_envs, device=self.device))
        if stagger:
            self.step_count = torch.randint(
                0, self.max_steps, (self.n_envs,), generator=self.generator, device=self.device
            )
        return self.observe()

    def _reset_ids(self, ids: torch.Tensor) -> None:
        """Fresh episode for drones `ids`: zero the counters, motors at hover, then the task places them."""
        self.state[ids] = 0.0
        self.state[ids, QUAT.start] = 1.0
        self.state[ids, RPM] = self.params.hover_rpm
        self.controller_integral[ids] = 0.0
        self.action[ids] = 0.0
        self.prev_action[ids] = 0.0
        self.step_count[ids] = 0
        self.episode_return[ids] = 0.0
        self._reset_task(ids)

    def physics(self, command: torch.Tensor) -> None:
        """Run SUBSTEPS physics steps with the command held fixed (a zero-order hold: the brain decides at 50 Hz).

        command: motor RPM (N, 4) in "motors" mode, or an attitude setpoint (N, 4) in "attitude" mode, in which case
        the onboard controller turns it into motor speeds before every physics step (500 Hz)."""
        attitude = self.action_mode == "attitude"
        if self.fast_physics and self.device.type == "cuda":
            if self.stepper is None:
                self.stepper = CudaGraphStepper(
                    self.n_envs,
                    self.device,
                    substeps=SUBSTEPS,
                    params=self.params,
                    compile=True,
                    variation=self.variation,
                    controller=GAINS if attitude else None,
                )
            if attitude:
                self.state, self.controller_integral = self.stepper.run_controlled(
                    self.state, command, self.controller_integral, variation=self.variation
                )
            else:
                self.state = self.stepper.run(self.state, command, variation=self.variation)
            return
        state, integral = self.state, self.controller_integral
        for _ in range(SUBSTEPS):
            if attitude:
                rpm_cmd, integral = controller_step(state, command, integral, PHYSICS_DT_S, self.params, GAINS)
            else:
                rpm_cmd = command
            state = step(state, rpm_cmd, params=self.params, variation=self.variation)
        self.state, self.controller_integral = state, integral

    def command(self, action: torch.Tensor) -> torch.Tensor:
        """The clamped action -> what `physics` takes: motor RPM, or an attitude setpoint (see action_mode)."""
        if self.action_mode == "attitude":
            return attitude_action_to_setpoint(action, self.params)
        return action_to_rpm(action, self.params)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Take one action for every drone. Returns (obs, reward, done, info).

        action: (N, 4), any real numbers (clamped to [−1, 1] here).
        obs: (N, obs_size) for the *next* step. For drones that just finished, this is already the first
             observation of their new episode; their last observation is in info["final_obs"].
        reward: (N,). done: (N,) bool = terminated | truncated.
        info: "terminated", "truncated" (N,) bool, "final_obs" (N, obs_size), and "episodes": a dict of
              per-episode numbers ("return", "length_s", plus the task's own) for the drones that finished.
        """
        self.prev_action = self.action
        self.action = action.clamp(-1.0, 1.0)
        self.physics(self.command(self.action))
        self.step_count += 1

        reward = self._reward()
        crashed = self._crashed()
        terminated = crashed | self._finished()
        truncated = ((self.step_count >= self.max_steps) | self._cut_short()) & ~terminated
        done = terminated | truncated
        self.episode_return += reward
        self._post_step()

        final_obs = self.observe()
        info = {"terminated": terminated, "truncated": truncated, "final_obs": final_obs, "episodes": {}}
        ids = done.nonzero(as_tuple=True)[0]
        if len(ids) > 0:
            info["episodes"] = {
                "return": self.episode_return[ids].clone(),
                "length_s": self.step_count[ids].float() * POLICY_DT_S,
                "crashed": crashed[ids].float(),
                **self._episode_info(ids),
            }
            self._reset_ids(ids)
            obs = self.observe()
        else:
            obs = final_obs
        return obs, reward, done, info

    def poses(self) -> torch.Tensor:
        """(N, 7) position + quaternion of every drone, the viewer's frame format."""
        return self.state[:, POS.start : QUAT.stop]
