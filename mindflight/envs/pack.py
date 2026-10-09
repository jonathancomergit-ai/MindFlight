"""Phase 2 · P5 task: packs. Many drones share one sky and fly in formation.

Until now every drone had a world to itself. Here the N drones are split into **worlds of G** (default 6):
drone i lives in world `i // G`. Drones in the same world share the air: they can bump into each other, and a
drone flying above another pushes it down with its prop wash (downwash). Different worlds never meet.

Each world flies one **formation** (hexagon, line or V, neighbours `spacing_m` = 0.7 m apart). Every 3 s the
formation's centre moves 0.5–2 m to a new waypoint, turns by up to ±90°, and half the time switches to a new
random shape. Each drone has a fixed **slot** (drone k of a world always takes slot k), so its target is

    target = centre + R_z(heading) · slot_offset[k]

Fixed slots are the simplest rule a real pack could use: nobody has to agree who goes where. The price: drones
must cross each other's paths. At the start they are scattered at random over a 3 × 3 × 2 grid of start points
(0.8 m apart) and the first formation is centred right there ("get into formation"); a change of shape (hexagon ->
line) sends drones through the middle again. That is where the neighbours matter.

**One brain, many drones (decentralized).** Every drone runs the same policy on its own observation. Nobody is in
charge. What a drone sees:

| Columns | What |
|---|---|
| 0:19 | the usual: velocity, rotation matrix, spin, last action (`base.py -> observe`) |
| 19:22 | its own target − its position (m), exactly like the waypoint task |
| 22:22+6k | its k nearest pack-mates, nearest first: their position − mine (3), their velocity − mine (3) |
| 22+6k | distance to the nearest pack-mate (m) |
| then 3k | (share_goals) where those pack-mates are heading: their target − their position (m), same order |

k = 3 by default: 50 numbers. We first showed it the whole pack (k = 5). Then the drones at the ends of a line
wobbled: the far pack-mates (up to 3.5 m away, all on one side) are big, odd inputs that have nothing to do with
holding a slot. Hiding the 2 furthest made the same brain reach 30 % more line waypoints, so k = 3.
Columns 0:22 mean the same as in the waypoint task, so a trained waypoint brain can be "widened" to start from
(`scripts/widen_policy.py`). Sharing goals is cheap on
real drones (the same radio that shares positions), and it answers a question position and velocity can't: "is
that drone coming at me going to stop next to me, or fly through me?"

**Reward** (per step), the waypoint reward plus a crowding term:

    reward = 3 − 0.5 · min(d, 4) + 0.5 · max(0, 1 − d / 0.3) + 0.5 · [d < 15 cm] − tilt, spin, twitch (as hover)
           − crowding                       in [0, 1]: pack-mates too close, or closing in fast (see `crowding`)
    crash: −30 once

d = distance to its own slot. The alive bonus is 3 (waypoint had 2) because crowding can take one more point away:
the reward must stay ≥ 0, or crashing would pay.

**Crashes.** Floor, upside down, spinning, leaving the 8 × 8 × 3.5 m arena, or a **collision**: two drones' centres
closer than 2 × collision radius (12 cm) at any moment of the step (checked along the straight path each drone flew
in the last 20 ms, so fast drones can't skip through each other). Both drones crash.
When any drone of a world crashes, the **whole world** starts over: the crashed drones get the −30 and a real ending;
their pack-mates did nothing wrong, so for them the episode is "cut short" (like running out of time, the future
still counts: `base.py -> _cut_short`).

**Success.** A waypoint counts when *every* drone of the world has been within 15 cm of its slot for the last 0.5 s
before the formation moves on. An episode is 12 s = 4 waypoints, and its success is the fraction reached (a crash
fails the rest). Every drone of a world reports the same number, so the average over drones is the average over
packs. Also reported: "collided" (this drone ended in a collision), "pack_collided" (its pack had a collision) and
"downwash_time" (share of steps it was pushed down by more than 5 % of its weight).
"""

import math

import torch

from mindflight.envs.base import random_unit_vectors
from mindflight.envs.hover import HoverEnv
from mindflight.physics.aero import DroneVariation, downwash_grouped
from mindflight.physics.dynamics import OMEGA, POS, QUAT, VEL
from mindflight.physics.rotations import quat_to_matrix

GROUP_SIZE = 6
NEIGHBOURS = 3  # each drone sees its 3 nearest pack-mates (see the note on k at the top)
SPACING_M = 0.7  # between neighbouring slots
WAYPOINT_HOLD_S = 3.0  # the formation moves on this often
SETTLE_S = 0.5
SUCCESS_RADIUS_M = 0.15
TOO_CLOSE_M = 0.5  # crowding penalty starts here (slots are ≥ 0.7 m apart, so a settled pack pays nothing)
APPROACH_RANGE_M = 1.0  # closing in on a pack-mate nearer than this costs a little (see `crowding`)
APPROACH_WEIGHT = 0.5  # per m/s of closing speed
CENTRE_STEP_M = (0.5, 2.0)  # each new formation centre is this far from the last one
RESHAPE_CHANCE = 0.5  # chance that the pack switches to a new random formation at a waypoint
CENTRE_XY_M = 1.25  # formation centres: x, y in [−1.25, 1.25] m ...
CENTRE_Z_M = (1.0, 2.0)  # ... and z in [1, 2] m. A 6-drone line is 3.5 m long, so slots reach |x|, |y| ≤ 3 m.
ARENA_XY_M = 4.0  # leaving |x| or |y| > 4 m, or going above 3.5 m, is a crash (1 m beyond the furthest slot)
ARENA_TOP_M = 3.5
MAX_TURN_DEG = 90.0  # the formation turns by up to this much at each new waypoint
START_CELL_M = 0.8  # start points: a grid with cells this far apart, two layers 0.6 m apart
CRASH_PENALTY = 30.0
FORMATIONS = ("hexagon", "line", "vee")


def _formation(shape: str, group_size: int, spacing_m: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(offsets (G, 3), links (G,)) before numbering the slots right to left. See `formation_offsets`."""
    i = torch.arange(group_size, dtype=torch.float32)
    k = torch.arange(group_size)
    offsets = torch.zeros(group_size, 3)
    if shape == "hexagon":
        radius = spacing_m / (2 * math.sin(math.pi / group_size))  # chord between neighbours = spacing_m
        angle = 2 * math.pi * i / group_size
        offsets[:, 0] = radius * torch.cos(angle)
        offsets[:, 1] = radius * torch.sin(angle)
        links = (k + 1) % group_size  # a ring
    elif shape == "line":
        offsets[:, 1] = (i - (group_size - 1) / 2) * spacing_m
        links = torch.clamp(k + 1, max=group_size - 1)  # a chain
    elif shape == "vee":
        rank = torch.div(i + 1, 2, rounding_mode="floor")  # 0, 1, 1, 2, 2, 3, ...: the leader, then pairs
        side = torch.where(i % 2 == 1, 1.0, -1.0)
        offsets[:, 0] = -rank * spacing_m * math.cos(math.radians(45))
        offsets[:, 1] = side * rank * spacing_m * math.sin(math.radians(45))
        links = torch.where(k <= 2, torch.zeros_like(k), k - 2)  # each drone to the one ahead on its arm
    else:
        raise ValueError(f"unknown formation {shape!r}; choose from {FORMATIONS}")
    return offsets - offsets.mean(dim=0), links


def _right_to_left(offsets: torch.Tensor) -> torch.Tensor:
    """Slot order: by y (right to left), ties front first. (Spots with equal y differ in x by ≥ 0.5 m.)"""
    return torch.argsort(offsets[:, 1] - 1e-3 * offsets[:, 0])


def formation_offsets(shape: str, group_size: int, spacing_m: float) -> torch.Tensor:
    """Slot offsets (G, 3) in m in the formation's own frame (x forward, y left, z up), centred on their average.

    hexagon: a ring, neighbours spacing_m apart (for 6 drones a regular hexagon with radius = spacing_m).
    line:    side by side across the direction of travel, spacing_m apart.
    vee:     one leader in front, two arms trailing 45° back, spacing_m apart along each arm.

    In every shape the slots are numbered from right to left (slot 0 is the right-most). So when a pack changes
    shape, drone k keeps roughly its place from right to left and mostly moves forward or back: fewer drones have
    to fly across the pack. (Our first numbering went round the hexagon, and a hexagon -> line change sent drones
    straight through each other.)
    """
    offsets, _ = _formation(shape, group_size, spacing_m)
    return offsets[_right_to_left(offsets)]


def formation_links(shape: str, group_size: int) -> torch.Tensor:
    """(G,) for drawing: slot k is linked to slot links[k] (itself = no link). Ring for the hexagon, a chain for the
    line, two arms for the V. Uses the same right-to-left slot numbers as `formation_offsets`."""
    offsets, links = _formation(shape, group_size, 1.0)
    order = _right_to_left(offsets)
    new_number = torch.argsort(order)  # raw slot -> its right-to-left number
    return new_number[links[order]]


def start_cells(group_size: int) -> torch.Tensor:
    """(C, 3) start points around a centre: a side × side grid START_CELL_M apart, in two layers 0.6 m apart, with
    side = ⌈√(G / 2)⌉ + 1 (3 × 3 × 2 = 18 points for 6 drones), so every pack gets a fresh random mix. The top layer
    is shifted half a cell sideways, so no drone starts right under another one's downwash."""
    side = math.ceil(math.sqrt(group_size / 2)) + 1
    ticks = (torch.arange(side, dtype=torch.float32) - (side - 1) / 2) * START_CELL_M
    x, y = torch.meshgrid(ticks, ticks, indexing="ij")
    layer = torch.stack((x.flatten(), y.flatten(), torch.zeros(side * side)), dim=1)
    low = layer + torch.tensor([-START_CELL_M / 4, -START_CELL_M / 4, -0.3])
    high = layer + torch.tensor([START_CELL_M / 4, START_CELL_M / 4, 0.3])
    return torch.cat((low, high))


class PackEnv(HoverEnv):
    """P5: worlds of G drones flying a formation through random waypoints. See the top of this file.

    group_size: drones per world (n_envs is rounded up to a multiple: check env.n_envs).
    spacing_m: between neighbouring slots. formations: comma-separated shapes to pick from ("hexagon,line,vee").
    max_turn_deg: largest turn of the formation at a new waypoint. reshape_chance: chance of a new shape there.
    neighbours: k, how many pack-mates each drone sees (default 3). share_goals: also see where they are heading.
    start: "ordered" (launch pad in slot order) or "scattered" (random order, many more crossings).
    downwash: drones above push drones below (physics/aero.py -> downwash_grouped).
    action_mode: "attitude" by default, the real-ready setpoints (base.py).
    """

    name = "pack"
    alive_bonus = 3.0  # covers the distance term (up to 2) and the crowding term (up to 1): reward stays ≥ 0
    distance_weight = 0.5
    distance_cap_m = 4.0
    precision_bonus = 0.5
    inside_bonus = 0.5
    crowding_weight = 1.0

    def __init__(
        self,
        n_envs: int,
        episode_s: float = 12.0,
        device: str | None = None,
        seed: int = 0,
        group_size: int = GROUP_SIZE,
        spacing_m: float = SPACING_M,
        formations: str = ",".join(FORMATIONS),
        max_turn_deg: float = MAX_TURN_DEG,
        reshape_chance: float = RESHAPE_CHANCE,
        neighbours: int | None = None,
        share_goals: bool = True,
        start: str = "ordered",
        downwash: bool = True,
        action_mode: str = "attitude",
        **options,
    ):
        group_size = int(group_size)  # the command line may hand us 6.0
        if group_size < 2:
            raise ValueError(f"group_size must be at least 2, not {group_size}")
        n_envs = -(-n_envs // group_size) * group_size  # round up to whole worlds: 4096 -> 4098, 1000 -> 1002
        self.group_size = group_size
        self.n_worlds = n_envs // group_size
        self.neighbours = min(NEIGHBOURS, group_size - 1) if neighbours is None else int(neighbours)
        if not 1 <= self.neighbours <= group_size - 1:
            raise ValueError(f"neighbours must be between 1 and group_size − 1 = {group_size - 1}")
        self.share_goals = bool(share_goals)
        self.task_obs_size = 3 + 6 * self.neighbours + 1 + (3 * self.neighbours if self.share_goals else 0)
        options.setdefault("success_radius_m", SUCCESS_RADIUS_M)
        super().__init__(n_envs, episode_s, device, seed, action_mode=action_mode, **options)

        self.spacing_m = float(spacing_m)
        self.max_turn_rad = math.radians(float(max_turn_deg))
        self.reshape_chance = float(reshape_chance)
        if start not in ("ordered", "scattered"):
            raise ValueError(f"start must be 'ordered' or 'scattered', not {start!r}")
        self.start = start
        self.formation_names = tuple(name.strip() for name in str(formations).split(",") if name.strip())
        self.slot_table = torch.stack(
            [formation_offsets(f, group_size, self.spacing_m) for f in self.formation_names]
        ).to(self.device)  # (S, G, 3)
        self.link_table = torch.stack([formation_links(f, group_size) for f in self.formation_names]).to(self.device)
        self.start_cells = start_cells(group_size).to(self.device)  # (C, 3)

        self.world_id = torch.arange(n_envs, device=self.device) // group_size  # (N,): the viewer groups by this
        self.shape = torch.zeros(self.n_worlds, dtype=torch.long, device=self.device)  # index into formation_names
        self.heading = torch.zeros(self.n_worlds, device=self.device)  # formation yaw (rad)
        self.centre = torch.zeros(self.n_worlds, 3, device=self.device)
        self.prev_pos = torch.zeros(n_envs, 3, device=self.device)  # where each drone was before the last step

        self.hold_steps = round(WAYPOINT_HOLD_S / 0.02)
        self.settle_steps = round(SETTLE_S / 0.02)
        self.waypoints_per_episode = max(1, self.max_steps // self.hold_steps)
        self.close_steps = torch.zeros(n_envs, dtype=torch.long, device=self.device)  # in a row within 15 cm
        self.waypoints_reached = torch.zeros(self.n_worlds, device=self.device)
        self.downwash_steps = torch.zeros(n_envs, device=self.device)

        self.downwash = bool(downwash)
        if self.downwash:  # worlds are consecutive blocks of G drones: the cheap W·G² downwash
            if self.variation is None:
                self.variation = DroneVariation()
            # Add to what the base class set up (real_world=True makes its own variation): keep it, switch on downwash.
            self.variation.world_id = self.world_id
            self.variation.downwash = True
            self.variation.group_size = group_size

    # ---------- shapes of things ----------

    def per_world(self, x: torch.Tensor) -> torch.Tensor:
        """(N, ...) -> (W, G, ...): the drones of each world in one row."""
        return x.reshape(self.n_worlds, self.group_size, *x.shape[1:])

    def world_drones(self, worlds: torch.Tensor) -> torch.Tensor:
        """Drone indices (len(worlds) · G,) of `worlds`, world by world."""
        return (worlds[:, None] * self.group_size + torch.arange(self.group_size, device=self.device)).flatten()

    def slot_positions(self, worlds: torch.Tensor) -> torch.Tensor:
        """(len(worlds), G, 3) world positions of the slots: centre + R_z(heading) · offset."""
        offsets = self.slot_table[self.shape[worlds]]  # (w, G, 3)
        cos, sin = torch.cos(self.heading[worlds])[:, None], torch.sin(self.heading[worlds])[:, None]
        turned = torch.stack(
            (cos * offsets[..., 0] - sin * offsets[..., 1], sin * offsets[..., 0] + cos * offsets[..., 1], offsets[..., 2]),
            dim=-1,
        )
        return self.centre[worlds, None, :] + turned

    def next_centres(self, worlds: torch.Tensor) -> torch.Tensor:
        """The next formation centre for `worlds`, (w, 3): 0.5–2 m from the last one in a random direction, then
        clamped into the box. Small steps like a real formation flight, not teleports across the room."""
        n = len(worlds)
        step = random_unit_vectors(n, self.generator, self.device) * self.uniform(n, *CENTRE_STEP_M)[:, None]
        centre = self.centre[worlds] + step
        low = torch.tensor([-CENTRE_XY_M, -CENTRE_XY_M, CENTRE_Z_M[0]], device=self.device)
        high = torch.tensor([CENTRE_XY_M, CENTRE_XY_M, CENTRE_Z_M[1]], device=self.device)
        return torch.maximum(torch.minimum(centre, high), low)

    def random_centres(self, n: int) -> torch.Tensor:
        centre = torch.empty(n, 3, device=self.device)
        centre[:, 0] = self.uniform(n, -CENTRE_XY_M, CENTRE_XY_M)
        centre[:, 1] = self.uniform(n, -CENTRE_XY_M, CENTRE_XY_M)
        centre[:, 2] = self.uniform(n, *CENTRE_Z_M)
        return centre

    def pairs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Everything about pack-mates, per world: relative position (W, G, G, 3) [w, i, j] = p_j − p_i (m), relative
        velocity (W, G, G, 3) = v_j − v_i (m/s), and distance (W, G, G) (m) with +inf on the diagonal (yourself)."""
        pos, vel = self.per_world(self.state[:, POS]), self.per_world(self.state[:, VEL])
        rel_pos = pos[:, None, :, :] - pos[:, :, None, :]
        rel_vel = vel[:, None, :, :] - vel[:, :, None, :]
        eye = torch.eye(self.group_size, dtype=torch.bool, device=self.device)
        distance = rel_pos.norm(dim=-1).masked_fill(eye, float("inf"))
        return rel_pos, rel_vel, distance

    def collisions(self) -> torch.Tensor:
        """(N,) bool: this drone came closer than 2 × collision radius to a pack-mate during the last step.

        Each drone is assumed to fly straight from where it was (prev_pos) to where it is now. For a pair, the gap
        goes r(s) = r0 + s · (r1 − r0), s from 0 to 1, and its closest point is at s* = clamp(−r0·Δ / |Δ|², 0, 1)
        with Δ = r1 − r0. Checking only the end points would miss two drones that swap places in one step."""
        start, end = self.per_world(self.prev_pos), self.per_world(self.state[:, POS])
        r0 = start[:, None, :, :] - start[:, :, None, :]  # (W, G, G, 3)
        delta = (end[:, None, :, :] - end[:, :, None, :]) - r0
        s = (-(r0 * delta).sum(-1) / (delta * delta).sum(-1).clamp_min(1e-12)).clamp(0.0, 1.0)
        closest = (r0 + s[..., None] * delta).norm(dim=-1)
        eye = torch.eye(self.group_size, dtype=torch.bool, device=self.device)
        hit = (closest < 2 * self.params.collision_radius_m) & ~eye
        return hit.any(dim=-1).flatten()

    # ---------- the hooks ----------

    def reset(self, stagger: bool = False) -> torch.Tensor:
        """Like base `reset`, but a staggered start gives each *world* (not each drone) its own clock, so a pack's
        drones always share one episode and one waypoint schedule."""
        obs = super().reset(stagger=False)
        if stagger:
            clock = torch.randint(0, self.max_steps, (self.n_worlds,), generator=self.generator, device=self.device)
            self.step_count = clock.repeat_interleave(self.group_size)
        return obs

    def _reset_task(self, ids: torch.Tensor) -> None:
        """New episode for the worlds of drones `ids` (always whole worlds: a world starts over together).

        The drones start on random start points near a random centre (tipped up to 15°, drifting up to 0.3 m/s).
        The first formation (random shape and heading) is centred right there, so the first 3 s are "get into
        formation": short trips, but some crossing paths.

        start="ordered" (default): like setting drones down on the launch pad in order, drone k gets the k-th
        start point from the right (as seen in the first formation's frame), the same order as its slot.
        start="scattered": drones get the start points in random order, so many more paths cross."""
        worlds = ids[ids % self.group_size == 0] // self.group_size
        n_w, g = len(worlds), self.group_size
        ids = self.world_drones(worlds)
        n = len(ids)
        choice = torch.randint(0, len(self.formation_names), (n_w,), generator=self.generator, device=self.device)
        self.shape[worlds] = choice
        self.heading[worlds] = self.uniform(n_w, -math.pi, math.pi)
        centre = self.random_centres(n_w)
        self.centre[worlds] = centre  # the first formation forms right where the pack starts

        cells = torch.rand(n_w, len(self.start_cells), generator=self.generator, device=self.device).argsort(dim=1)
        points = self.start_cells[cells[:, :g]]  # (n_w, G, 3) around the centre
        if self.start == "ordered":  # sort by "how far left" in the formation's frame: y' = −sin(h) x + cos(h) y
            heading = self.heading[worlds][:, None]
            x, y = points[..., 0], points[..., 1]
            left = -torch.sin(heading) * x + torch.cos(heading) * y
            forward = torch.cos(heading) * x + torch.sin(heading) * y
            order = torch.argsort(left - 1e-3 * forward, dim=1)
            points = points.gather(1, order[..., None].expand(-1, -1, 3))
        start = (centre[:, None, :] + points).reshape(n, 3) + self.uniform(n, -0.05, 0.05, size=3)
        self.state[ids, POS] = start
        self.state[ids, QUAT] = self.random_tilt(n, math.radians(15))
        self.state[ids, VEL] = self.uniform(n, -0.3, 0.3, size=3)
        self.state[ids, OMEGA] = self.uniform(n, -0.5, 0.5, size=3)
        self.prev_pos[ids] = start

        self.target[ids] = self.slot_positions(worlds).reshape(n, 3)
        self.close_steps[ids] = 0
        self.waypoints_reached[worlds] = 0.0
        self.downwash_steps[ids] = 0.0

    def physics(self, command: torch.Tensor) -> None:
        self.prev_pos = self.state[:, POS].clone()  # for the swept collision check
        super().physics(command)

    def _task_obs(self) -> torch.Tensor:
        """Own target − position (3), then the k nearest pack-mates' (relative position, relative velocity) (6 each),
        nearest first, then the distance to the nearest (1). See the table at the top of this file."""
        rel_pos, rel_vel, distance = self.pairs()
        k = self.neighbours
        order = distance.argsort(dim=-1)[..., :k]  # (W, G, k): yourself (inf) always sorts last
        index = order[..., None].expand(-1, -1, -1, 3)
        mates = torch.cat((rel_pos.gather(2, index), rel_vel.gather(2, index)), dim=-1)  # (W, G, k, 6)
        nearest = distance.gather(2, order[..., :1])  # (W, G, 1)
        columns = [self.target - self.state[:, POS], mates.reshape(self.n_envs, 6 * k), nearest.reshape(self.n_envs, 1)]
        if self.share_goals:  # where each of those pack-mates is heading: its target − its position, same order
            to_go = self.per_world(self.target - self.state[:, POS])  # (W, G, 3)
            to_go = to_go[:, None, :, :].expand(-1, self.group_size, -1, -1)  # [w, i, j] = drone j's
            columns.append(to_go.gather(2, index).reshape(self.n_envs, 3 * k))
        return torch.cat(columns, dim=1)

    def crowding(self) -> torch.Tensor:
        """(N,) in [0, 1]: how crowded each drone is, summed over its pack-mates and capped at 1:

            (1 − d / 0.5 m)²                     when closer than 50 cm (too close)
          + 0.5 · closing speed · (1 − d / 1 m)   when closer than 1 m and the gap is shrinking

        closing speed = −(Δp · Δv) / d in m/s: how fast the gap to that pack-mate is getting smaller. The second
        term warns early: at 2 m/s two drones are only 0.25 s apart at 50 cm, too late to swerve."""
        rel_pos, rel_vel, distance = self.pairs()
        close = (1.0 - distance / TOO_CLOSE_M).clamp_min(0.0) ** 2
        closing_mps = (-(rel_pos * rel_vel).sum(-1) / distance).clamp_min(0.0)  # 0 on the diagonal (d = inf)
        approach = APPROACH_WEIGHT * closing_mps * (1.0 - distance / APPROACH_RANGE_M).clamp_min(0.0)
        return (close + approach).sum(dim=-1).clamp(max=1.0).flatten()

    def _reward(self) -> torch.Tensor:
        distance = self.distance()
        upright = quat_to_matrix(self.state[:, QUAT])[:, 2, 2]  # R_zz = cos(tilt)
        spin = (self.state[:, OMEGA] ** 2).sum(dim=1)
        twitch = ((self.action - self.prev_action) ** 2).sum(dim=1)
        precision = self.precision_bonus * (1.0 - distance / 0.3).clamp_min(0.0)
        precision = precision + self.inside_bonus * (distance < self.success_radius_m).float()
        reward = (
            self.alive_bonus
            + precision
            - self.distance_weight * distance.clamp(max=self.distance_cap_m)
            - 0.2 * (1.0 - upright)
            - 0.02 * spin
            - 0.1 * twitch
            - self.crowding_weight * self.crowding()
        )
        return torch.where(self._crashed(), reward - CRASH_PENALTY, reward)

    def _out_of_bounds(self) -> torch.Tensor:
        pos = self.state[:, POS]
        return (pos[:, 0].abs() > ARENA_XY_M) | (pos[:, 1].abs() > ARENA_XY_M) | (pos[:, 2] > ARENA_TOP_M)

    def _crashed(self) -> torch.Tensor:
        """Hover's crashes (floor, upside down, spinning, out of the arena) or a collision with a pack-mate."""
        return super()._crashed() | self.collisions()

    def _cut_short(self) -> torch.Tensor:
        """Pack-mates of a crashed drone: the world starts over, but it wasn't their fault (see base.py)."""
        return self.per_world(self._crashed()).any(dim=1).repeat_interleave(self.group_size)

    def _post_step(self) -> None:
        """Count steps each drone spends within 15 cm of its slot. When the formation's 3 s are up, the waypoint
        counts if the whole pack has settled; then the formation moves to a new centre and turns."""
        close = self.distance() < self.success_radius_m
        self.close_steps = torch.where(close, self.close_steps + 1, torch.zeros_like(self.close_steps))
        if self.downwash:
            push = -downwash_grouped(self.per_world(self.state[:, POS]), self.params)[..., 2].flatten()
            self.downwash_steps += (push > 0.05 * self.params.weight_n).float()

        clock = self.per_world(self.step_count)[:, 0]  # a world's drones share one clock
        switch = (clock % self.hold_steps == 0) & (clock > 0)
        settled = (self.per_world(self.close_steps) >= self.settle_steps).all(dim=1)
        self.waypoints_reached += (switch & settled).float()
        # A world whose time is up keeps its last waypoint, so the final distance is measured to it.
        worlds = (switch & (clock < self.max_steps)).nonzero(as_tuple=True)[0]
        if len(worlds) > 0:
            self.centre[worlds] = self.next_centres(worlds)
            reshape = worlds[self.uniform(len(worlds), 0.0, 1.0) < self.reshape_chance]
            self.shape[reshape] = torch.randint(
                0, len(self.formation_names), (len(reshape),), generator=self.generator, device=self.device
            )
            turn = self.uniform(len(worlds), -self.max_turn_rad, self.max_turn_rad)
            self.heading[worlds] = torch.remainder(self.heading[worlds] + turn + math.pi, 2 * math.pi) - math.pi
            ids = self.world_drones(worlds)
            self.target[ids] = self.slot_positions(worlds).reshape(-1, 3)
            self.close_steps[ids] = 0

    def _episode_info(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        worlds = ids // self.group_size
        collided = self.collisions()
        return {
            "success": self.waypoints_reached[worlds] / self.waypoints_per_episode,
            "collided": collided[ids].float(),
            "pack_collided": self.per_world(collided).any(dim=1)[worlds].float(),
            "final_distance_m": self.distance()[ids],
            "downwash_time": self.downwash_steps[ids] / self.step_count[ids].clamp_min(1).float(),
        }

    def display_objects(self) -> dict[str, torch.Tensor]:
        """Slots as targets, and lines linking the slots into the formation's outline (ring, chain or V)."""
        links = self.link_table[self.shape].flatten()  # (N,): which slot of the same world each slot links to
        partner = self.world_id * self.group_size + links
        lines = torch.cat((self.target, self.target[partner]), dim=1)[:, None, :]  # (N, 1, 6); self-link = skipped
        return {"targets": self.target, "lines": lines, "lines_colors": ["#5fd3f3"]}
