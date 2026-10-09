"""Live demo: the real physics in the 3D viewer. Run `python -m mindflight.demo --n 64`, open the printed URL.

The drones are split into 4 groups, each in its own corner of the floor. Every drone starts level at 1 m with its
motors already at their group's speed. Each group gets a different, fixed motor command:

    group    motors (m0, m1, m2, m3)           you should see
    hover    all at hover RPM                  stays put at 1 m
    off      all 0                             falls and stops on the floor
    tip      left motors m2, m3 +2 %           rolls right, drifts toward −y, then crashes
    spin     CW m1, m3 +2 %, CCW m0, m2 −2 %   spins in place (yaw), faster and faster

Why, in one line each (more in docs/level0.md):
- hover: 4 · kf · hover_rpm² = m · g, so lift equals weight and nothing moves.
- off: no lift, only gravity: z = 1 − ½ · g · t², the floor at 0.45 s.
- tip: more lift on the left (+y) makes roll torque τx = d · (−F0 − F1 + F2 + F3) > 0; the drone leans right
  and its thrust pushes it toward −y. Nothing levels it again, so it rolls over and falls.
- spin: CW props are faster, so yaw torque τz = −Q0 + Q1 − Q2 + Q3 > 0 turns the body counter-clockwise. The
  roll and pitch torques cancel, so it stays level. There is no air drag in the model yet, so the spin keeps
  speeding up (about 12 rad/s² for the Crazyflie).

Crash rule (demo only, not physics): a drone that touches the floor (z = 0) has crashed. Its props stop and its
spin stops, so it lies still. Without this a crashed "tip" drone would keep tumbling on the floor with its
motors still running, because the physics has no ground contact forces yet.

The sim starts over every `--loop-s` seconds (3 s by default), or when you press R in the page.
"""

import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch
import uvicorn

from mindflight.physics.dynamics import OMEGA, RPM, make_state, step
from mindflight.physics.params import CRAZYFLIE, CrazyflieParams
from mindflight.viewer.__main__ import HOST
from mindflight.viewer.server import create_app
from mindflight.viewer.sim_source import SimSource

START_HEIGHT_M = 1.0

# The spin group runs CW motors at 1.02 × hover and CCW motors at 0.98 × hover. Lift goes with rpm², and
# 2 · 1.02² + 2 · 0.98² = 4.0016, a hair more than 4, so it would slowly climb (about 2 mm in the first second).
# Scaling all four by this trim (0.9998) brings the lift back to exactly the weight, so it truly spins in place.
SPIN_TRIM = math.sqrt(4 / (2 * 1.02**2 + 2 * 0.98**2))


@dataclass(frozen=True)
class DemoGroup:
    """One group of drones that all get the same motor command."""

    name: str
    motors: str  # what we do to the motors, in words (for the legend)
    expect: str  # what you should see (for the legend)
    rpm_scale: tuple[float, float, float, float]  # each motor's command as a multiple of hover RPM, m0..m3
    corner: tuple[int, int]  # which corner of the floor the group sits in: (sign of x, sign of y)


GROUPS = (
    DemoGroup("hover", "all at hover RPM", "stays put at 1 m", (1.0, 1.0, 1.0, 1.0), (+1, +1)),
    DemoGroup("off", "motors off", "falls and stops on the floor", (0.0, 0.0, 0.0, 0.0), (+1, -1)),
    # The tip group sits on the −y side so it drifts away from the others, not into them.
    DemoGroup("tip", "left m2, m3 +2 %", "tips right, drifts toward -y", (1.0, 1.0, 1.02, 1.02), (-1, -1)),
    DemoGroup(
        "spin",
        "CW m1, m3 +2 %, CCW m0, m2 -2 %",
        "spins in place (yaw)",
        (0.98 * SPIN_TRIM, 1.02 * SPIN_TRIM, 0.98 * SPIN_TRIM, 1.02 * SPIN_TRIM),
        (-1, +1),
    ),
)


class FourGroupDemo:
    """The 4 groups above: where each drone starts, its motor command, and one physics step for all of them.

    Hand `start_state` and `step` to `SimSource` to watch it live.
    """

    def __init__(
        self,
        n_drones: int,
        device: torch.device | str = "cpu",
        spacing_m: float = 0.5,
        gap_m: float = 1.0,
        start_height_m: float = START_HEIGHT_M,
        params: CrazyflieParams = CRAZYFLIE,
    ):
        """
        n_drones: total, split as evenly as possible over the 4 groups (in order: drones 0.. are "hover", ...).
        spacing_m: distance between drones inside a group. gap_m: empty space between two groups.
        """
        self.n_drones = n_drones
        self.device = torch.device(device)
        self.params = params

        # Drone i belongs to group group_of_drone[i]; each group is one block of consecutive drone numbers.
        blocks = np.array_split(np.arange(n_drones), len(GROUPS))
        self.group_of_drone = np.concatenate([np.full(len(block), g) for g, block in enumerate(blocks)])
        self.group_first = [int(block[0]) if len(block) else 0 for block in blocks]
        self.group_count = [len(block) for block in blocks]

        # Each group is a small square grid, centred in its own corner of the floor.
        side = math.ceil(math.sqrt(max(self.group_count)))  # drones along one edge of a group
        half_width_m = (side - 1) / 2 * spacing_m
        corner_offset_m = half_width_m + gap_m / 2  # from the origin to a group's centre, along x and along y
        start_pos_m = np.zeros((n_drones, 3), dtype=np.float32)
        start_pos_m[:, 2] = start_height_m
        for group, block in zip(GROUPS, blocks):
            index_in_group = np.arange(len(block))
            row, col = index_in_group // side, index_in_group % side
            start_pos_m[block, 0] = group.corner[0] * corner_offset_m + (row * spacing_m - half_width_m)
            start_pos_m[block, 1] = group.corner[1] * corner_offset_m + (col * spacing_m - half_width_m)
        self.start_pos_m = torch.as_tensor(start_pos_m, device=self.device)

        # Motor command per drone in RPM, (N, 4): its group's scale × hover RPM.
        rpm_scale = torch.tensor([group.rpm_scale for group in GROUPS], dtype=torch.float32)
        self.group_rpm_cmd = (rpm_scale[torch.as_tensor(self.group_of_drone)] * params.hover_rpm).to(self.device)
        self.crashed = torch.zeros(n_drones, dtype=torch.bool, device=self.device)

    def start_state(self) -> torch.Tensor:
        """Every drone level at its start spot, at rest, with its motors already at the commanded speed."""
        self.crashed = torch.zeros(self.n_drones, dtype=torch.bool, device=self.device)
        return make_state(self.n_drones, self.device, pos=self.start_pos_m, rpm=self.group_rpm_cmd)

    def step(self, state: torch.Tensor) -> torch.Tensor:
        """One physics step (`dynamics.step`, 2 ms), then the demo's crash rule.

        Crash rule: the physics floor puts a drone that goes below z = 0 back at exactly z = 0 with zero velocity.
        We call that a crash: from then on its motor command is 0, its motor speeds are 0 and its angular velocity
        is 0, so it lies still where it landed (tilted, if it landed tilted).
        """
        rpm_cmd = torch.where(self.crashed[:, None], torch.zeros_like(self.group_rpm_cmd), self.group_rpm_cmd)
        new_state = step(state, rpm_cmd, params=self.params)
        self.crashed = self.crashed | (new_state[:, 2] <= 0.0)
        new_state[self.crashed, RPM.start : RPM.stop] = 0.0
        new_state[self.crashed, OMEGA.start : OMEGA.stop] = 0.0
        return new_state

    def config(self) -> dict:
        """The groups for the viewer's legend (`GET /config`): name, motors, what to expect, and which drones."""
        return {
            "groups": [
                {"name": g.name, "motors": g.motors, "expect": g.expect, "first": first, "count": count}
                for g, first, count in zip(GROUPS, self.group_first, self.group_count)
            ]
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m mindflight.demo",
        description="Real Crazyflie physics, live in the 3D viewer: 4 groups of drones with different motor speeds.",
    )
    parser.add_argument("--n", type=int, default=64, help="number of drones, split over 4 groups (default 64)")
    parser.add_argument("--port", type=int, default=8000, help="port on 127.0.0.1 (default 8000)")
    parser.add_argument(
        "--device", default="cpu", help="torch device (default cpu: faster than cuda below about 4,096 drones)"
    )
    parser.add_argument("--speed", type=float, default=1.0, help="sim seconds per real second (0.25 = slow motion)")
    parser.add_argument("--loop-s", type=float, default=3.0, help="start over after this many sim seconds (0 = never)")
    args = parser.parse_args()
    if args.n < len(GROUPS):
        parser.error(f"--n must be at least {len(GROUPS)} (one drone per group)")
    if args.speed <= 0:
        parser.error("--speed must be more than 0")

    demo = FourGroupDemo(args.n, device=args.device)
    restart_after_s = args.loop_s if args.loop_s > 0 else None
    source = SimSource(demo.start_state, demo.step, speed=args.speed, restart_after_s=restart_after_s)
    app = create_app(
        source, extra_config={**demo.config(), "restart_after_s": restart_after_s}, on_reset=source.reset
    )

    print(f"MindFlight live demo: http://{HOST}:{args.port}/  ({args.n} drones on {args.device}, Ctrl+C to stop)")
    for group, count in zip(GROUPS, demo.group_count):
        print(f"  {group.name:<6} {count:>5} drones  {group.motors:<32} -> {group.expect}")
    loop_text = f"every {args.loop_s:g} s of sim time" if restart_after_s else "only when you press R"
    print(f"  Starts over {loop_text}. Press R in the page to restart.")
    uvicorn.run(app, host=HOST, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
