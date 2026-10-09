"""Fly a trained brain: on the fake drone first, then on the real one.

    python -m mindflight.deploy --policy policies/waypoint_attitude.pt --sitl
    python -m mindflight.deploy --policy policies/waypoint_attitude.pt --sitl --waypoints 0.5,0,0.5 0.5,0.5,0.8
    python -m mindflight.deploy --policy policies/hover_attitude.pt --sitl --calibrate-hover
    python -m mindflight.deploy --policy policies/hover_attitude.pt --uri radio://0/80/2M/E7E7E7E7E7 --height 0.3

Press k (or Ctrl+C) at any time: motors off.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from mindflight.deploy.crazyflie import DEFAULT_URI, RealCrazyflie
from mindflight.deploy.interface import firmware_hover_command
from mindflight.deploy.policy import load_flight_policy
from mindflight.deploy.runner import Fence, FlightRunner, KeyboardKill, Mission, hover_command_from
from mindflight.deploy.sitl import SitlConfig, SitlCrazyflie
from mindflight.physics.params import load_drone

FIRST_FLIGHT_CHECK = """
REAL DRONE. Before it flies:
  [ ] props on the right motors, battery charged and strapped down
  [ ] 2 m of clear space all round, people behind you, soft floor or net
  [ ] this brain already flew this mission in --sitl
  [ ] your hand is on the keyboard: k = motors off
Type fly and press Enter to start (anything else stops): """


def parse_point(text: str) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in text.split(","))
    return (x, y, z)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m mindflight.deploy", description=__doc__.split("\n")[0])
    parser.add_argument("--policy", required=True, help="a trained attitude brain (.pt) or its ONNX file (.onnx)")
    parser.add_argument("--onnx", action="store_true", help="with a .pt: export to ONNX first and fly the ONNX file")
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--sitl", action="store_true", help="fly the simulated Crazyflie")
    where.add_argument("--uri", nargs="?", const=DEFAULT_URI, help=f"fly a real Crazyflie (default {DEFAULT_URI})")
    parser.add_argument("--drone", default=None, help="drone file (default: the one the brain trained on)")
    parser.add_argument("--hover-command", type=int, default=None, help="thrust integer that holds the drone up")
    parser.add_argument("--calibrate-hover", action="store_true", help="take off, hover, land; measure the hover command")
    parser.add_argument("--mission", default=None, help="mission .json (waypoints, fence, heights; see runner.Mission)")
    parser.add_argument("--waypoints", nargs="*", default=None, metavar="X,Y,Z", help="waypoints in m, e.g. 0.5,0,0.5")
    parser.add_argument("--height", type=float, default=None, help="takeoff height in m (default 0.5)")
    parser.add_argument("--hover-s", type=float, default=None, help="seconds to hover after takeoff (default 3)")
    parser.add_argument("--fence", nargs=5, type=float, default=None, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMAX"))
    parser.add_argument("--log-dir", default="runs/flights")
    parser.add_argument("--realtime", action="store_true", help="--sitl: run at real speed (default: as fast as it can)")
    parser.add_argument("--sitl-delay-ms", nargs=2, type=float, default=(5, 20), metavar=("MIN", "MAX"))
    parser.add_argument("--sitl-no-noise", action="store_true", help="--sitl: a perfect state estimate")
    parser.add_argument("--sitl-thrust-scale", type=float, default=1.0, help="--sitl: drone this much stronger/weaker")
    parser.add_argument("--yes", action="store_true", help="real drone: skip the typed confirmation")
    args = parser.parse_args()

    policy = load_flight_policy(args.policy, use_onnx=args.onnx)
    params = load_drone(args.drone if args.drone is not None else policy.env_kwargs.get("drone"))
    if args.hover_command is not None:
        params = replace(params, hover_thrust_command=args.hover_command)

    mission = Mission.from_json(args.mission) if args.mission else Mission()
    if args.waypoints is not None:
        mission.waypoints = [parse_point(p) for p in args.waypoints]
    if args.height is not None:
        mission.takeoff_height_m = args.height
    if args.hover_s is not None:
        mission.hover_s = args.hover_s
    if args.fence is not None:
        x0, x1, y0, y1, z1 = args.fence
        mission.fence = Fence((x0, x1), (y0, y1), z1)
    try:
        mission.check()
    except ValueError as error:
        parser.error(str(error))

    if args.sitl:
        lo, hi = args.sitl_delay_ms
        config = SitlConfig(uplink_delay_s=(lo / 1000, hi / 1000), downlink_delay_s=(lo / 1000, hi / 1000),
                            realtime=args.realtime, thrust_scale=args.sitl_thrust_scale)
        if args.sitl_no_noise:
            config.position_noise_m = config.velocity_noise_mps = config.attitude_noise_rad = 0.0
            config.gyro_noise_radps = config.battery_noise_v = 0.0
        link = SitlCrazyflie(params, config)
        print(f"SITL: fake Crazyflie, radio delay {lo:.0f}-{hi:.0f} ms each way")
    else:
        link = RealCrazyflie(args.uri)
        if not args.yes and input(FIRST_FLIGHT_CHECK).strip().lower() != "fly":
            print("not flying.")
            return

    print(f"brain: {policy.source}, task {policy.task}, hover command {params.hover_thrust_command}")
    print("press k (or Ctrl+C) at any time: motors off")
    if args.calibrate_hover:
        mission.waypoints = []  # just take off, hover still, land
        mission.hover_s = max(mission.hover_s, 6.0)
    with link:
        link.reset_estimator()
        runner = FlightRunner(link, policy, mission, params=params, log_dir=args.log_dir)
        with KeyboardKill(runner):
            result = runner.fly()
    if args.calibrate_hover:
        report_calibration(hover_command_from(result), result, params, Path(args.log_dir), link)


def report_calibration(hover, result, params, log_dir: Path, link) -> None:
    if hover is None:
        print(f"calibration failed: not enough steady hover ({result.outcome}: {result.reason})")
        return
    change = 100 * (hover / params.hover_thrust_command - 1)
    print(f"hover thrust command: {hover:.0f}  (was {params.hover_thrust_command}, {change:+.1f} %;"
          f" firmware line for {params.mass_kg * 1000:.0f} g: {firmware_hover_command(params):.0f})")
    if isinstance(link, SitlCrazyflie):
        print(f"(SITL truth: {link.true_hover_command:.0f})")
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"hover_calibration_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"hover_thrust_command": round(hover), "drone": params.name,
                                "samples": len(result.hover_thrust_commands), "flight_log": result.log_path}, indent=2))
    print(f"saved {path}. Use it with --hover-command {hover:.0f}, or put it in the drone file.")


if __name__ == "__main__":
    main()
