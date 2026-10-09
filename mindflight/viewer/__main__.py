"""Start the viewer: `python -m mindflight.viewer --fake --n 64`, then open the printed URL."""

import argparse

import uvicorn

from mindflight.viewer.server import FakeSwarm, create_app

HOST = "127.0.0.1"  # this machine only: the viewer is never reachable from the network


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m mindflight.viewer", description=__doc__)
    parser.add_argument("--fake", action="store_true", help="stream a fake bobbing grid instead of the simulator")
    parser.add_argument("--n", type=int, default=64, help="number of drones (default 64)")
    parser.add_argument("--port", type=int, default=8000, help="port on 127.0.0.1 (default 8000)")
    args = parser.parse_args()

    if not args.fake:
        parser.error("only --fake works for now; the real simulator is wired in by issue #7")
    if args.n < 1:
        parser.error("--n must be at least 1")

    app = create_app(FakeSwarm(args.n))
    print(f"MindFlight viewer: http://{HOST}:{args.port}/  ({args.n} fake drones, Ctrl+C to stop)")
    uvicorn.run(app, host=HOST, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
