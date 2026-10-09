"""Run the Level 6 race without the viewer, as fast as the computer can: `python scripts/race_tournament.py 100`.

Flies N rounds of `mindflight.race` (the default entrants, 2 drones each, a new random course every round) and
prints the final leaderboard as a Markdown table.
"""

import sys

import torch

from mindflight.race import RaceSource, parse_entrants
from mindflight.rl.train import load_policy


def main(rounds: int = 100, downwash: bool = False, seed: int = 0) -> None:
    torch.manual_seed(seed)
    entrants = parse_entrants(None)
    models = [load_policy(e.path, "cpu")[0] for e in entrants]
    race = RaceSource(entrants, models, downwash=downwash)
    race.env.generator.manual_seed(seed)
    while race.round <= rounds:
        race.advance(1)
        if race.round == rounds and race.pause_s > 0:
            break
    print(f"After {rounds} rounds{' (with downwash)' if downwash else ''}:\n")
    print("| " + " | ".join(race.board["columns"][:-1]) + " |")
    print("|" + "---|" * (len(race.board["columns"]) - 1))
    for row in race.board["rows"]:
        print("| " + " | ".join(str(c).split("|")[-1] for c in row[:-1]) + " |")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100, downwash="--downwash" in sys.argv)
