import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mindflight.hub import SHOWS, Built, HubSource, Show, available_shows, build, create_hub_app
from mindflight.viewer.server import FakeSwarm


class Tagged(FakeSwarm):
    """A fake show that says which one it is in an info line."""

    def __init__(self, name: str, n: int):
        super().__init__(n)
        self.name = name
        self.resets = 0
        self.speed = 1.0

    def state_messages(self) -> list[dict]:
        return [{"type": "info", "lines": [self.name]}]

    def reset(self) -> None:
        self.resets += 1


SHOW_A = Show("a", "group 1", "Show A", "the first one")
SHOW_B = Show("b", "group 2", "Show B", "the second one")


def fake_builder(show: Show) -> Built:
    source = Tagged(show.id, 4 if show.id == "a" else 9)
    return Built(source, on_reset=source.reset, groups=[{"name": show.id, "first": 0, "count": 1}])


def make_hub() -> HubSource:
    return HubSource([SHOW_A, SHOW_B], "a", builder=fake_builder)


def texts_until_info(websocket) -> list[dict]:
    """Read messages until an info message arrives; return the text ones."""
    messages = []
    while True:
        message = websocket.receive()
        if message.get("text"):
            messages.append(json.loads(message["text"]))
            if messages[-1]["type"] == "info":
                return messages


def test_hub_lists_shows_and_hands_frames_to_the_current_one():
    hub = make_hub()
    assert hub.n_drones == 4 and hub.frame(0.0).shape == (4, 7)
    message = hub.state_messages()[0]
    assert message["type"] == "hub" and message["current"] == "a" and message["generation"] == 0
    assert [s["id"] for s in message["shows"]] == ["a", "b"]
    assert message["groups"] == [{"name": "a", "first": 0, "count": 1}]


def test_switch_swaps_the_show_and_bumps_the_generation():
    hub = make_hub()
    hub.set_speed(0.5)
    hub.switch("b")
    assert hub.current_id == "b" and hub.generation == 1 and hub.n_drones == 9
    assert hub.current.source.speed == 0.5  # the speed carries over to the new show
    hub.reset()
    assert hub.current.source.resets == 1
    with pytest.raises(KeyError):
        hub.switch("nope")


def test_shows_you_leave_keep_flying_and_come_back_without_a_rebuild():
    built = []

    def builder(show):
        built.append(show.id)
        return fake_builder(show)

    hub = HubSource([SHOW_A, SHOW_B], "a", builder=builder)
    first_a = hub.current
    hub.switch("b")
    hub.switch("a")
    assert hub.current is first_a and built == ["a", "b"] and hub.generation == 2


def test_switching_stops_the_old_show():
    stopped = []

    def builder(show):
        built = fake_builder(show)
        built.stop = lambda: stopped.append(show.id)
        return built

    hub = HubSource([SHOW_A, SHOW_B], "a", builder=builder)
    hub.switch("b")
    assert stopped == ["a"]
    hub.switch("a")  # a stopped show (live training) is never kept: going back starts it again
    assert stopped == ["a", "b"] and hub.current_id == "a"


def test_a_failed_switch_keeps_the_old_show_and_reports_it():
    def builder(show):
        if show.id == "b":
            raise RuntimeError("no brain")
        return fake_builder(show)

    hub = HubSource([SHOW_A, SHOW_B], "a", builder=builder)
    with pytest.raises(RuntimeError):
        hub.switch("b")
    assert hub.current_id == "a" and hub.generation == 0 and "no brain" in hub.hub_message()["error"]


def test_routes_switch_and_set_speed_and_the_page_is_told_everything_again():
    hub = make_hub()
    client = TestClient(create_hub_app(hub))
    with client.websocket_connect("/ws") as websocket:
        first = texts_until_info(websocket)
        assert first[0]["type"] == "hub" and first[-1]["lines"] == ["a"]
        assert client.post("/show/b").json() == {"ok": True, "current": "b"}
        # After the switch the page gets 9 drones, the new hub message first, then the new show's own messages.
        while len(websocket.receive_bytes()) != 9 * 7 * 4:
            pass
        after = texts_until_info(websocket)
        assert after[0]["type"] == "hub" and after[0]["generation"] == 1 and after[-1]["lines"] == ["b"]
    assert client.post("/show/zzz").status_code == 404
    assert client.post("/speed", params={"value": 0.25}).json()["speed"] == 0.25
    assert client.post("/speed", params={"value": 3}).status_code == 400
    assert client.post("/reset").json() == {"ok": True}


def test_every_show_has_its_brain_and_a_unique_id():
    assert len({s.id for s in SHOWS}) == len(SHOWS)
    assert available_shows() == SHOWS  # all the brains are in policies/


@pytest.mark.parametrize("show_id", ["hover", "pack", "race", "physics"])
def test_real_shows_build_and_stream(show_id):
    built = build(next(s for s in SHOWS if s.id == show_id))
    poses = built.source.frame(0.0)
    assert poses.shape[1] == 7 and np.isfinite(poses).all()
    assert built.on_reset is not None


def test_fake_crazyflie_mission_records_a_flight_and_replays_it():
    built = build(next(s for s in SHOWS if s.id == "mission"))
    source = built.source
    assert source.duration_s > 10  # takeoff, hover, 3 waypoints, landing
    heights = source.poses[:, 2]
    assert heights.max() > 0.5 and heights[-1] < 0.1  # it flew, and it landed
    clock = [0.0]
    source._clock = lambda: clock[0]
    source.reset()
    clock[0] = 0.1
    assert source.frame(0.0).shape == (1, 7)
    info = source.state_messages()[1]["lines"]
    assert any("result: landed" in line for line in info)
