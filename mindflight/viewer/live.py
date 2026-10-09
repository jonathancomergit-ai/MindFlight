"""Watch training live: `python -m mindflight.train --task hover --view`, then open the printed URL.

A small copy of the task (on the CPU) flies in the viewer with the policy as it is *right now*: after every training
iteration the newest weights are copied in. Charts in the corner show the learning curves as they grow.
"""

import copy
import threading

import torch
import uvicorn

from mindflight.envs.base import DroneEnv
from mindflight.rl.ppo import ActorCritic
from mindflight.viewer.__main__ import HOST
from mindflight.viewer.policy_source import SHOW, PolicySource, env_for_showing
from mindflight.viewer.server import create_app

CHARTS = [
    {"key": "episode_return", "label": "episode return (total reward)"},
    {"key": "episode_success", "label": "success rate (training)"},
    {"key": "eval_success", "label": "success rate (test, hardest)"},
    {"key": "episode_length_s", "label": "episode length (s)"},
    {"key": "difficulty", "label": "curriculum difficulty"},
]


class TrainingView:
    """Starts the viewer in a background thread; call it after each iteration as `train(..., on_iteration=view)`."""

    def __init__(self, task: str, env_kwargs: dict, model: ActorCritic, port: int = 8000):
        layout, n, spacing_m, round_s = SHOW.get(task, ("grid", 16, 4.0, None))
        self.env = env_for_showing(task, env_kwargs, n, "cpu")
        self.model = copy.deepcopy(model).to("cpu").eval()
        self.source = PolicySource(self.env, self.model, layout=layout, spacing_m=spacing_m, round_s=round_s)
        self.source.chart_series = [c for c in CHARTS if c["key"] != "difficulty" or self.env_has_curriculum()]
        self.source.info_lines = [f"training {task} · live: the newest policy, copied in after every iteration"]
        app = create_app(self.source, on_reset=self.source.reset)
        server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=port, log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        print(f"live view: http://{HOST}:{port}/")

    def env_has_curriculum(self) -> bool:
        return getattr(self.env, "has_curriculum", False)

    def __call__(self, env: DroneEnv, metrics: dict, model: ActorCritic) -> None:
        self.source.set_policy({k: v.detach().to("cpu") for k, v in model.state_dict().items()})
        if self.env_has_curriculum():
            self.env.difficulty = env.difficulty  # show the courses the learner is on now
        point = {"iteration": metrics["iteration"]}
        for chart in self.source.chart_series:
            if chart["key"] in metrics:
                point[chart["key"]] = round(float(metrics[chart["key"]]), 4)
        self.source.metric_points.append(point)
        line = f"iteration {metrics['iteration']} · {metrics['samples'] / 1e6:.1f} M samples"
        if "episode_success" in metrics:
            line += f" · success {100 * metrics['episode_success']:.0f}%"
        self.source.info_lines = self.source.info_lines[:1] + [line]
