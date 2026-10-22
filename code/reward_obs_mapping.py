"""How the walker reward inputs relate to the 24-d observation (random actions, 2000 control steps, seed 0).
Writes ../results/reward_obs_mapping.json.
"""
import json
import os

import numpy as np

from common_setup import ROOT, make_env_and_cfg

env, cfg = make_env_and_cfg(5)
e = env.envs[cfg.tasks.index("walker-walk")]
while not (hasattr(e, "physics") and hasattr(e, "task")):
    e = e.env
p = e.physics
e.reset()
rng = np.random.default_rng(0)
rows = []
for k in range(2000):
    ts = e.step(rng.uniform(-1, 1, 6))
    if ts.last():
        e.reset()
    ob = e.task.get_observation(p)
    rows.append((p.torso_upright(), ob["orientations"][0], p.torso_height(), float(ob["height"]), p.horizontal_velocity(), ob["velocity"][1]))
r = np.array(rows, dtype=float)
res = {
    "n_steps": len(r),
    "max_abs(torso_upright - orientations[0])": float(np.abs(r[:, 0] - r[:, 1]).max()),
    "max_abs(torso_height - height)": float(np.abs(r[:, 2] - r[:, 3]).max()),
    "corr(horizontal_velocity, velocity[1] (rootx))": float(np.corrcoef(r[:, 4], r[:, 5])[0, 1]),
    "max_abs(horizontal_velocity - velocity[1])": float(np.abs(r[:, 4] - r[:, 5]).max()),
    "qvel_names": list(p.named.data.qvel.axes.row.names),
    "orientation_bodies (xx, xz each)": list(p.named.data.xmat.axes.row.names)[1:],
}
json.dump(res, open(os.path.join(ROOT, "results", "reward_obs_mapping.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
