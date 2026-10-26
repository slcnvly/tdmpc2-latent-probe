"""DATA_SUMMARY numbers (Step 4): per condition counts, returns, body-state distributions, fallen ratio. CPU.
  python summarize_data.py --data ../data/rollouts --out ../results/exp1/data_summary.json
Fallen state (EXPERIMENT1_PREREG.md 2): torso_height < 0.6 m. Also reported: torso_height < 1.2 m.
"""
import argparse
import glob
import json
import os

import numpy as np

from collect_rollouts import raw_dm_env
from common_setup import WALKER_TASKS, make_env_and_cfg

TYPES = ["A", "B", "C"]
REQUIRED_FOR_Q = ["obs", "a_plan", "a_exec", "noise"]
REQUIRED_META = ["task_idx", "checkpoint_sha256", "versions", "episode_seed"]


def q(v):
    return {"mean": float(v.mean()), "std": float(v.std()), "q05": float(np.quantile(v, .05)), "q50": float(np.quantile(v, .5)), "q95": float(np.quantile(v, .95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    env, cfg = make_env_and_cfg(48)
    out, q_ok = {}, True
    for task in WALKER_TASKS:
        k = cfg.tasks.index(task)
        dm = raw_dm_env(env, k)
        for ty in TYPES:
            files = sorted(glob.glob(os.path.join(a.data, task, ty, "ep*.npz")))
            rets, h, up, v, devs = [], [], [], [], set()
            for f in files:
                m = json.load(open(f[:-4] + ".json"))
                d = np.load(f)
                q_ok &= all(x in d.files for x in REQUIRED_FOR_Q) and all(x in m for x in REQUIRED_META)
                rets.append(m["return"]); devs.add(m["device"])
                for s in d["state"]:
                    dm.physics.set_state(s); dm.physics.forward()
                    h.append(float(dm.physics.torso_height())); up.append(float(dm.physics.torso_upright())); v.append(float(dm.physics.horizontal_velocity()))
            h, up, v = map(np.array, (h, up, v))
            out[f"{task}/{ty}"] = {"episodes": len(files), "states": int(h.size), "episode_ids": [int(os.path.basename(f)[2:4]) for f in files],
                                   "return_mean": float(np.mean(rets)) if rets else None, "return_std": float(np.std(rets)) if rets else None,
                                   "torso_height_m": q(h) if h.size else None, "torso_upright": q(up) if up.size else None,
                                   "horizontal_velocity_mps": q(v) if v.size else None,
                                   "fallen_ratio_h_lt_0.6m": float((h < 0.6).mean()) if h.size else None,
                                   "ratio_h_lt_1.2m": float((h < 1.2).mean()) if h.size else None, "devices": sorted(devs)}
            print(task, ty, out[f"{task}/{ty}"]["episodes"], out[f"{task}/{ty}"]["return_mean"], flush=True)
    res = {"conditions": out, "q_recompute_info_present": bool(q_ok),
           "q_recompute_fields": {"per_step": REQUIRED_FOR_Q, "per_episode_meta": REQUIRED_META}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
