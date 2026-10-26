"""Integrity checks of saved rollouts (Step 4 procedure 1). CPU.

For every episode file (or --episodes subset):
  - restore s_t -> recomputed observation equals stored o_t (max |diff| <= 1e-5) and stored obs_raw
  - restore the 2nd substep end state (= s_{t+1}) -> r_source equals the stored 2nd substep reward (<= 1e-6)
  - r_logged_t equals the float32 sum of the two substep rewards (<= 1e-5)
  - sub_state[t, 1] equals state[t + 1] exactly; next_obs[t] equals obs[t + 1]
  python check_rollouts.py --data ../data/rollouts --episodes 0 --out ../results/exp1/check_ep0.json
"""
import argparse
import glob
import json
import os

import numpy as np

from collect_rollouts import parse_eps, raw_dm_env
from common_setup import WALKER_TASKS, make_env_and_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--episodes", default="0-19")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    env, cfg = make_env_and_cfg(48)
    eps = set(parse_eps(a.episodes))
    res, all_pass = {}, True
    for f in sorted(glob.glob(os.path.join(a.data, "*", "*", "ep*.npz"))):
        meta = json.load(open(f[:-4] + ".json"))
        if meta["episode"] not in eps:
            continue
        d = np.load(f)
        k = meta["task_idx"]
        dm = raw_dm_env(env, k)
        phys, task = dm.physics, dm.task
        T = d["state"].shape[0]
        obs_err = raw_err = rew_err = 0.0
        for t in range(T):
            phys.set_state(d["state"][t]); phys.forward()
            o = np.concatenate([v.flatten() for v in task.get_observation(phys).values()]).astype(np.float32)
            obs_err = max(obs_err, float(np.abs(o - d["obs"][t]).max()))
            raw_err = max(raw_err, float(np.abs(o - d["obs_raw"][t]).max()))
            phys.set_state(d["sub_state"][t, 1]); phys.forward()
            rew_err = max(rew_err, abs(float(task.get_reward(phys)) - float(d["sub_reward"][t, 1])))
        sum_err = float(np.abs(d["r_logged"] - d["sub_reward"].sum(1).astype(np.float32)).max())
        chain = bool(np.array_equal(d["sub_state"][:-1, 1], d["state"][1:]))
        nobs = bool(np.array_equal(d["next_obs"][:-1], d["obs"][1:]))
        ok = obs_err <= 1e-5 and raw_err <= 1e-5 and rew_err <= 1e-6 and sum_err <= 1e-5 and chain and nobs
        all_pass &= ok
        key = f"{meta['task']}/{meta['rollout_type']}/ep{meta['episode']:02d}"
        res[key] = {"pass": ok, "obs_restore_max_abs": obs_err, "obs_raw_max_abs": raw_err, "reward_restore_max_abs": rew_err,
                    "r_logged_vs_substep_sum_max_abs": sum_err, "substate_chain_equal": chain, "next_obs_chain_equal": nobs,
                    "return": meta["return"], "steps": T}
        print(key, "PASS" if ok else "FAIL", {kk: v for kk, v in res[key].items() if kk != "pass"}, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"all_pass": all_pass, "n_episodes": len(res), "episodes": res}, open(a.out, "w"), indent=1)
    print("ALL_PASS" if all_pass else "SOME_FAILED", len(res))


if __name__ == "__main__":
    main()
