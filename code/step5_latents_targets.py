"""Experiment 1, Step 5: pooled state set, latents, reward targets, and the reward-target checks. CPU.

Tolerances (fixed before running; committed with this file):
  CHECK 1 (time alignment): the observation fed to the encoder is the stored o_t; restoring the stored s_t and
          recomputing the observation must give o_t within 1e-5 (max abs), for every state.
  CHECK 2 (logged value): for t >= 1, r_source(s_t) must equal the 2nd substep reward of step t-1
          (= last substep component of r_logged_{t-1}) within 1e-6 (max abs). t = 0 is excluded.
  CHECK 3 (no side effect): after computing the five task rewards for a state, every task env's physics state
          must equal exactly (np.array_equal) the restored s_t, i.e. nothing advanced time.
  If any check fails, the script writes the failure and does not write the target file used by Step 6.

Outputs (data/pooled/): index.npz (episode key, task, type, ep, t, split-independent), obs.npy (N,24),
  state.npy (N,S), targets.npy (N,5) = r_tau'(s_t) for tau' in WALKER_TASKS order,
  latent_learned.npy (5,N,768), latent_random.npy (5,N,768); results/exp1/step5_checks.json, target_distribution.json.
  python step5_latents_targets.py --data ../data/rollouts --out ../data/pooled
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from collect_rollouts import raw_dm_env
from common_setup import ROOT, WALKER_TASKS, load_world_model, make_env_and_cfg

TOL_OBS, TOL_REW = 1e-5, 1e-6
RANDOM_INIT_SEED = 20261028
TYPES = ["A", "B", "C"]


def load_pool(data):
    files = []
    for task in WALKER_TASKS:
        for ty in TYPES:
            files += sorted(glob.glob(os.path.join(data, task, ty, "ep*.npz")))
    obs, state, sub_r1, keys = [], [], [], []
    meta_all = []
    for f in files:
        m = json.load(open(f[:-4] + ".json"))
        d = np.load(f)
        T = d["obs"].shape[0]
        obs.append(d["obs"]); state.append(d["state"]); sub_r1.append(d["sub_reward"][:, 1])
        keys.append(np.array([[WALKER_TASKS.index(m["task"]), TYPES.index(m["rollout_type"]), m["episode"], t] for t in range(T)]))
        meta_all.append({k: m[k] for k in ("task", "rollout_type", "episode", "episode_seed", "return", "device")})
    return np.concatenate(obs), np.concatenate(state), np.concatenate(sub_r1), np.concatenate(keys), meta_all


@torch.no_grad()
def encode_all(model, obs, task_ids, out_path, batch=4096):
    N = obs.shape[0]
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(len(task_ids), N, model.cfg.latent_dim))
    x = torch.from_numpy(obs)
    for i, k in enumerate(task_ids):
        for s in range(0, N, batch):
            xb = x[s:s + batch]
            mm[i, s:s + batch] = model.encode(xb, torch.full((xb.shape[0],), k, dtype=torch.long)).numpy()  # official encode
    mm.flush()
    return mm


def describe(v):
    return {"n": int(v.size), "mean": float(v.mean()), "var": float(v.var()), "q05": float(np.quantile(v, 0.05)),
            "q50": float(np.quantile(v, 0.5)), "q95": float(np.quantile(v, 0.95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res_dir = os.path.join(ROOT, "results", "exp1")
    os.makedirs(res_dir, exist_ok=True)
    obs, state, sub_r1, keys, meta = load_pool(a.data)
    N = obs.shape[0]
    print("pooled states", N, "episodes", len(meta), flush=True)

    env, cfg = make_env_and_cfg(48)
    ids = [cfg.tasks.index(t) for t in WALKER_TASKS]
    envs = [raw_dm_env(env, k) for k in ids]

    targets = np.zeros((N, 5), dtype=np.float64)
    max_obs_err, max_side, rew2 = 0.0, True, []
    for n in range(N):
        s = state[n]
        for j, dm in enumerate(envs):
            dm.physics.set_state(s)
            dm.physics.forward()
            if j == keys[n, 0]:  # CHECK 1 in the source task's env
                o = np.concatenate([v.flatten() for v in dm.task.get_observation(dm.physics).values()]).astype(np.float32)
                max_obs_err = max(max_obs_err, float(np.abs(o - obs[n]).max()))
            targets[n, j] = dm.task.get_reward(dm.physics)
            if not np.array_equal(dm.physics.get_state(), s):  # CHECK 3
                max_side = False
        if keys[n, 3] >= 1:  # CHECK 2: r_source(s_t) vs 2nd substep reward of step t-1
            rew2.append(abs(targets[n, keys[n, 0]] - sub_r1[n - 1]))
        if n % 20000 == 0:
            print("targets", n, flush=True)
    checks = {
        "n_states": int(N), "n_episodes": len(meta),
        "check1_time_alignment": {"max_abs_obs_recompute": max_obs_err, "tol": TOL_OBS, "pass": max_obs_err <= TOL_OBS,
                                  "code": "encoder input = stored obs[n]; reward = get_reward after set_state(state[n]); both index n (= t of the same episode)"},
        "check2_logged_value": {"n_compared": len(rew2), "max_abs": float(max(rew2)), "tol": TOL_REW, "pass": float(max(rew2)) <= TOL_REW,
                                "excluded": "t = 0 of every episode"},
        "check3_no_side_effect": {"pass": bool(max_side), "how": "np.array_equal(get_state() after reward, restored s_t) for all 5 envs and all states"},
    }
    all_pass = all(checks[k]["pass"] for k in checks if k.startswith("check"))
    checks["all_pass"] = all_pass
    json.dump(checks, open(os.path.join(res_dir, "step5_checks.json"), "w"), indent=1)
    print(json.dumps(checks, indent=1), flush=True)

    # target distributions (CHECK 4, descriptive)
    dist = {}
    for j, t in enumerate(WALKER_TASKS):
        dist[t] = {"pooled": describe(targets[:, j])}
        for ti, ty in enumerate(TYPES):
            dist[t][ty] = describe(targets[keys[:, 1] == ti, j])
    json.dump(dist, open(os.path.join(res_dir, "target_distribution.json"), "w"), indent=1)

    np.savez(os.path.join(a.out, "index.npz"), task=keys[:, 0], rollout_type=keys[:, 1], episode=keys[:, 2], t=keys[:, 3],
             task_names=np.array(WALKER_TASKS), type_names=np.array(TYPES))
    np.save(os.path.join(a.out, "obs.npy"), obs)
    np.save(os.path.join(a.out, "state.npy"), state)
    if not all_pass:
        print("STEP 5 CHECKS FAILED: targets.npy not written", flush=True)
        return
    np.save(os.path.join(a.out, "targets.npy"), targets)

    torch.set_num_threads(os.cpu_count())
    learned = load_world_model(cfg)  # same weights as the GPU agent (sha256 checked at collection)
    encode_all(learned, obs, ids, os.path.join(a.out, "latent_learned.npy"))
    from common.world_model import WorldModel

    torch.manual_seed(RANDOM_INIT_SEED)
    rnd = WorldModel(cfg).eval()  # same architecture, random init, its own random task-embedding table
    encode_all(rnd, obs, ids, os.path.join(a.out, "latent_random.npy"))
    json.dump({"random_init_seed": RANDOM_INIT_SEED, "random_task_emb_norms": [float(rnd._task_emb.weight[k].norm()) for k in ids],
               "learned_task_ids": ids, "latent_dim": cfg.latent_dim, "encode": "WorldModel.encode (world_model.py:105-114), CPU float32"},
              open(os.path.join(res_dir, "latent_meta.json"), "w"), indent=1)
    print("latents written", flush=True)


if __name__ == "__main__":
    main()
