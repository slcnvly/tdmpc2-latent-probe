"""Experiment 1, Step 4: rollout collection (EXPERIMENT1_PREREG.md 1, EXPERIMENT_PLAN.md V6).

One episode = one file data/rollouts/<task>/<A|B|C>/ep<NN>.npz (+ .json metadata), written atomically.
Existing files are skipped. Seeds: episode_seed = 10000 + 1000*task_idx + 100*type_idx + ep.

  python collect_rollouts.py --types A B --episodes 0 --worker 0 --num-workers 2 --out ../data/rollouts   (GPU)
  python collect_rollouts.py --types C --episodes 0-19 --out ../data/rollouts                            (CPU, no planner)
"""
import argparse
import hashlib
import json
import os
import platform
import time

import numpy as np

from common_setup import WALKER_TASKS, make_cfg

TYPES = {"A": 0, "B": 1, "C": 2}
NOISE_STD = 0.3
CKPT_SHA256 = "bacb9ae1891a9b03dc3942a287080beb573220ea9d0263c5efd7a94a5c2ec419"
T = 500


def episode_seed(task_idx, type_name, ep):
    return 10000 + 1000 * task_idx + 100 * TYPES[type_name] + ep


def raw_dm_env(env, task_idx):
    e = env.envs[task_idx]
    while not (hasattr(e, "physics") and hasattr(e, "task")):
        e = e.env
    return e


def parse_eps(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def versions():
    import dm_control
    import mujoco
    import torch

    return {"torch": torch.__version__, "mujoco": mujoco.__version__, "dm_control": getattr(dm_control, "__version__", "1.0.16"),
            "numpy": np.__version__, "python": platform.python_version()}


def run_episode(env, agent, task_idx, type_name, ep, device_name):
    import torch

    seed = episode_seed(task_idx, type_name, ep)
    dm = raw_dm_env(env, task_idx)
    phys, task = dm.physics, dm.task
    torch.manual_seed(seed)  # planner RNG (CPU + all CUDA devices)
    task._random = np.random.RandomState(seed)  # environment initial state RNG
    act_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 1])))  # B noise / C actions

    # record each physics substep of the real env.step (DMControlWrapper calls dm.step twice, envs/dmcontrol.py:57-59)
    sub = []
    orig_step = dm.step

    def logged_step(action, _orig=orig_step):
        ts = _orig(action)
        sub.append((float(ts.reward), phys.get_state().copy()))
        return ts

    dm.step = logged_step
    try:
        obs = env.reset(task_idx=task_idx)
        S = phys.get_state().shape[0]
        rec = {k: np.zeros(shp, dtype=dt) for k, shp, dt in [
            ("state", (T, S), np.float64), ("sub_state", (T, 2, S), np.float64), ("obs", (T, 24), np.float32),
            ("obs_raw", (T, 24), np.float32), ("next_obs", (T, 24), np.float32), ("a_plan", (T, 6), np.float32),
            ("noise", (T, 6), np.float32), ("a_exec", (T, 6), np.float32), ("r_logged", (T,), np.float32),
            ("sub_reward", (T, 2), np.float64)]}
        rec["a_plan"][:] = np.nan
        rec["noise"][:] = np.nan
        ep_reward = 0
        t0 = time.time()
        for t in range(T):
            rec["state"][t] = phys.get_state()
            rec["obs"][t] = obs.numpy()
            rec["obs_raw"][t] = np.concatenate([v.flatten() for v in task.get_observation(phys).values()]).astype(np.float32)
            if type_name in ("A", "B"):
                a_plan = agent.act(obs, t0=t == 0, task=task_idx)  # evaluate.py:80 (eval_mode False)
                rec["a_plan"][t] = a_plan.numpy()
                if type_name == "B":
                    eps = act_rng.normal(0.0, NOISE_STD, size=6).astype(np.float32)
                    rec["noise"][t] = eps
                    a = torch.from_numpy(np.clip(a_plan.numpy() + eps, -1.0, 1.0).astype(np.float32))
                else:
                    a = a_plan
            else:
                a = torch.from_numpy(act_rng.uniform(-1.0, 1.0, size=6).astype(np.float32))
            rec["a_exec"][t] = a.numpy()
            sub.clear()
            obs, reward, done, info = env.step(a)
            assert len(sub) == 2
            ep_reward += reward  # evaluate.py:82
            rec["r_logged"][t] = float(reward)
            rec["sub_reward"][t] = [sub[0][0], sub[1][0]]
            rec["sub_state"][t, 0], rec["sub_state"][t, 1] = sub[0][1], sub[1][1]
            rec["next_obs"][t] = obs.numpy()
            if done:
                assert t == T - 1, t
        meta = {"task": WALKER_TASKS[[0, 1, 2, 19, 20].index(task_idx)], "task_idx": task_idx, "source_task": WALKER_TASKS[[0, 1, 2, 19, 20].index(task_idx)],
                "source_task_meaningful": type_name != "C", "rollout_type": type_name, "episode": ep, "episode_seed": seed,
                "noise_std": NOISE_STD if type_name == "B" else None, "return": float(ep_reward), "seconds": round(time.time() - t0, 1),
                "device": device_name, "state_size": int(S), "state_layout": "physics.get_state(): [time, qpos(9), qvel(9), act(0)]"}
    finally:
        dm.step = orig_step
    return rec, meta


def save(out_dir, rec, meta, common):
    d = os.path.join(out_dir, meta["task"], meta["rollout_type"])
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, f"ep{meta['episode']:02d}")
    np.savez_compressed(base + ".tmp.npz", **rec)
    json.dump({**meta, **common}, open(base + ".tmp.json", "w"), indent=1)
    os.replace(base + ".tmp.json", base + ".json")
    os.replace(base + ".tmp.npz", base + ".npz")  # npz last: its presence marks a finished episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=WALKER_TASKS)
    ap.add_argument("--episodes", default="0-19")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    need_planner = any(t in ("A", "B") for t in a.types)

    import torch

    cfg = make_cfg(48, seed=1, compile="true", save_video="false", enable_wandb="false")
    assert sha256(cfg.checkpoint) == CKPT_SHA256, "checkpoint sha256 differs from GATE 2"
    common = {"checkpoint": os.path.basename(cfg.checkpoint), "checkpoint_sha256": CKPT_SHA256, "versions": versions(),
              "tdmpc2_commit": "332f19e4aba4bffe2dc736f45ea7f7bdc585903a", "model_size": 48, "compile": True, "eval_mode": False}
    from common.seed import set_seed
    from envs import make_env

    agent = None
    if need_planner:
        torch.backends.cudnn.benchmark = True  # evaluate.py:17
        from tdmpc2 import TDMPC2

        set_seed(cfg.seed)
        env = make_env(cfg)
        agent = TDMPC2(cfg)
        agent.load(cfg.checkpoint)
        device_name = f"{torch.cuda.get_device_name(0)} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')})"
        obs = env.reset(task_idx=0)
        for i in range(3):  # warm-up: torch.compile / CUDA graph capture, results discarded (PREREG 1)
            agent.act(obs, t0=i == 0, task=0)
        torch.cuda.synchronize()
    else:
        set_seed(cfg.seed)
        env = make_env(cfg)
        device_name = "cpu (no planner)"
    items = [(ep, task, ty) for ep in parse_eps(a.episodes) for ty in a.types for task in a.tasks]
    items = [it for i, it in enumerate(items) if i % a.num_workers == a.worker]
    for ep, task, ty in items:
        k = cfg.tasks.index(task)
        if os.path.exists(os.path.join(a.out, task, ty, f"ep{ep:02d}.npz")):
            continue
        rec, meta = run_episode(env, agent, k, ty, ep, device_name)
        save(a.out, rec, meta, common)
        print(f"[w{a.worker}] {task} {ty} ep{ep:02d} seed {meta['episode_seed']} return {meta['return']:.1f} {meta['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
