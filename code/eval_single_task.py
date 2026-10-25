"""Single-task evaluation of a TD-MPC2 multi-task (mt30) checkpoint on chosen tasks.

Runs the loop of the official evaluate.py (tdmpc2/evaluate.py:44-86) for chosen task indices only:
same parse_cfg, set_seed(cfg.seed) BEFORE make_env, make_env (MultitaskWrapper over all 30 envs),
TDMPC2 agent + load, agent.act(obs, t0=t==0, task=task_idx), return accumulated as in evaluate.py:76-82.
No repo code is modified.

V1 changes (VERIFICATION_RESULTS.md V1): set_seed moved before make_env (evaluate.py:45 vs :54),
torch.backends.cudnn.benchmark = True (evaluate.py:17), return accumulated like evaluate.py:76,82
(`ep_reward = 0; ep_reward += reward` with reward a float32 tensor). The pre-V1 version is
eval_single_task_pre_v1.py.

Modes:
  cost   per model size: time per agent.act (planning included) and peak GPU memory
  eval   episodes on the walker tasks; optional per-step reward-component log; optional mpc=false
  trace  record the first N steps (obs, encoder input, task, action) of each task's first episode
"""
import argparse
import json
import os
import time

import numpy as np

from common_setup import WALKER_TASKS, make_cfg

STAND_HEIGHT = 1.2  # dm_control/suite/walker.py:33


def build_agent(size, seed, compile_, mpc=True):
    import torch

    torch.backends.cudnn.benchmark = True  # evaluate.py:17
    from common.seed import set_seed
    from envs import make_env
    from tdmpc2 import TDMPC2

    cfg = make_cfg(size, seed=seed, compile=str(compile_).lower(), save_video="false", enable_wandb="false", mpc=str(mpc).lower())
    set_seed(cfg.seed)  # evaluate.py:45
    env = make_env(cfg)  # evaluate.py:54
    agent = TDMPC2(cfg)  # evaluate.py:57
    agent.load(cfg.checkpoint)  # evaluate.py:59
    torch.cuda.synchronize()
    return env, cfg, agent


def raw_dm_env(env, task_idx):
    e = env.envs[task_idx]
    while not (hasattr(e, "physics") and hasattr(e, "task")):
        e = e.env
    return e


def reward_components(phys, task):
    """Reward components as in dm_control/suite/walker.py:139-154 and envs/tasks/walker.py:114-128."""
    from dm_control.utils import rewards

    h, up_raw, v = float(phys.torso_height()), float(phys.torso_upright()), float(phys.horizontal_velocity())
    standing = rewards.tolerance(h, bounds=(STAND_HEIGHT, float("inf")), margin=STAND_HEIGHT / 2)
    upright = (1 + up_raw) / 2
    stand_reward = (3 * standing + upright) / 4
    speed = task._move_speed
    backwards = type(task).__name__ == "BackwardsPlanarWalker"
    if speed == 0:
        move, total = None, stand_reward
    else:
        bounds = (-float("inf"), -speed) if backwards else (speed, float("inf"))
        move = rewards.tolerance(v, bounds=bounds, margin=speed / 2, value_at_margin=0.5, sigmoid="linear")
        total = stand_reward * (5 * move + 1) / 6
    assert abs(total - float(task.get_reward(phys))) < 1e-9, (total, task.get_reward(phys))
    return {"torso_height": h, "torso_upright": up_raw, "horizontal_velocity": v, "standing": float(standing),
            "upright": float(upright), "move": None if move is None else float(move), "r": float(total)}


def run_episode(env, agent, task_idx, max_steps=None, step_times=None, components=None, trace=None, trace_steps=0):
    """evaluate.py:76-86 for one episode (eval_mode is not passed there, so it stays False)."""
    import torch

    obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0
    dm = raw_dm_env(env, task_idx) if components is not None else None
    while not done:
        if trace is not None and t < trace_steps:
            with torch.no_grad():
                enc_in = agent.model.task_emb(obs.to(agent.device).unsqueeze(0), torch.tensor([task_idx], device=agent.device))
            trace.append({"t": t, "task": task_idx, "obs": obs.numpy().tolist(), "enc_in": enc_in[0].cpu().numpy().tolist()})
        t0 = time.perf_counter()
        action = agent.act(obs, t0=t == 0, task=task_idx)
        torch.cuda.synchronize()
        if step_times is not None:
            step_times.append(time.perf_counter() - t0)
        if trace is not None and t < trace_steps:
            trace[-1]["action"] = action.numpy().tolist()
        obs, reward, done, info = env.step(action)
        ep_reward += reward  # evaluate.py:82 (float32 tensor accumulation)
        if components is not None:
            c = reward_components(dm.physics, dm.task)
            c["r_logged"] = float(reward)
            components.append(c)
        t += 1
        if max_steps and t >= max_steps:
            break
    return float(ep_reward), t


def summarize_components(comp):
    a = lambda k: np.array([c[k] for c in comp], dtype=float)
    out = {k: float(a(k).mean()) for k in ("standing", "upright", "r", "torso_height", "torso_upright")}
    out["move"] = None if comp[0]["move"] is None else float(a("move").mean())
    v = a("horizontal_velocity")
    out["horizontal_velocity_mean_mps"], out["horizontal_velocity_std_mps"] = float(v.mean()), float(v.std())
    out["frac_steps_torso_height_below_1.2m"] = float((a("torso_height") < STAND_HEIGHT).mean())
    out["sum_r_logged"] = float(a("r_logged").sum())
    return out


def cmd_cost(a):
    import torch

    out = {}
    for size in a.sizes:
        for comp in (True, False):
            torch.cuda.reset_peak_memory_stats()
            t_build = time.perf_counter()
            env, cfg, agent = build_agent(size, a.seed, comp)
            build_s = time.perf_counter() - t_build
            k = cfg.tasks.index("walker-walk")
            times = []
            t_ep = time.perf_counter()
            _, n = run_episode(env, agent, k, max_steps=a.steps, step_times=times)
            total = time.perf_counter() - t_ep
            warm = times[10:]
            out[f"{size}M_compile{int(comp)}"] = {
                "model_size": size, "compile": comp, "steps": n, "build_and_load_s": round(build_s, 2),
                "first_step_s": round(times[0], 3), "mean_step_s_after_10": sum(warm) / len(warm),
                "median_step_s_after_10": sorted(warm)[len(warm) // 2], "total_s": round(total, 2),
                "peak_gpu_mem_MB": torch.cuda.max_memory_allocated() / 2**20, "latent_dim": cfg.latent_dim,
                "task_dim": cfg.task_dim, "iterations": cfg.iterations, "num_samples": cfg.num_samples, "horizon": cfg.horizon,
                "gpu": torch.cuda.get_device_name(0),
            }
            print(json.dumps(out[f"{size}M_compile{int(comp)}"]), flush=True)
            del agent
            torch.cuda.empty_cache()
            json.dump(out, open(a.out, "w"), indent=1)


def cmd_eval(a):
    import torch

    env, cfg, agent = build_agent(a.size, a.seed, True, mpc=a.mpc)
    res = {"model_size": a.size, "seed": a.seed, "episodes_per_task": a.episodes, "gpu": torch.cuda.get_device_name(0),
           "compile": True, "eval_mode": False, "mpc": a.mpc, "script_version": "post-V1", "tasks": {}}
    for task in WALKER_TASKS:
        k = cfg.tasks.index(task)
        eps = []
        for i in range(a.episodes):
            t0 = time.perf_counter()
            comp = [] if a.components else None
            ret, n = run_episode(env, agent, k, components=comp)
            e = {"episode": i, "return": ret, "steps": n, "seconds": round(time.perf_counter() - t0, 1)}
            if comp is not None:
                e["components"] = summarize_components(comp)
            eps.append(e)
            print(f"{task} ep{i} return {ret:.1f} steps {n} {e['seconds']}s", flush=True)
        res["tasks"][task] = {"task_idx": k, "episodes": eps, "mean_return": sum(e["return"] for e in eps) / len(eps)}
        json.dump(res, open(a.out, "w"), indent=1)
    res["peak_gpu_mem_MB"] = torch.cuda.max_memory_allocated() / 2**20
    json.dump(res, open(a.out, "w"), indent=1)


def cmd_trace(a):
    """Own-path trace for V2: tasks run in the given order (full first episodes, as evaluate.py does);
    before a task listed in --restore-rng-for, the torch CPU/CUDA RNG states saved by the official run
    at that task's first act are restored."""
    import torch

    env, cfg, agent = build_agent(a.size, a.seed, True)
    snaps = torch.load(a.rng_file, weights_only=False) if a.rng_file else {}
    out = {"tasks": {}, "order": a.tasks}
    for task in a.tasks:
        k = cfg.tasks.index(task)
        if task in a.restore_rng_for:
            torch.set_rng_state(snaps[k]["cpu"])
            torch.cuda.set_rng_state(snaps[k]["cuda"])
        tr = []
        ret, n = run_episode(env, agent, k, trace=tr, trace_steps=a.steps, max_steps=None if a.full else a.steps)
        out["tasks"][task] = {"task_idx": k, "return": ret, "steps": n, "trace": tr, "rng_restored": task in a.restore_rng_for}
        print(f"[own trace] {task} return {ret:.2f} steps {n}", flush=True)
        json.dump(out, open(a.out, "w"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cost")
    c.add_argument("--sizes", type=int, nargs="+", default=[1, 5, 19, 48, 317])
    c.add_argument("--steps", type=int, default=200)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--size", type=int, required=True)
    e.add_argument("--episodes", type=int, default=5)
    e.add_argument("--seed", type=int, default=1)
    e.add_argument("--components", action="store_true")
    e.add_argument("--mpc", type=lambda s: s.lower() == "true", default=True)
    e.add_argument("--out", required=True)
    t = sub.add_parser("trace")
    t.add_argument("--size", type=int, required=True)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--tasks", nargs="+", required=True)
    t.add_argument("--restore-rng-for", nargs="*", default=[])
    t.add_argument("--rng-file", default="")
    t.add_argument("--steps", type=int, default=50)
    t.add_argument("--full", action="store_true", help="finish each episode (keeps the RNG stream like evaluate.py)")
    t.add_argument("--out", required=True)
    a = ap.parse_args()
    {"cost": cmd_cost, "eval": cmd_eval, "trace": cmd_trace}[a.cmd](a)


if __name__ == "__main__":
    main()
