"""Single-task evaluation of a TD-MPC2 multi-task (mt30) checkpoint on chosen tasks.

The official evaluate.py (tdmpc2/evaluate.py:61-99) evaluates a multi-task checkpoint on all tasks of the
set. This script runs the very same loop -- same parse_cfg, same make_env (MultitaskWrapper over all 30
envs), same TDMPC2 agent and agent.act(obs, t0, task) calls, same seeding (set_seed once) -- but only for
the requested task indices. No repo code is modified.

Two modes:
  cost   per model size: wall-clock per env step (planning included) and peak GPU memory
  eval   episodes on the walker tasks, per-episode return, per-step reward log

  python eval_single_task.py cost --sizes 1 5 19 48 317 --steps 200 --out ../results/cost.json
  python eval_single_task.py eval --size 48 --episodes 5 --seed 1 --out ../results/exp0_eval.json
"""
import argparse
import json
import os
import time

from common_setup import WALKER_TASKS, make_env_and_cfg


def build_agent(size, seed, compile_):
    import torch

    from common.seed import set_seed
    from tdmpc2 import TDMPC2

    env, cfg = make_env_and_cfg(size, seed=seed, compile=str(compile_).lower(), save_video="false", enable_wandb="false")
    set_seed(cfg.seed)  # evaluate.py:45
    agent = TDMPC2(cfg)
    agent.load(cfg.checkpoint)  # evaluate.py:59
    torch.cuda.synchronize()
    return env, cfg, agent


def run_episode(env, agent, task_idx, max_steps=None, step_times=None):
    """evaluate.py:76-86 for one episode (eval_mode is not passed there either, so it stays False)."""
    import torch

    obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0.0, 0
    rewards = []
    while not done:
        t0 = time.perf_counter()
        action = agent.act(obs, t0=t == 0, task=task_idx)
        torch.cuda.synchronize()
        if step_times is not None:
            step_times.append(time.perf_counter() - t0)
        obs, reward, done, info = env.step(action)
        ep_reward += float(reward)
        rewards.append(float(reward))
        t += 1
        if max_steps and t >= max_steps:
            break
    return ep_reward, t, rewards


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
            _, n, _ = run_episode(env, agent, k, max_steps=a.steps, step_times=times)
            total = time.perf_counter() - t_ep
            warm = times[10:]  # first calls include torch.compile / CUDA graph capture
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

    env, cfg, agent = build_agent(a.size, a.seed, True)
    res = {"model_size": a.size, "seed": a.seed, "episodes_per_task": a.episodes, "gpu": torch.cuda.get_device_name(0),
           "compile": True, "eval_mode": False, "tasks": {}}
    for task in WALKER_TASKS:
        k = cfg.tasks.index(task)
        eps = []
        for i in range(a.episodes):
            t0 = time.perf_counter()
            ret, n, rewards = run_episode(env, agent, k)
            eps.append({"episode": i, "return": ret, "steps": n, "seconds": round(time.perf_counter() - t0, 1)})
            print(f"{task} ep{i} return {ret:.1f} steps {n} {eps[-1]['seconds']}s", flush=True)
        res["tasks"][task] = {"task_idx": k, "episodes": eps, "mean_return": sum(e["return"] for e in eps) / len(eps)}
        json.dump(res, open(a.out, "w"), indent=1)
    res["peak_gpu_mem_MB"] = torch.cuda.max_memory_allocated() / 2**20
    json.dump(res, open(a.out, "w"), indent=1)


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
    e.add_argument("--out", required=True)
    a = ap.parse_args()
    {"cost": cmd_cost, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
