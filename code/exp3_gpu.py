"""EXPERIMENT3_AUDIT.md items 8-10 on a GPU (own path), mt80-48M, Meta-World tasks. No training, planning unchanged.

Own path = the loop of evaluate.py:44-87 for chosen task indices: parse_cfg, set_seed(cfg.seed) before make_env,
make_env (MultitaskWrapper over all 80 envs), TDMPC2 + load, agent.act(obs, t0=t==0, task=k),
ep_reward += reward (float32 tensor), success = info['success'] of the last step (evaluate.py:87).
common_setup.patch_timeout_setter() is applied first (official make_env fails without it at 332f19e).

  python exp3_gpu.py cost   --out cost.json
  python exp3_gpu.py trace  --rng-dir <official out dir> --out own_trace.json     # item 9
  python exp3_gpu.py sanity --episodes 5 --out sanity.json                         # item 10
"""
import argparse
import json
import os
import time

from common_setup import CKPT_DIR, make_cfg, patch_timeout_setter

MW = ["mw-reach", "mw-push", "mw-pick-place"]


def build_agent(seed=1):
    import torch

    torch.backends.cudnn.benchmark = True  # evaluate.py:17
    patch_timeout_setter()
    from common.seed import set_seed
    from envs import make_env
    from tdmpc2 import TDMPC2

    cfg = make_cfg(48, task="mt80", checkpoint=os.path.join(CKPT_DIR, "mt80-48M.pt"), seed=seed, save_video="false", enable_wandb="false")
    set_seed(cfg.seed)  # evaluate.py:45
    env = make_env(cfg)  # evaluate.py:54
    agent = TDMPC2(cfg)
    agent.load(cfg.checkpoint)
    torch.cuda.synchronize()
    return env, cfg, agent


def run_episode(env, agent, k, trace=None, trace_steps=0, step_times=None, np_state=None, torch_state=None):
    import numpy as np
    import torch

    if np_state is not None:
        np.random.set_state(np_state)
    obs, done, ep_reward, t = env.reset(task_idx=k), False, 0, 0
    if torch_state is not None:
        torch.set_rng_state(torch_state["cpu"])
        torch.cuda.set_rng_state(torch_state["cuda"])
    succ = []
    while not done:
        if trace is not None and t < trace_steps:
            with torch.no_grad():
                enc_in = agent.model.task_emb(obs.to(agent.device).unsqueeze(0), torch.tensor([k], device=agent.device))
            trace.append({"t": t, "task": k, "obs": obs.numpy().tolist(), "enc_in": enc_in[0].cpu().numpy().tolist()})
        t0 = time.perf_counter()
        action = agent.act(obs, t0=t == 0, task=k)
        torch.cuda.synchronize()
        if step_times is not None:
            step_times.append(time.perf_counter() - t0)
        if trace is not None and t < trace_steps:
            trace[-1]["action"] = action.numpy().tolist()
        obs, reward, done, info = env.step(action)
        ep_reward += reward  # evaluate.py:82
        succ.append(float(info["success"]))
        t += 1
    return float(ep_reward), t, succ


def cmd_cost(a):
    import torch

    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    env, cfg, agent = build_agent()
    build = time.perf_counter() - t
    out = {"build_and_load_s": build, "gpu": torch.cuda.get_device_name(0), "compile": cfg.compile, "iterations": cfg.iterations,
           "num_samples": cfg.num_samples, "horizon": cfg.horizon, "latent_dim": cfg.latent_dim, "task_dim": cfg.task_dim, "tasks": {}}
    for task in ("mw-reach", "mw-push"):
        k = cfg.tasks.index(task)
        times = []
        ret, n, s = run_episode(env, agent, k, step_times=times)
        warm = times[10:]
        out["tasks"][task] = {"steps": n, "first_step_s": times[0], "mean_step_s_after_10": sum(warm) / len(warm),
                              "median_step_s_after_10": sorted(warm)[len(warm) // 2], "return": ret}
        print(task, out["tasks"][task], flush=True)
    out["peak_gpu_mem_MB"] = torch.cuda.max_memory_allocated() / 2**20
    json.dump(out, open(a.out, "w"), indent=1)


def cmd_trace(a):
    import torch

    env, cfg, agent = build_agent()
    snaps = torch.load(os.path.join(a.rng_dir, "official_rng_states.pt"), weights_only=False)
    np_snaps = torch.load(os.path.join(a.rng_dir, "official_np_rng_states.pt"), weights_only=False)
    out = {"tasks": {}}
    for task in a.tasks:  # official order (cfg.tasks index ascending)
        k = cfg.tasks.index(task)
        tr = []
        ret, n, s = run_episode(env, agent, k, trace=tr, trace_steps=a.steps, np_state=np_snaps[k], torch_state=snaps[k])
        out["tasks"][task] = {"task_idx": k, "return": ret, "steps": n, "success_last": s[-1], "trace": tr}
        print(f"[own trace] {task} return {ret:.2f} steps {n} success {s[-1]}", flush=True)
    json.dump(out, open(a.out, "w"))


def cmd_sanity(a):
    env, cfg, agent = build_agent()
    out = {"episodes_per_task": a.episodes, "seed": 1, "tasks": {}}
    for task in MW:
        k = cfg.tasks.index(task)
        eps = []
        for i in range(a.episodes):
            t0 = time.perf_counter()
            ret, n, s = run_episode(env, agent, k)
            eps.append({"episode": i, "return": ret, "steps": n, "success_last_step": s[-1], "success_any_step": max(s),
                        "seconds": time.perf_counter() - t0})
            print(task, eps[-1], flush=True)
        out["tasks"][task] = {"task_idx": k, "episodes": eps, "mean_return": sum(e["return"] for e in eps) / len(eps),
                              "success_rate_last_step": sum(e["success_last_step"] for e in eps) / len(eps)}
        json.dump(out, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cost")
    c.add_argument("--out", required=True)
    t = sub.add_parser("trace")
    t.add_argument("--rng-dir", required=True)
    t.add_argument("--tasks", nargs="+", default=["mw-push", "mw-reach"])
    t.add_argument("--steps", type=int, default=50)
    t.add_argument("--out", required=True)
    s = sub.add_parser("sanity")
    s.add_argument("--episodes", type=int, default=5)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"cost": cmd_cost, "trace": cmd_trace, "sanity": cmd_sanity}[a.cmd](a)


if __name__ == "__main__":
    main()
