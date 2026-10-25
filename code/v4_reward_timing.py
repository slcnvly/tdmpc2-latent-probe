"""V4: reward timing. CPU. Writes ../results/v4_reward_timing.json.

Terms (VERIFICATION_PLAN.md): s_t = physics state at which o_t is computed; r_logged_t = reward returned by
env.step(a_t) (sum over the 2 action-repeat substeps, envs/dmcontrol.py:55-60); r_tau(s_t) = reward function of
task tau applied to restored s_t with no time integration (set_state + forward()).
"""
import json
import os

import numpy as np
import torch

from common_setup import ROOT, WALKER_TASKS, make_env_and_cfg
from eval_single_task import raw_dm_env

env, cfg = make_env_and_cfg(5)
res = {"tasks": {}}
N = 40
for task in WALKER_TASKS:
    k = cfg.tasks.index(task)
    obs = env.reset(task_idx=k)
    dm = raw_dm_env(env, k)
    phys, tk = dm.physics, dm.task
    rng = np.random.default_rng(0)
    rows = []
    # record each substep of the REAL env.step: wrap the dm_control-level step that DMControlWrapper calls twice
    sub_log = []
    orig = dm.step

    def logged_step(action, _orig=orig):
        ts = _orig(action)
        sub_log.append((float(ts.reward), phys.get_state().copy()))
        return ts

    dm.step = logged_step
    for t in range(N):
        a = rng.uniform(-1, 1, 6).astype(np.float32)
        s_t = phys.get_state().copy()
        # r_tau(s_t): restore s_t, no time integration (state is unchanged by this)
        phys.set_state(s_t); phys.forward()
        r_s_t = float(tk.get_reward(phys))
        sub_log.clear()
        obs_next, r_logged, done, info = env.step(torch.from_numpy(a))
        s_next = phys.get_state().copy()
        assert len(sub_log) == 2
        # r_tau(s_{t+1}): restore the last substep's end state (= s_{t+1}) and apply the reward function
        phys.set_state(sub_log[-1][1]); phys.forward()
        r_s_next = float(tk.get_reward(phys))
        phys.set_state(s_next); phys.forward()
        sub_r = [x[0] for x in sub_log]
        rows.append({"t": t, "r_logged_t": float(r_logged), "substep_rewards": sub_r, "r_tau(s_t)": r_s_t, "r_tau(s_t+1)": r_s_next,
                     "last_substep_state_equals_s_t+1": bool(np.array_equal(sub_log[-1][1], s_next)),
                     "abs(r_tau(s_t+1) - last_substep_reward)": abs(r_s_next - sub_r[1]),
                     "abs(r_logged - sum_substeps)": abs(float(r_logged) - sum(sub_r))})
    dm.step = orig
    # 4-4 numeric: change ctrl only, forward(), reward unchanged?
    s = phys.get_state().copy()
    phys.set_state(s); phys.forward()
    base = float(tk.get_reward(phys))
    ctrl_diffs = []
    for j in range(20):
        phys.data.ctrl[:] = rng.uniform(-1, 1, 6)
        phys.forward()
        ctrl_diffs.append(abs(float(tk.get_reward(phys)) - base))
    phys.data.ctrl[:] = 0
    res["tasks"][task] = {
        "task_idx": k, "n_steps": N,
        "max_abs(r_tau(s_t+1) - last_substep_reward)": max(r["abs(r_tau(s_t+1) - last_substep_reward)"] for r in rows),
        "all_last_substep_state_equals_s_t+1": all(r["last_substep_state_equals_s_t+1"] for r in rows),
        "max_abs(r_logged - sum_substeps)": max(r["abs(r_logged - sum_substeps)"] for r in rows),
        "max_abs(r_tau(s_t) - r_logged_t)": max(abs(r["r_tau(s_t)"] - r["r_logged_t"]) for r in rows),
        "mean_abs(r_tau(s_t) - r_logged_t)": float(np.mean([abs(r["r_tau(s_t)"] - r["r_logged_t"]) for r in rows])),
        "max_abs(r_tau(s) change when only ctrl changes)": max(ctrl_diffs),
        "examples": rows[:5] + rows[-2:],
        "get_reward_signature": "get_reward(self, physics)",
    }
    print(task, {kk: v for kk, v in res["tasks"][task].items() if kk not in ("examples",)})
json.dump(res, open(os.path.join(ROOT, "results", "v4_reward_timing.json"), "w"), indent=1)

# Extra record (not a V0 criterion): stepping again from a restored s_t is not guaranteed to reproduce the
# real next state bit-for-bit (MuJoCo keeps solver warm-start data outside get_state()). Size of the gap:
gaps = {}
for task in WALKER_TASKS:
    k = cfg.tasks.index(task)
    env.reset(task_idx=k)
    dm = raw_dm_env(env, k)
    phys = dm.physics
    rng = np.random.default_rng(1)
    mx = 0.0
    for t in range(40):
        a = rng.uniform(-1, 1, 6).astype(np.float32)
        s_t = phys.get_state().copy()
        env.step(torch.from_numpy(a))
        real = phys.get_state().copy()
        phys.set_state(s_t); phys.forward()
        for _ in range(2):
            dm.step(a.astype(np.float64))
        mx = max(mx, float(np.abs(phys.get_state() - real).max()))
        phys.set_state(real); phys.forward()
    gaps[task] = mx
res["replay_from_restored_s_t_max_abs_state_gap"] = gaps
print("replay gap", gaps)
json.dump(res, open(os.path.join(ROOT, "results", "v4_reward_timing.json"), "w"), indent=1)
