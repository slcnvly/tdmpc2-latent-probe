"""Run the official tdmpc2/evaluate.py UNMODIFIED (runpy), with recording hooks only (V2 / V3).

Hooks (values are copied, nothing is changed):
  - TDMPC2.act: for walker tasks, the first --trace-steps calls of each task's first episode:
    obs, encoder input (= model.task_emb(obs, task), the exact tensor encode() feeds the encoder,
    world_model.py:110-111), task index, returned action; and, at every task's first act (t0=True),
    the torch CPU and CUDA RNG states.
  - MultitaskWrapper.step: per-task logged rewards (r_logged_t) of every step.
Outputs go to --out-dir (absolute; Hydra changes the working directory).

  python official_eval_hooked.py --out-dir /abs/dir -- task=mt30 model_size=48 checkpoint=/abs/mt30-48M.pt eval_episodes=1 seed=1 save_video=false
"""
import atexit
import json
import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TDMPC2_DIR = os.path.join(os.path.dirname(HERE), "tdmpc2", "tdmpc2")
WALKER_IDS = {0, 1, 2, 19, 20}

args = sys.argv[1:]
sep = args.index("--")
own, hydra_args = args[:sep], args[sep + 1:]
out_dir = own[own.index("--out-dir") + 1]
trace_steps = int(own[own.index("--trace-steps") + 1]) if "--trace-steps" in own else 50
if "--trace-ids" in own:  # Experiment 3 audit: other task indices (default: the mt30 walker ids, as in GATE 2)
    WALKER_IDS = {int(x) for x in own[own.index("--trace-ids") + 1].split(",")}
PATCH_TIMEOUT = "--patch-timeout" in own  # Experiment 3 audit: see common_setup.patch_timeout_setter
os.makedirs(out_dir, exist_ok=True)
sys.path.insert(0, TDMPC2_DIR)
os.chdir(TDMPC2_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

if PATCH_TIMEOUT:
    sys.path.insert(0, HERE)
    from common_setup import patch_timeout_setter  # noqa: E402

    patch_timeout_setter()

import tdmpc2 as tdmpc2_module  # noqa: E402  (evaluate.py does `from tdmpc2 import TDMPC2`: same module object)
from envs.wrappers import multitask as mt_module  # noqa: E402

state = {"np_rng": {}, "rng": {}, "trace": {}, "step_count": {}, "first_episode_done": set(), "rewards": {}, "errors": [], "t_start": time.time()}
orig_act = tdmpc2_module.TDMPC2.act
orig_step = mt_module.MultitaskWrapper.step
orig_reset = mt_module.MultitaskWrapper.reset


def hooked_reset(self, task_idx=-1):
    try:  # global numpy RNG state before the first reset of a traced task (Meta-World resets draw from np.random)
        if int(task_idx) in WALKER_IDS and int(task_idx) not in state["np_rng"]:
            state["np_rng"][int(task_idx)] = np.random.get_state()
    except Exception as e:
        state["errors"].append(repr(e))
    return orig_reset(self, task_idx=task_idx)


def hooked_act(self, obs, t0=False, eval_mode=False, task=None):
    k = int(task) if task is not None else -1
    try:
        if t0:
            if k in state["step_count"]:
                state["first_episode_done"].add(k)
            else:
                state["rng"][k] = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}
                state["step_count"][k] = 0
        rec = k in WALKER_IDS and k not in state["first_episode_done"] and state["step_count"].get(k, 0) < trace_steps
        if rec:
            with torch.no_grad():
                enc_in = self.model.task_emb(obs.to(self.device).unsqueeze(0), torch.tensor([k], device=self.device))
            entry = {"t": state["step_count"][k], "task": k, "obs": obs.numpy().tolist(), "enc_in": enc_in[0].cpu().numpy().tolist()}
    except Exception as e:  # recording must never change the run
        state["errors"].append(repr(e))
        rec = False
    action = orig_act(self, obs, t0=t0, eval_mode=eval_mode, task=task)
    try:
        if rec:
            entry["action"] = action.numpy().tolist()
            state["trace"].setdefault(k, []).append(entry)
        if k in state["step_count"] and k not in state["first_episode_done"]:
            state["step_count"][k] += 1
    except Exception as e:
        state["errors"].append(repr(e))
    return action


def hooked_step(self, action):
    out = orig_step(self, action)
    try:
        state["rewards"].setdefault(int(self.task_idx), []).append(float(out[1]))
        if isinstance(out[3], dict) and "success" in out[3]:
            state.setdefault("success", {}).setdefault(int(self.task_idx), []).append(float(out[3]["success"]))
    except Exception as e:
        state["errors"].append(repr(e))
    return out


tdmpc2_module.TDMPC2.act = hooked_act
mt_module.MultitaskWrapper.step = hooked_step
mt_module.MultitaskWrapper.reset = hooked_reset


def dump():
    torch.save(state["rng"], os.path.join(out_dir, "official_rng_states.pt"))
    if state["np_rng"]:
        torch.save(state["np_rng"], os.path.join(out_dir, "official_np_rng_states.pt"))
    peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
    json.dump({"trace": {str(k): v for k, v in state["trace"].items()}, "rewards": {str(k): v for k, v in state["rewards"].items()},
               "success": {str(k): v for k, v in state.get("success", {}).items()},
               "errors": state["errors"], "peak_gpu_mem_MB_torch": peak, "wall_s": time.time() - state["t_start"],
               "hydra_args": hydra_args}, open(os.path.join(out_dir, "official_hooked.json"), "w"))


atexit.register(dump)
sys.argv = [os.path.join(TDMPC2_DIR, "evaluate.py")] + hydra_args
runpy.run_path(os.path.join(TDMPC2_DIR, "evaluate.py"), run_name="__main__")
