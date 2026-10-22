"""Kaggle GPU kernel for GATE 1: Step 1 item 9 (cost per model size) and Step 3 (Experiment 0).

Model-size rule (fixed in EXPERIMENT_PLAN.md before any measurement):
  planned Step 4 planning workload = 5 tasks x 2 planning rollout types x 20 episodes x 500 steps = 100,000 steps;
  pick the LARGEST mt30 size whose mean time per env step (official settings, compile=true, after warm-up)
  x 100,000 <= 5 GPU-hours (i.e. <= 0.18 s/step) and whose peak GPU memory <= 12 GB.
"""
import json
import os
import subprocess
import sys
import time

TEMP = "/kaggle/temp"
OUT = "/kaggle/working/out"
os.makedirs(OUT, exist_ok=True)
os.makedirs(TEMP, exist_ok=True)
T0 = time.time()


def sh(cmd, env=None, check=True):
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    rc = subprocess.run("set -o pipefail; " + cmd, shell=True, env=env, executable="/bin/bash").returncode
    print(f"[rc={rc} {time.time() - t:.1f}s]", flush=True)
    if check and rc:
        raise SystemExit(f"FAILED: {cmd}")
    return rc


base = dict(os.environ, UV_CACHE_DIR=f"{TEMP}/uvcache", UV_PYTHON_INSTALL_DIR=f"{TEMP}/uvpy", PYTHONUNBUFFERED="1")
sh("nvidia-smi; nproc; free -g | head -2", env=base, check=False)
if subprocess.run("which uv", shell=True).returncode:
    sh(f"curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR={TEMP}/uvbin UV_NO_MODIFY_PATH=1 sh", env=base)
    base["PATH"] = f"{TEMP}/uvbin:{TEMP}/uvbin/bin:" + base["PATH"]
ROOT = f"{TEMP}/tdmpc2-latent-probe"
sh(f"git clone -q https://github.com/slcnvly/tdmpc2-latent-probe {ROOT} && cd {ROOT} && git checkout -q {PROBE_COMMIT} && git log -1 --oneline", env=base)
sh(f"git clone -q https://github.com/nicklashansen/tdmpc2 {ROOT}/tdmpc2 && cd {ROOT}/tdmpc2 && git checkout -q {TDMPC2_COMMIT} && git log -1 --oneline", env=base)
sh(f"uv venv -q --python 3.11 {ROOT}/venv", env=base)
sh(f"uv pip install -q --python {ROOT}/venv/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128", env=base)
sh(f"cd {ROOT} && grep -v '^torch==' requirements.txt > /tmp/req.txt && uv pip install -q --python venv/bin/python -r /tmp/req.txt", env=base)
sh(f"uv pip freeze --python {ROOT}/venv/bin/python > {OUT}/pip_freeze.txt; grep -i -E '^(torch|dm-control|mujoco|tensordict|numpy|hydra-core)==' {OUT}/pip_freeze.txt", env=base)
for s in SIZES:
    sh(f"{ROOT}/venv/bin/hf download nicklashansen/tdmpc2 multitask/mt30-{s}M.pt --local-dir {ROOT}/ckpts > /dev/null && ls -la {ROOT}/ckpts/multitask/mt30-{s}M.pt", env=base)
env = dict(base, MUJOCO_GL="egl", PATH=f"{ROOT}/venv/bin:" + base["PATH"])
py = f"{ROOT}/venv/bin/python"
sh(f"cd {ROOT}/code && {py} -c \"import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))\"", env=env)

# Step 1 item 9: cost
sh(f"cd {ROOT}/code && {py} eval_single_task.py cost --sizes {' '.join(map(str, SIZES))} --steps {COST_STEPS} --seed 1 --out {OUT}/cost.json 2>&1 | grep -v -i warn", env=env, check=False)
cost = json.load(open(f"{OUT}/cost.json"))
ok = [v["model_size"] for v in cost.values() if v["compile"] and v["mean_step_s_after_10"] * 100_000 <= 5 * 3600 and v["peak_gpu_mem_MB"] <= 12 * 1024]
chosen = max(ok) if ok else min(SIZES)
json.dump({"rule": "largest size with mean step time x 100000 <= 5 h and peak GPU memory <= 12 GB (compile=true)",
           "eligible": sorted(ok), "chosen": chosen}, open(f"{OUT}/size_choice.json", "w"), indent=1)
print("CHOSEN SIZE", chosen, "eligible", ok, flush=True)

# Step 3: Experiment 0
sh(f"cd {ROOT}/code && {py} eval_single_task.py eval --size {chosen} --episodes {EPISODES} --seed {EVAL_SEED} --out {OUT}/exp0_eval.json 2>&1 | grep -v -i warn", env=env, check=False)
print(f"DONE in {time.time() - T0:.0f}s", flush=True)
