"""Kaggle GPU kernel for GATE 2 (VERIFICATION_PLAN.md): V2 traces, V3 official 30-task run (the single approved run),
V5-1 Experiment 0 rerun with reward components (post-V1 script), V5-3 policy-prior-only evaluation (official mpc=false)."""
import os
import subprocess
import time

TEMP = "/kaggle/temp"
OUT = "/kaggle/working/out"
os.makedirs(OUT, exist_ok=True)
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
sh(f"git clone -q https://github.com/nicklashansen/tdmpc2 {ROOT}/tdmpc2 && cd {ROOT}/tdmpc2 && git checkout -q {TDMPC2_COMMIT} && git log -1 --oneline && git status --porcelain | wc -l", env=base)
sh(f"uv venv -q --python 3.11 {ROOT}/venv", env=base)
sh(f"uv pip install -q --python {ROOT}/venv/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128", env=base)
sh(f"cd {ROOT} && grep -v '^torch==' requirements.txt > /tmp/req.txt && uv pip install -q --python venv/bin/python -r /tmp/req.txt", env=base)
sh(f"uv pip freeze --python {ROOT}/venv/bin/python > {OUT}/pip_freeze.txt", env=base)
sh(f"{ROOT}/venv/bin/hf download nicklashansen/tdmpc2 multitask/mt30-48M.pt --local-dir {ROOT}/ckpts > /dev/null && sha256sum {ROOT}/ckpts/multitask/mt30-48M.pt | tee {OUT}/ckpt_sha256.txt", env=base)
env = dict(base, MUJOCO_GL="egl", PATH=f"{ROOT}/venv/bin:" + base["PATH"])
py = f"{ROOT}/venv/bin/python"
CK = f"{ROOT}/ckpts/multitask/mt30-48M.pt"
C = f"cd {ROOT}/code && "

# smoke test of the own-path trace code (3 steps) so a recording bug does not waste the official run
sh(C + f"{py} eval_single_task.py trace --size 48 --tasks walker-walk --steps 3 --out {OUT}/smoke_trace.json 2>&1 | grep -v -i warn | tail -n 5", env=env)

# V3 / V2 official path: evaluate.py unmodified, hooks record only; GPU sampled every 5 s
smi = subprocess.Popen(f"nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -lms 5000 > {OUT}/official_nvidia_smi.csv", shell=True)
t = time.time()
sh(C + f"{py} official_eval_hooked.py --out-dir {OUT}/official --trace-steps 50 -- task=mt30 model_size=48 checkpoint={CK} "
   f"eval_episodes=1 seed=1 save_video=false enable_wandb=false 2>&1 | grep -v -i warn | tee {OUT}/official_stdout.txt", env=env, check=False)
open(f"{OUT}/official_wall_seconds.txt", "w").write(f"{time.time() - t:.1f}\n")
smi.terminate()

# V2 own path: same task order as evaluate.py for tasks 0, 1, 2 (full first episodes), RNG restore for task 20
sh(C + f"{py} eval_single_task.py trace --size 48 --seed 1 --tasks walker-stand walker-walk walker-run walker-run-backwards "
   f"--restore-rng-for walker-run-backwards --rng-file {OUT}/official/official_rng_states.pt --steps 50 --full "
   f"--out {OUT}/own_trace.json 2>&1 | grep -v -i warn | tail -n 8", env=env, check=False)

# V5-1: Experiment 0 rerun with the post-V1 script, same seed and episode count, reward components logged
sh(C + f"{py} eval_single_task.py eval --size 48 --episodes 5 --seed 1 --components --out {OUT}/exp0_rerun_components.json 2>&1 | grep -v -i warn", env=env, check=False)
# V5-3: policy prior only (official config switch mpc=false, tdmpc2.py:114-120), same seed and episode count
sh(C + f"{py} eval_single_task.py eval --size 48 --episodes 5 --seed 1 --components --mpc false --out {OUT}/policy_prior_eval.json 2>&1 | grep -v -i warn", env=env, check=False)
print(f"DONE in {time.time() - T0:.0f}s", flush=True)
