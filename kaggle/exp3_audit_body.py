"""Kaggle kernel: EXPERIMENT3_AUDIT.md items 1 (Kaggle install), 8 (step cost), 9 (official vs own trace), 10 (sanity).
mt80-48M, Meta-World mw-reach / mw-push / mw-pick-place. No training; planning unchanged."""
import glob
import json
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
sh("nvidia-smi -L; nproc; free -g | head -2; df -h /kaggle/temp | tail -1", env=base, check=False)
if subprocess.run("which uv", shell=True).returncode:
    sh(f"curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR={TEMP}/uvbin UV_NO_MODIFY_PATH=1 sh", env=base)
    base["PATH"] = f"{TEMP}/uvbin:{TEMP}/uvbin/bin:" + base["PATH"]
ROOT = f"{TEMP}/tdmpc2-latent-probe"
sh(f"git clone -q https://github.com/slcnvly/tdmpc2-latent-probe {ROOT} && cd {ROOT} && git checkout -q {PROBE_COMMIT} && git log -1 --oneline", env=base)
sh(f"git clone -q https://github.com/nicklashansen/tdmpc2 {ROOT}/tdmpc2 && cd {ROOT}/tdmpc2 && git checkout -q {TDMPC2_COMMIT}", env=base)
sh(f"uv venv -q --python 3.11 {ROOT}/venv", env=base)
sh(f"uv pip install -q --python {ROOT}/venv/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128", env=base)
sh(f"cd {ROOT} && grep -v '^torch==' requirements.txt > /tmp/req.txt && uv pip install -q --python venv/bin/python -r /tmp/req.txt", env=base)
sh(f"uv pip freeze --python {ROOT}/venv/bin/python > {OUT}/pip_freeze.txt", env=base)
sh(f"{ROOT}/venv/bin/hf download nicklashansen/tdmpc2 multitask/mt80-48M.pt --local-dir {ROOT}/ckpts > /dev/null && sha256sum {ROOT}/ckpts/multitask/mt80-48M.pt | tee {OUT}/ckpt_sha256.txt", env=base)

# ---- item 1: Meta-World stack (docker/environment.yaml:34-49 pins), every step timed, failures recorded, not fatal
inst = []
def step(name, cmd, env_=None):
    t = time.time()
    rc = sh(cmd + f" > {OUT}/install_{name}.log 2>&1", env=env_ or base, check=False)
    inst.append({"step": name, "rc": rc, "seconds": round(time.time() - t, 1)})
    json.dump(inst, open(f"{OUT}/install_steps.json", "w"), indent=1)
    if rc:
        sh(f"tail -n 25 {OUT}/install_{name}.log", env=base, check=False)
    return rc
MJ = "/root/.mujoco"
step("gcc_version", "gcc --version | head -1; cat /etc/os-release | head -2")
# libgl1-mesa-glx does not exist on Ubuntu 24.04 (Kaggle image, 2026-10-27 run); libgl1 + libglx-mesa0 are its replacements there
step("apt", "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libosmesa6-dev libgl1 libglx-mesa0 libglfw3 libglew-dev patchelf")
step("mujoco210", f"mkdir -p {MJ} && curl -sSL https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz | tar -xz -C {MJ} && ls {MJ}/mujoco210/bin | head -3")
step("pip23_cython_setuptools", f"uv pip install --python {ROOT}/venv/bin/python 'pip==23.3.2' 'cython<3' 'wheel==0.38.0' 'setuptools==65.5.0'")
step("gym_0_21_0", f"{ROOT}/venv/bin/python -m pip install --no-build-isolation gym==0.21.0")
step("mujoco_py_2_1_2_14", f"{ROOT}/venv/bin/python -m pip install --no-build-isolation mujoco-py==2.1.2.14")
step("metaworld_04be337a", f"{ROOT}/venv/bin/python -m pip install --no-build-isolation --no-deps 'git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb'")
menv = dict(base, MUJOCO_GL="egl", LD_LIBRARY_PATH=f"{MJ}/mujoco210/bin:/usr/lib/nvidia:" + base.get("LD_LIBRARY_PATH", ""), PATH=f"{ROOT}/venv/bin:" + base["PATH"])
rc = step("import_mujoco_py_first_build", f"cd {TEMP} && {ROOT}/venv/bin/python -c 'import mujoco_py; print(mujoco_py.__version__)'", menv)
if rc:
    menv["CFLAGS"] = "-Wno-error=incompatible-pointer-types -Wno-error=implicit-function-declaration"
    rc = step("import_mujoco_py_retry_relaxed_cflags", f"cd {TEMP} && {ROOT}/venv/bin/python -c 'import mujoco_py; print(mujoco_py.__version__)'", menv)
rc2 = step("import_metaworld_make_3_tasks", f"cd {ROOT}/code && {ROOT}/venv/bin/python -c \"from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as A; [print(t, A[t+'-v2-goal-observable'](seed=1).reset().shape) for t in ('reach','push','pick-place')]\"", menv)
sh(f"uv pip freeze --python {ROOT}/venv/bin/python > {OUT}/pip_freeze.txt", env=base, check=False)
if rc or rc2:
    print("META-WORLD NOT USABLE ON KAGGLE: items 8-10 not run", flush=True)
    raise SystemExit(0)

C = f"cd {ROOT}/code && "
py = f"{ROOT}/venv/bin/python"
# ---- item 8: step cost (planning included)
sh(C + f"{py} exp3_gpu.py cost --out {OUT}/cost.json 2>&1 | grep -v -i warn | tail -n 5", env=menv, check=False)
# ---- item 9 (official side) + item 10 (official per-task results): unmodified evaluate.py via runpy, recording hooks only
t = time.time()
sh(C + f"{py} official_eval_hooked.py --out-dir {OUT}/official --trace-ids 56,63,65 --trace-steps 50 --patch-timeout -- "
   f"task=mt80 model_size=48 checkpoint={ROOT}/ckpts/multitask/mt80-48M.pt eval_episodes=1 seed=1 save_video=false enable_wandb=false "
   f"2>&1 | grep -v -i warn | tee {OUT}/official_stdout.txt | tail -n 90", env=menv, check=False)
open(f"{OUT}/official_seconds.txt", "w").write(f"{time.time() - t:.1f}\n")
# ---- item 9 (own side): RNG states restored from the official run
sh(C + f"{py} exp3_gpu.py trace --rng-dir {OUT}/official --out {OUT}/own_trace.json 2>&1 | grep -v -i warn | tail -n 5", env=menv, check=False)
# ---- item 10: 5 episodes per task, own path
sh(C + f"{py} exp3_gpu.py sanity --episodes 5 --out {OUT}/sanity.json 2>&1 | grep -v -i warn | tail -n 20", env=menv, check=False)
print(f"DONE in {time.time() - T0:.0f}s", flush=True)
