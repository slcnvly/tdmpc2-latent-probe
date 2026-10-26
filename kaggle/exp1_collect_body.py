"""Kaggle kernel: Experiment 1 Step 4 rollout collection (EXPERIMENT1_PREREG.md).
Phase 1: episode 0 of all 15 conditions, then integrity checks. Phase 2 only if every check passes."""
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
sh("nvidia-smi -L; nproc; free -g | head -2; df -h /kaggle/working | tail -1", env=base, check=False)
if subprocess.run("which uv", shell=True).returncode:
    sh(f"curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR={TEMP}/uvbin UV_NO_MODIFY_PATH=1 sh", env=base)
    base["PATH"] = f"{TEMP}/uvbin:{TEMP}/uvbin/bin:" + base["PATH"]
ROOT = f"{TEMP}/tdmpc2-latent-probe"
sh(f"git clone -q https://github.com/slcnvly/tdmpc2-latent-probe {ROOT} && cd {ROOT} && git checkout -q {PROBE_COMMIT} && git log -1 --oneline", env=base)
sh(f"git clone -q https://github.com/nicklashansen/tdmpc2 {ROOT}/tdmpc2 && cd {ROOT}/tdmpc2 && git checkout -q {TDMPC2_COMMIT} && git status --porcelain | wc -l", env=base)
sh(f"uv venv -q --python 3.11 {ROOT}/venv", env=base)
sh(f"uv pip install -q --python {ROOT}/venv/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128", env=base)
sh(f"cd {ROOT} && grep -v '^torch==' requirements.txt > /tmp/req.txt && uv pip install -q --python venv/bin/python -r /tmp/req.txt", env=base)
sh(f"uv pip freeze --python {ROOT}/venv/bin/python > {OUT}/pip_freeze.txt", env=base)
sh(f"{ROOT}/venv/bin/hf download nicklashansen/tdmpc2 multitask/mt30-48M.pt --local-dir {ROOT}/ckpts > /dev/null && sha256sum {ROOT}/ckpts/multitask/mt30-48M.pt | tee {OUT}/ckpt_sha256.txt", env=base)
env = dict(base, MUJOCO_GL="egl", PATH=f"{ROOT}/venv/bin:" + base["PATH"])
py = f"{ROOT}/venv/bin/python"
DATA = f"{OUT}/rollouts"
C = f"cd {ROOT}/code && "


def phase(eps, tag):
    procs = []
    for k in range(2):
        e = dict(env, CUDA_VISIBLE_DEVICES=str(k))
        log = open(f"{OUT}/collect_{tag}_gpu{k}.log", "w")
        procs.append(subprocess.Popen(C + f"{py} collect_rollouts.py --types A B --episodes {eps} --worker {k} --num-workers 2 --out {DATA}",
                                      shell=True, env=e, stdout=log, stderr=subprocess.STDOUT))
    log = open(f"{OUT}/collect_{tag}_cpu.log", "w")
    procs.append(subprocess.Popen(C + f"{py} collect_rollouts.py --types C --episodes {eps} --out {DATA}",
                                  shell=True, env=dict(env, CUDA_VISIBLE_DEVICES=""), stdout=log, stderr=subprocess.STDOUT))
    t = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(60)
        n = sum(len([f for f in fs if f.endswith(".npz") and ".tmp" not in f]) for _, _, fs in os.walk(DATA))
        print(f"[{tag}] {time.time() - t:.0f}s, episode files: {n}", flush=True)
    rcs = [p.returncode for p in procs]
    print(f"[{tag}] return codes {rcs}", flush=True)
    sh(f"grep -v -i warn {OUT}/collect_{tag}_*.log | grep -i 'error\\|traceback\\|return' | tail -n 40", env=env, check=False)
    return rcs


t1 = time.time()
rcs = phase("0", "phase1")
sh(C + f"{py} check_rollouts.py --data {DATA} --episodes 0 --out {OUT}/check_ep0.json 2>&1 | grep -v -i warn | tail -n 20", env=env, check=False)
import json  # noqa: E402

ok = os.path.exists(f"{OUT}/check_ep0.json") and json.load(open(f"{OUT}/check_ep0.json"))["all_pass"] and \
    json.load(open(f"{OUT}/check_ep0.json"))["n_episodes"] == 15
open(f"{OUT}/phase1_seconds.txt", "w").write(f"{time.time() - t1:.1f}\n")
if not ok:
    print("PHASE 1 CHECKS FAILED OR INCOMPLETE -> not collecting the rest", flush=True)
else:
    t2 = time.time()
    phase("1-19", "phase2")
    open(f"{OUT}/phase2_seconds.txt", "w").write(f"{time.time() - t2:.1f}\n")
    sh(C + f"{py} check_rollouts.py --data {DATA} --episodes 0-19 --out {OUT}/check_all.json 2>&1 | grep -v -i warn | tail -n 5", env=env, check=False)
sh(f"du -sh {DATA}; find {DATA} -name '*.npz' | wc -l", env=env, check=False)
print(f"DONE in {time.time() - T0:.0f}s", flush=True)
