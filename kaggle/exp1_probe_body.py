"""Kaggle kernel: Experiment 1 Step 6 probes. Rebuilds pooled data / latents / split from the private rollout dataset
with the same scripts, compares sha256 with the local files, then trains all probes (linear on CPU, MLP on 2 x T4)."""
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
sh(f"{ROOT}/venv/bin/hf download nicklashansen/tdmpc2 multitask/mt30-48M.pt --local-dir {ROOT}/ckpts > /dev/null && sha256sum {ROOT}/ckpts/multitask/mt30-48M.pt | tee {OUT}/ckpt_sha256.txt", env=base)
env = dict(base, MUJOCO_GL="egl", PATH=f"{ROOT}/venv/bin:" + base["PATH"])
py = f"{ROOT}/venv/bin/python"
C = f"cd {ROOT}/code && "
DATA = f"{ROOT}/data"

# 1. rollouts from the private dataset, verified against the manifest made before upload
src = [p for p in glob.glob("/kaggle/input/**/rollouts", recursive=True) if os.path.isdir(p)]
assert src, "rollouts dir not found in /kaggle/input"
os.makedirs(DATA, exist_ok=True)
sh(f"cp -r {src[0]} {DATA}/rollouts && cd {DATA} && sha256sum -c {ROOT}/results/exp1/rollouts_manifest_sha256.txt --quiet && echo ROLLOUTS_SHA_OK | tee {OUT}/rollouts_sha_check.txt", env=env)

# 2. rebuild pooled data, latents, split with the same scripts; compare sha256 with the local files
t = time.time()
sh(C + f"{py} step5_latents_targets.py --data {DATA}/rollouts --out {DATA}/pooled 2>&1 | grep -v -i warn | tail -n 25", env=env)
sh(C + f"{py} extract_normmatched.py --pooled {DATA}/pooled 2>&1 | grep -v -i warn | tail -n 5", env=env)
sh(C + f"{py} torso_height.py --pooled {DATA}/pooled 2>&1 | grep -v -i warn | tail -n 2", env=env)
sh(C + f"{py} make_split.py --pooled {DATA}/pooled --out {DATA}/split.json --report {OUT}/split_check_kaggle.json > /dev/null", env=env)
sh(f"cd {DATA} && sha256sum -c {ROOT}/results/exp1/local_data_sha256.txt 2>&1 | tee {OUT}/data_sha_vs_local.txt", env=env, check=False)
sh(f"cp {ROOT}/results/exp1/step5_checks.json {OUT}/step5_checks_kaggle.json; cp {ROOT}/results/exp1/normmatched_meta.json {OUT}/normmatched_meta_kaggle.json", env=env, check=False)
open(f"{OUT}/rebuild_seconds.txt", "w").write(f"{time.time() - t:.1f}\n")
# keep the exact files the probes are trained/evaluated on (small ones) and a latent subset for the local comparison
sh(f"mkdir -p {OUT}/data_kaggle && cp {DATA}/pooled/targets.npy {DATA}/pooled/torso_height.npy {DATA}/pooled/index.npz {DATA}/split.json {DATA}/split_labels.npy {OUT}/data_kaggle/ "
   f"&& cd {DATA} && sha256sum pooled/*.npy pooled/index.npz split.json split_labels.npy > {OUT}/data_kaggle/sha256_kaggle.txt && lscpu | head -20 > {OUT}/cpu_info.txt", env=env, check=False)
sh(f"{py} -c \"import numpy as np; d='{DATA}/pooled'; s=np.arange(0,150000,50); "
   f"np.savez('{OUT}/data_kaggle/latent_subset_every50.npz', idx=s, **{{k: np.load(d+'/latent_'+k+'.npy', mmap_mode='r')[:, s] for k in ('learned','random','normmatched')}})\"", env=env, check=False)

# 3. linear probes (CPU), MLP probes (4 processes on 2 GPUs)
PROBES = f"{OUT}/probes"
t = time.time()
sh(C + f"{py} probe.py linear --data {DATA} --out {PROBES} 2>&1 | tail -n 20", env=env)
open(f"{OUT}/linear_seconds.txt", "w").write(f"{time.time() - t:.1f}\n")
t = time.time()
deadline = T0 + DEADLINE_H * 3600
procs = []
for w in range(4):
    e = dict(env, CUDA_VISIBLE_DEVICES=str(w % 2))
    log = open(f"{OUT}/mlp_worker{w}.log", "w")
    procs.append(subprocess.Popen(C + f"{py} probe.py mlp --data {DATA} --out {PROBES} --device cuda --threads 1 --worker {w} --num-workers 4 --tag w{w} --deadline {deadline}",
                                  shell=True, env=e, stdout=log, stderr=subprocess.STDOUT))
while any(p.poll() is None for p in procs):
    time.sleep(120)
    n = 0
    for w in range(4):
        f = f"{PROBES}/mlp_meta_w{w}.json"
        if os.path.exists(f):
            try:
                n += len(json.load(open(f)))
            except Exception:
                pass
    print(f"[mlp] {time.time() - t:.0f}s, probes done {n}/400", flush=True)
print("mlp return codes", [p.returncode for p in procs], flush=True)
open(f"{OUT}/mlp_seconds.txt", "w").write(f"{time.time() - t:.1f}\n")
sh(f"tail -n 3 {OUT}/mlp_worker*.log; ls -la {PROBES}", env=env, check=False)
print(f"DONE in {time.time() - T0:.0f}s", flush=True)
