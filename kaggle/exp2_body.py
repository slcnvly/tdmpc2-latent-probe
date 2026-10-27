"""Kaggle kernel: Experiment 2 (EXPERIMENT2_PREREG.md). Rebuilds obs / targets / torso_height / split from the private rollout
dataset with the Experiment 1 scripts, stops unless their sha256 equal the PREREG values, runs the inject() check, then
encodes and probes the 37 rows (4 processes on 2 x T4), then the standing-only probes."""
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


EXPECT = {"pooled/obs.npy": "64b9e182d5be7e585c7a3ca967d1e9a5f924f577b3a0c6216b6e2fd217e1477b",
          "pooled/targets.npy": "10fcd0ebfb2b45164458062f011aebf2e631176d626cd9060dfcd78cf6cf2f64",
          "pooled/torso_height.npy": "1610695ec26a0b7579a36723f8c833e1759948145629ae552a38724b470258ca",
          "split_labels.npy": "5c2121764a9d5d80259e55463623cd089efd50f48a3057b38e5ac2d5eb24dcb2"}
src = [p for p in glob.glob("/kaggle/input/**/rollouts", recursive=True) if os.path.isdir(p)]
assert src, "rollouts dir not found in /kaggle/input"
os.makedirs(DATA, exist_ok=True)
sh(f"cp -r {src[0]} {DATA}/rollouts && cd {DATA} && sha256sum -c {ROOT}/results/exp1/rollouts_manifest_sha256.txt --quiet && echo ROLLOUTS_SHA_OK | tee {OUT}/rollouts_sha_check.txt", env=env)
t = time.time()
sh(C + f"{py} step5_latents_targets.py --data {DATA}/rollouts --out {DATA}/pooled 2>&1 | grep -v -i warn | tail -n 5", env=env)
sh(C + f"{py} torso_height.py --pooled {DATA}/pooled 2>&1 | grep -v -i warn | tail -n 2", env=env)
sh(C + f"{py} make_split.py --pooled {DATA}/pooled --out {DATA}/split.json --report {OUT}/split_check_kaggle.json > /dev/null", env=env)
open(f"{OUT}/expected_sha256.txt", "w").write("".join(f"{v}  {k}\n" for k, v in EXPECT.items()))
sh(f"cd {DATA} && sha256sum -c {OUT}/expected_sha256.txt 2>&1 | tee {OUT}/data_sha_check.txt", env=env)  # stops the kernel on mismatch
open(f"{OUT}/rebuild_seconds.txt", "w").write(f"{time.time() - t:.1f}\n")
sh(C + f"{py} exp2.py check --data {DATA} --out-json {OUT}/inject_check_kaggle.json 2>&1 | grep -v -i warn | tail -n 30", env=env)
sh(f"rm -f {DATA}/pooled/latent_*.npy; lscpu | head -20 > {OUT}/cpu_info.txt", env=env, check=False)

ROWS = f"{OUT}/rows"
t = time.time()
procs = []
for w in range(4):
    e = dict(env, CUDA_VISIBLE_DEVICES=str(w % 2), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    log = open(f"{OUT}/worker{w}.log", "w")
    procs.append(subprocess.Popen(C + f"{py} exp2.py run --data {DATA} --out {ROWS} --device cuda --threads 1 --worker {w} --num-workers 4",
                                  shell=True, env=e, stdout=log, stderr=subprocess.STDOUT))
while any(p.poll() is None for p in procs):
    time.sleep(120)
    n = len(glob.glob(f"{ROWS}/row_*.json"))
    print(f"[rows] {time.time() - t:.0f}s, rows done {n}/37", flush=True)
print("worker return codes", [p.returncode for p in procs], flush=True)
open(f"{OUT}/rows_seconds.txt", "w").write(f"{time.time() - t:.1f}\n")
sh(f"tail -n 3 {OUT}/worker*.log", env=env, check=False)
t = time.time()
sh(C + f"{py} exp2.py standing --data {DATA} --out {ROWS} 2>&1 | grep -v -i warn | tail -n 20", env=env)
open(f"{OUT}/standing_seconds.txt", "w").write(f"{time.time() - t:.1f}\n")
print(f"DONE in {time.time() - T0:.0f}s", flush=True)
