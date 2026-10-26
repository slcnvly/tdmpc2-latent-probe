"""Optional Step 7 / Experiment 3: counterfactual task swap distances (EXPERIMENT_PLAN.md Step 7). CPU, local latents.

Distance (SimNorm-aware): the 768-d latent is 96 groups of 8 entries, each group a softmax (a categorical
distribution; common/layers.py:74-91). d_TV(z, z') = mean over groups of total variation 0.5 * ||z_g - z'_g||_1,
in [0, 1]. Plain L2 ||z - z'||_2 is reported alongside.
Distributions (same state set: test-split states; seed 20261029 for pairings):
  swap      : same s_t, embedding tau vs tau' (10 unordered walker pairs)
  diff-state: same tau, s vs a randomly paired different test state (one random partner per state)
  adjacent  : same tau, s_t vs s_{t+1} of the same episode
Inputs whose task embedding was swapped (z(s, tau') for s visited under tau) may be outside the training distribution.
  python task_swap.py --data ../data --out ../results/exp1/task_swap
"""
import argparse
import json
import os

import numpy as np

TASKS = ["walker-stand", "walker-walk", "walker-run", "walker-walk-backwards", "walker-run-backwards"]


def tv(a, b):
    return 0.5 * np.abs(a - b).reshape(a.shape[0], -1, 8).sum(-1).mean(-1)


def l2(a, b):
    return np.sqrt(((a - b) ** 2).sum(-1))


def q(v):
    return {"n": int(v.size), "mean": float(v.mean()), "q05": float(np.quantile(v, .05)), "q50": float(np.quantile(v, .5)), "q95": float(np.quantile(v, .95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lab = np.load(os.path.join(a.data, "split_labels.npy"))
    te = np.where(lab == "test")[0]
    idx = np.load(os.path.join(a.data, "pooled", "index.npz"))
    rng = np.random.default_rng(20261029)
    partner = rng.permutation(te)
    same = partner == te
    partner[same] = np.roll(partner, 1)[same]
    nxt = te + 1
    adj = (idx["t"][te] < 499)
    res = {}
    for kind in ("learned", "random", "normmatched"):
        Z = np.load(os.path.join(a.data, "pooled", f"latent_{kind}.npy"), mmap_mode="r")
        Zt = [np.asarray(Z[k, te]) for k in range(5)]
        r = {"swap": {}, "diff_state": {}, "adjacent": {}}
        for i in range(5):
            for j in range(i + 1, 5):
                r["swap"][f"{TASKS[i]}|{TASKS[j]}"] = {"tv": q(tv(Zt[i], Zt[j])), "l2": q(l2(Zt[i], Zt[j]))}
            Zp = np.asarray(Z[i, partner])
            r["diff_state"][TASKS[i]] = {"tv": q(tv(Zt[i], Zp)), "l2": q(l2(Zt[i], Zp))}
            Zn = np.asarray(Z[i, nxt[adj]])
            r["adjacent"][TASKS[i]] = {"tv": q(tv(Zt[i][adj], Zn)), "l2": q(l2(Zt[i][adj], Zn))}
        res[kind] = r
        print(kind, "swap tv medians", {k: round(v["tv"]["q50"], 4) for k, v in r["swap"].items()},
              "diff-state", {k: round(v["tv"]["q50"], 4) for k, v in r["diff_state"].items()},
              "adjacent", {k: round(v["tv"]["q50"], 4) for k, v in r["adjacent"].items()}, flush=True)
    json.dump({"definition": __doc__, "results": res}, open(os.path.join(a.out, "task_swap.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
