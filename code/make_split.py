"""Step 6-0: episode-level split (EXPERIMENT1_PREREG.md 4) and its checks.
  python make_split.py --pooled ../data/pooled --out ../data/split.json --report ../results/exp1/split_check.json
"""
import argparse
import json
import os

import numpy as np

TASKS = ["walker-stand", "walker-walk", "walker-run", "walker-walk-backwards", "walker-run-backwards"]
TYPES = ["A", "B", "C"]
SPLIT_SEED = 20261026


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooled", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    idx = np.load(os.path.join(a.pooled, "index.npz"))
    task, ty, ep, t = idx["task"], idx["rollout_type"], idx["episode"], idx["t"]
    state = np.load(os.path.join(a.pooled, "state.npy"), mmap_mode="r")
    rng = np.random.default_rng(SPLIT_SEED)
    assign = {}  # episode key "task/type/ep" -> split
    strata = [(f"{TASKS[k]}/{TYPES[y]}", [(k, y, e) for e in range(20)]) for y in (0, 1) for k in range(5)]
    strata.append(("C", [(k, 2, e) for k in range(5) for e in range(20)]))
    for name, eps in strata:  # fixed order of strata; shuffle within each stratum, then cut 60/20/20
        order = rng.permutation(len(eps))
        n = len(eps)
        n_tr, n_va = int(round(0.6 * n)), int(round(0.2 * n))
        for r, i in enumerate(order):
            k, y, e = eps[i]
            assign[f"{TASKS[k]}/{TYPES[y]}/{e}"] = "train" if r < n_tr else ("val" if r < n_tr + n_va else "test")
    json.dump({"seed": SPLIT_SEED, "rule": "EXPERIMENT1_PREREG.md 4: A,B stratified by (source task, type) 12/4/4; C pooled 60/20/20",
               "strata_order": [s for s, _ in strata], "episodes": assign}, open(a.out, "w"), indent=1)

    # per-state split label and checks
    keys = np.array([f"{TASKS[k]}/{TYPES[y]}/{e}" for k, y, e in zip(task, ty, ep)])
    lab = np.array([assign[x] for x in keys])
    rep = {"n_states": int(len(lab)), "split_file": os.path.basename(a.out)}
    for s in ("train", "val", "test"):
        m = lab == s
        rep[s] = {"episodes": int(len(set(keys[m]))), "states": int(m.sum())}
    per = {}
    for name, eps in strata:
        c = {"train": 0, "val": 0, "test": 0}
        for k, y, e in eps:
            c[assign[f"{TASKS[k]}/{TYPES[y]}/{e}"]] += 1
        per[name] = c
    rep["episodes_per_stratum"] = per
    # C test/val/train per source label (descriptive; C is not stratified by label)
    rep["C_episodes_per_label"] = {TASKS[k]: {s: sum(1 for e in range(20) if assign[f"{TASKS[k]}/C/{e}"] == s) for s in ("train", "val", "test")} for k in range(5)}
    # no episode in two splits: by construction one label per key; verify every state of an episode has one label
    ep_labels = {}
    for kk, ll in zip(keys, lab):
        ep_labels.setdefault(kk, set()).add(ll)
    rep["episode_in_two_splits"] = int(sum(len(v) > 1 for v in ep_labels.values()))
    rep["all_300_episodes_assigned"] = len(assign) == 300 and len(ep_labels) == 300
    rep["same_state_five_latents_same_split"] = "latent arrays are (5, N, 768) indexed by the same state index n; the split label is per n, so the 5 latents of a state share it"
    json.dump(rep, open(a.report, "w"), indent=1)
    np.save(os.path.join(os.path.dirname(a.out), "split_labels.npy"), lab)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
