"""Step 6 evaluation: R^2, episode-level bootstrap CIs, paired bootstrap differences, heatmaps, tables.
Criteria: EXPERIMENT1_PREREG.md 6 (+ addendum A1, A2). Reads the probe predictions only.

  python evaluate_probes.py --data ../data --probes ../results/exp1/probes --out ../results/exp1/eval
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

TASKS = ["walker-stand", "walker-walk", "walker-run", "walker-walk-backwards", "walker-run-backwards"]
B, BOOT_SEED, SST_MIN = 1000, 20261027, 1e-10
KINDS = ["learned", "random", "normmatched"]


def load_preds(probes):
    lin = dict(np.load(os.path.join(probes, "pred_linear.npz")))
    mlp = {}
    for f in sorted(glob.glob(os.path.join(probes, "pred_mlp_*.npz"))):
        mlp.update(dict(np.load(f)))
    return lin, mlp


def episode_ids(idx_npz, sel):
    key = idx_npz["task"][sel] * 10000 + idx_npz["rollout_type"][sel] * 100 + idx_npz["episode"][sel]
    uniq, inv = np.unique(key, return_inverse=True)
    return uniq, inv


class Subset:
    """A set of test states, grouped by episode, with one shared bootstrap weight matrix."""

    def __init__(self, name, mask_test, idx_test, y_test):
        self.name = name
        self.mask = mask_test
        sel_idx = {k: v[mask_test] for k, v in idx_test.items()}
        self.uniq, self.inv = episode_ids(sel_idx, np.ones(mask_test.sum(), bool))
        self.E = len(self.uniq)
        self.y = y_test[mask_test]
        self.n_e = np.bincount(self.inv, minlength=self.E).astype(float)
        self.Sy = np.stack([np.bincount(self.inv, self.y[:, j], self.E) for j in range(5)], 1)
        self.Syy = np.stack([np.bincount(self.inv, self.y[:, j] ** 2, self.E) for j in range(5)], 1)
        rng = np.random.default_rng(BOOT_SEED)
        draws = rng.integers(0, self.E, size=(B, self.E))
        self.W = np.stack([np.bincount(d, minlength=self.E) for d in draws]).astype(float)

    def sst(self, j, w):
        n = w @ self.n_e
        return w @ self.Syy[:, j] - (w @ self.Sy[:, j]) ** 2 / n

    def sse_e(self, j, pred):
        return np.bincount(self.inv, (self.y[:, j] - pred[self.mask]) ** 2, self.E)

    def r2_point(self, j, pred):
        one = np.ones(self.E)
        sst = self.sst(j, one)
        if self.E < 2 or sst < SST_MIN:
            return None
        return float(1 - self.sse_e(j, pred).sum() / sst)

    def r2_boot(self, j, preds):
        """preds: list of prediction vectors (MLP seeds) -> (B,) mean-over-seeds R^2, NaN where SST too small."""
        sst = self.W @ self.Syy[:, j] - (self.W @ self.Sy[:, j]) ** 2 / (self.W @ self.n_e)
        r = np.mean([1 - (self.W @ self.sse_e(j, p)) / sst for p in preds], axis=0)
        r[sst < SST_MIN] = np.nan
        return r


def ci(v):
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], int(B - len(v))


def plot(mat, title, fname, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    m = np.array([[np.nan if x is None else x for x in r] for r in mat], dtype=float)
    im = ax.imshow(np.clip(m, -1, 1), cmap="RdBu", vmin=-1, vmax=1)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, "n/a" if np.isnan(m[i, j]) else f"{m[i, j]:.3f}", ha="center", va="center", fontsize=10,
                    color="white" if np.isfinite(m[i, j]) and abs(np.clip(m[i, j], -1, 1)) > 0.6 else "black")
    ax.set_xticks(range(5))
    ax.set_xticklabels(TASKS, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("target: reward of task tau'  r_tau'(s_t)")
    ax.set_ylabel("encoder input task embedding tau")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="test R^2 (color clipped to [-1, 1]; numbers unclipped)")
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lab = np.load(os.path.join(a.data, "split_labels.npy"))
    te = np.where(lab == "test")[0]
    idx = np.load(os.path.join(a.data, "pooled", "index.npz"))
    idx_test = {k: idx[k][te] for k in ("task", "rollout_type", "episode", "t")}
    y_test = np.load(os.path.join(a.data, "pooled", "targets.npy"))[te]
    h_test = np.load(os.path.join(a.data, "pooled", "torso_height.npy"))[te]
    lin, mlp = load_preds(a.probes)

    subsets = {"pooled": Subset("pooled", np.ones(len(te), bool), idx_test, y_test),
               "not_fallen": Subset("not_fallen", h_test >= 0.6, idx_test, y_test)}
    for k in range(5):
        subsets[f"indist_A_row{k}"] = Subset(f"indist_A_row{k}", (idx_test["task"] == k) & (idx_test["rollout_type"] == 0), idx_test, y_test)
        subsets[f"indist_AB_row{k}"] = Subset(f"indist_AB_row{k}", (idx_test["task"] == k) & (idx_test["rollout_type"] <= 1), idx_test, y_test)

    def preds_of(probe, name, j):
        if probe == "linear":
            return [lin[f"{name}|{j}|test"]]
        ps = [mlp.get(f"{name}|{j}|{s}|test") for s in range(5)]
        return [p for p in ps if p is not None]

    cells, diffs, boots = [], [], {}
    for sname, S in subsets.items():
        rows_here = [int(sname[-1])] if sname.startswith("indist") else list(range(5))
        inputs = [f"{kind}:{k}" for kind in KINDS for k in rows_here] + ["raw"]
        for probe in ("linear", "mlp"):
            for name in inputs:
                for j in range(5):
                    ps = preds_of(probe, name, j)
                    if not ps:
                        continue
                    seed_r2 = [S.r2_point(j, p) for p in ps]
                    point = None if any(x is None for x in seed_r2) else float(np.mean(seed_r2))
                    bt = S.r2_boot(j, ps)
                    boots[(sname, probe, name, j)] = bt
                    (lo, hi), dropped = ci(bt) if point is not None else ([None, None], B)
                    cells.append({"subset": sname, "probe": probe, "input": name, "target": TASKS[j], "r2": point, "ci_lo": lo, "ci_hi": hi,
                                  "boot_dropped": dropped, "n_states": int(S.mask.sum()), "n_episodes": S.E, "n_seeds": len(ps),
                                  "seed_r2": seed_r2 if probe == "mlp" else None,
                                  "seed_std": float(np.std(seed_r2)) if probe == "mlp" and point is not None else None,
                                  "note": None if point is not None else "R2 not computed: target variance < 1e-10 or < 2 episodes"})
            # paired differences (primary criterion), same bootstrap draws
            for j in range(5):
                comps = []
                for k in rows_here:
                    comps += [("a_learned_vs_raw", f"learned:{k}", "raw"), ("b_learned_vs_random", f"learned:{k}", f"random:{k}"),
                              ("d_learned_vs_normmatched", f"learned:{k}", f"normmatched:{k}")]
                if not sname.startswith("indist") and j < 5:
                    comps += [("c_diag_vs_row", f"learned:{j}", f"learned:{k}") for k in range(5) if k != j]
                for cname, x1, x2 in comps:
                    b1, b2 = boots.get((sname, probe, x1, j)), boots.get((sname, probe, x2, j))
                    if b1 is None or b2 is None:
                        continue
                    p1 = next(c["r2"] for c in cells if c["subset"] == sname and c["probe"] == probe and c["input"] == x1 and c["target"] == TASKS[j])
                    p2 = next(c["r2"] for c in cells if c["subset"] == sname and c["probe"] == probe and c["input"] == x2 and c["target"] == TASKS[j])
                    if p1 is None or p2 is None:
                        diffs.append({"subset": sname, "probe": probe, "comparison": cname, "x1": x1, "x2": x2, "target": TASKS[j], "diff": None, "verdict": "not computed"})
                        continue
                    d = b1 - b2
                    (lo, hi), dropped = ci(d)
                    c1 = next(c for c in cells if c["subset"] == sname and c["probe"] == probe and c["input"] == x1 and c["target"] == TASKS[j])
                    c2 = next(c for c in cells if c["subset"] == sname and c["probe"] == probe and c["input"] == x2 and c["target"] == TASKS[j])
                    overlap = not (c1["ci_hi"] < c2["ci_lo"] or c2["ci_hi"] < c1["ci_lo"])
                    diffs.append({"subset": sname, "probe": probe, "comparison": cname, "x1": x1, "x2": x2, "target": TASKS[j],
                                  "diff": p1 - p2, "ci_lo": lo, "ci_hi": hi, "boot_dropped": dropped,
                                  "verdict": "difference" if (lo > 0 or hi < 0) else "not distinguishable",
                                  "aux_ci_overlap": overlap})
    with open(os.path.join(a.out, "cells.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
        w.writeheader()
        for c in cells:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in c.items()})
    with open(os.path.join(a.out, "paired_differences.csv"), "w", newline="") as f:
        keys = ["subset", "probe", "comparison", "x1", "x2", "target", "diff", "ci_lo", "ci_hi", "boot_dropped", "verdict", "aux_ci_overlap"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for d in diffs:
            w.writerow({k: d.get(k) for k in keys})
    # subset descriptions
    desc = {s: {"n_states": int(S.mask.sum()), "n_episodes": S.E,
                "target_mean": [float(S.y[:, j].mean()) for j in range(5)], "target_var": [float(S.y[:, j].var()) for j in range(5)]}
            for s, S in subsets.items()}
    json.dump({"subsets": desc, "B": B, "boot_seed": BOOT_SEED}, open(os.path.join(a.out, "subsets.json"), "w"), indent=1)
    # heatmaps
    get = lambda s, p, name, j: next((c["r2"] for c in cells if c["subset"] == s and c["probe"] == p and c["input"] == name and c["target"] == TASKS[j]), None)
    for s in ("pooled", "not_fallen"):
        for p in ("linear", "mlp"):
            for kind, tag in (("learned", "trained"), ("random", "random_init"), ("normmatched", "normmatched_random_init")):
                mat = [[get(s, p, f"{kind}:{k}", j) for j in range(5)] for k in range(5)]
                sub = "pooled test" if s == "pooled" else "pooled test without fallen states (torso height < 0.6 m)"
                fn = f"heatmap_{p}_{tag}_{'pooled' if s == 'pooled' else 'not_fallen'}.png"
                plot(mat, f"{p.upper()} probe, {kind} encoder z(s, tau), {sub}\n" + ("MLP: mean of 5 seeds" if p == "mlp" else "ridge"),
                     os.path.join(a.out, fn), TASKS)
    json.dump({"cells": len(cells), "diffs": len(diffs)}, open(os.path.join(a.out, "done.json"), "w"))
    print("cells", len(cells), "diffs", len(diffs))


if __name__ == "__main__":
    main()
