"""V2 trace comparison and V3 score comparison from the GATE 2 kernel output (CPU, after download).

  python v2_v3_compare.py --dir ../results/gate2
Writes <dir>/v2_trace_compare.json, <dir>/v2_trace_diff.png, <dir>/v3_scores.json. Criteria: VERIFICATION_PLAN.md.
"""
import argparse
import json
import os
import re

import numpy as np

TASKS = {"walker-walk": 1, "walker-run": 2, "walker-run-backwards": 20}
WALKER = {"walker-stand": 0, "walker-walk": 1, "walker-run": 2, "walker-walk-backwards": 19, "walker-run-backwards": 20}
TOL_OBS, TOL_ACT = 1e-5, 1e-4


def v2(d):
    off = json.load(open(os.path.join(d, "official", "official_hooked.json")))
    own = json.load(open(os.path.join(d, "own_trace.json")))
    res = {"errors_in_official_hooks": off["errors"]}
    curves = {}
    for task, k in TASKS.items():
        a, b = off["trace"].get(str(k), []), own["tasks"].get(task, {}).get("trace", [])
        n = min(len(a), len(b))
        rows = []
        for i in range(n):
            x, y = a[i], b[i]
            assert x["t"] == y["t"] == i
            rows.append({"t": i, "task_equal": x["task"] == y["task"] == k,
                         "obs": float(np.abs(np.array(x["obs"]) - np.array(y["obs"])).max()),
                         "enc_in": float(np.abs(np.array(x["enc_in"]) - np.array(y["enc_in"])).max()),
                         "action": float(np.abs(np.array(x["action"]) - np.array(y["action"])).max())})
        if n == 0:
            res[task] = {"verdict": "undetermined (no trace)", "n_steps": 0}
            continue
        ok = lambda r: r["task_equal"] and r["obs"] <= TOL_OBS and r["enc_in"] <= TOL_OBS and r["action"] <= TOL_ACT
        step0_ok = ok(rows[0])
        all_ok = all(ok(r) for r in rows) and n >= 50
        first_bad = next((r["t"] for r in rows if not ok(r)), None)
        verdict = "pass" if all_ok else ("fail" if not step0_ok else "undetermined (nondeterminism)")
        if n < 50 and step0_ok and first_bad is None:
            verdict = f"undetermined (only {n} steps compared)"
        res[task] = {"verdict": verdict, "n_steps": n, "first_step_over_tolerance": first_bad,
                     "max_over_steps": {q: max(r[q] for r in rows) for q in ("obs", "enc_in", "action")},
                     "step0": rows[0], "rows": rows, "rng_restored_in_own_path": own["tasks"][task]["rng_restored"]}
        curves[task] = rows
    json.dump(res, open(os.path.join(d, "v2_trace_compare.json"), "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (task, rows) in zip(axes, curves.items()):
        for q, c in (("obs", "#1f77b4"), ("enc_in", "#2ca02c"), ("action", "#d62728")):
            v = np.array([r[q] for r in rows])
            ax.semilogy(range(len(v)), np.maximum(v, 1e-12), color=c, label=q, lw=1.4)
        ax.axhline(TOL_OBS, color="#1f77b4", ls=":", lw=1)
        ax.axhline(TOL_ACT, color="#d62728", ls=":", lw=1)
        ax.set_title(f"{task}: {res[task]['verdict']}")
        ax.set_xlabel("step t")
    axes[0].set_ylabel("max |official - own|  (zero plotted at 1e-12)")
    axes[0].legend()
    fig.suptitle("V2 trace equivalence, first 50 steps (dotted: tolerance 1e-5 obs/enc_in, 1e-4 action)")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "v2_trace_diff.png"), dpi=130)
    return res, off, own


def v3(d, off, own):
    txt = open(os.path.join(d, "official_stdout.txt")).read()
    printed = {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s+(\S+)\s+R:\s+([-0-9.]+)", txt, flags=re.M)}
    norm = re.search(r"Normalized score:\s+([-0-9.]+)", txt)
    rewards = off["rewards"]
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tdmpc2", "tdmpc2"))
    from common import TASK_SET
    tasks = TASK_SET["mt30"]
    import torch

    def evaluate_py_sum(rs):  # evaluate.py:76, 82: ep_reward = 0; ep_reward += reward (float32 tensor), in step order
        acc = 0
        for r in rs:
            acc += torch.tensor(r, dtype=torch.float32)
        return float(acc)

    hooked = {t: evaluate_py_sum(rewards[str(i)]) for i, t in enumerate(tasks) if str(i) in rewards}
    hooked_pairwise = {t: float(np.sum(np.array(rewards[str(i)], dtype=np.float32))) for i, t in enumerate(tasks) if str(i) in rewards}
    scores = [hooked[t] / 10 for t in tasks if t in hooked]  # evaluate.py:94 (all mt30 tasks are DMControl)
    exp0_old = json.load(open(os.path.join(d, "..", "gate1", "exp0_eval.json")))
    exp0_new = json.load(open(os.path.join(d, "exp0_rerun_components.json")))
    walker = {}
    for t in WALKER:
        o = hooked.get(t)
        row = {"official_1ep": o, "official_printed": printed.get(t)}
        for name, e in (("exp0_pre_v1", exp0_old), ("exp0_post_v1", exp0_new)):
            r = [x["return"] for x in e["tasks"][t]["episodes"]]
            row[name] = {"returns": r, "min": min(r), "max": max(r), "in_range": (o is not None) and min(r) <= o <= max(r)}
        if t in own["tasks"]:
            row["own_trace_first_episode"] = own["tasks"][t]["return"]
        walker[t] = row
    res = {"n_tasks_with_rewards": len(hooked), "per_task_return": hooked, "per_task_return_np_sum_first_version": hooked_pairwise, "printed_R": printed,
           "normalized_score_recomputed": float(np.mean(scores)) if len(scores) == 30 else None,
           "normalized_score_printed": float(norm.group(1)) if norm else None, "paper_mt30_48M": 59.4, "walker": walker}
    json.dump(res, open(os.path.join(d, "v3_scores.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    r2, off, own = v2(a.dir)
    for t in TASKS:
        print("V2", t, {k: v for k, v in r2[t].items() if k != "rows"})
    r3 = v3(a.dir, off, own)
    print("V3 normalized", r3["normalized_score_recomputed"], r3["normalized_score_printed"])
    for t, v in r3["walker"].items():
        print("V3", t, v)
