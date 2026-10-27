"""Experiment 2 evaluation (EXPERIMENT2_PREREG.md 5-8). Reads the per-row predictions written by exp2.py.

  python exp2_evaluate.py --data ../data_eval_kaggle --rows ../results/exp2/kaggle/rows \
      --emb ../results/exp2/kaggle/inject_check_kaggle.json --out ../results/exp2/eval
"""
import argparse
import csv
import json
import os

import numpy as np

from evaluate_probes import B, BOOT_SEED, SST_MIN, episode_ids

TASKS = ["walker-stand", "walker-walk", "walker-run", "walker-walk-backwards", "walker-run-backwards"]
THR = 0.02
RMSE_RATIO = 1.5
REF = "randinit:walker-stand"


def fname(name):
    return f"row_{name.replace(':', '__')}"


class Sub:
    def __init__(self, mask, idx_test, y_test):
        self.mask = mask
        self.uniq, self.inv = episode_ids({k: v[mask] for k, v in idx_test.items()}, np.ones(mask.sum(), bool))
        self.E = len(self.uniq)
        self.y = y_test[mask]
        self.n_e = np.bincount(self.inv, minlength=self.E).astype(float)
        self.Sy = np.stack([np.bincount(self.inv, self.y[:, j], self.E) for j in range(5)], 1)
        self.Syy = np.stack([np.bincount(self.inv, self.y[:, j] ** 2, self.E) for j in range(5)], 1)
        draws = np.random.default_rng(BOOT_SEED).integers(0, self.E, size=(B, self.E))
        self.W = np.stack([np.bincount(d, minlength=self.E) for d in draws]).astype(float)
        self.sst_b = self.W @ self.Syy - (self.W @ self.Sy) ** 2 / (self.W @ self.n_e)[:, None]  # (B, 5)
        self.sst = self.Syy.sum(0) - self.Sy.sum(0) ** 2 / self.n_e.sum()

    def stats(self, preds, j):
        """preds: list (seeds) of (n_sub,) -> point R2, boot R2 (B,), point RMSE, boot RMSE (B,); seed means."""
        sse = [np.bincount(self.inv, (self.y[:, j] - p) ** 2, self.E) for p in preds]
        r2p = float(np.mean([1 - s.sum() / self.sst[j] for s in sse]))
        r2b = np.mean([1 - (self.W @ s) / self.sst_b[:, j] for s in sse], axis=0)
        r2b[self.sst_b[:, j] < SST_MIN] = np.nan
        rmp = float(np.mean([np.sqrt(s.sum() / self.n_e.sum()) for s in sse]))
        rmb = np.mean([np.sqrt((self.W @ s) / (self.W @ self.n_e)) for s in sse], axis=0)
        return r2p, r2b, rmp, rmb


def pct(v):
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(B - len(v))


def evaluate(setting, S, preds_by_row):
    """preds_by_row: row -> list of (n_sub, 5) arrays. Returns per-cell records."""
    st = {r: [S.stats([p[:, j] for p in ps], j) for j in range(5)] for r, ps in preds_by_row.items()}
    out = []
    for r in preds_by_row:
        for j in range(5):
            r2p, r2b, rmp, rmb = st[r][j]
            d_r2p, d_r2b, d_rmp, _ = st[f"task:{TASKS[j]}"][j]
            lo, hi, drop = pct(r2b)
            rlo, rhi, _ = pct(rmb)
            dlo, dhi, ddrop = pct(d_r2b - r2b)
            diff = d_r2p - r2p
            low = bool(dlo > 0 and diff >= THR)
            ref = st[REF][j][0] if REF in st else None
            out.append({"setting": setting, "row": r, "target": TASKS[j], "r2": r2p, "r2_lo": lo, "r2_hi": hi, "boot_dropped": drop,
                        "rmse": rmp, "rmse_lo": rlo, "rmse_hi": rhi, "diag_r2": d_r2p, "diff": diff, "diff_lo": dlo, "diff_hi": dhi,
                        "diff_boot_dropped": ddrop, "verdict": "실질적으로 낮음" if low else "구분되지 않음 또는 실질적 차이 없음",
                        "rmse_ratio": rmp / d_rmp, "rmse_ratio_ge_1_5": bool(rmp / d_rmp >= RMSE_RATIO),
                        "randinit_ref_r2": ref, "below_randinit": None if ref is None else bool(r2p < ref),
                        "n_states": int(S.mask.sum()), "n_episodes": S.E})
    return out


def plot_heat(mat, rows, title, fn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.array(mat, float)
    fig, ax = plt.subplots(figsize=(9.5, 0.32 * len(rows) + 2.2))
    im = ax.imshow(np.clip(m, -1, 1), cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(np.clip(m[i, j], -1, 1)) > 0.6 else "black")
    ax.set_xticks(range(5))
    ax.set_xticklabels(TASKS, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7.5)
    ax.set_xlabel("target: r_tau'(s_t)")
    ax.set_ylabel("task embedding fed to the encoder")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, label="test R^2 (color clipped to [-1, 1]; numbers unclipped)")
    fig.tight_layout()
    fig.savefig(fn, dpi=130)
    plt.close(fig)


def rank(v):
    o = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    r[o] = np.arange(len(v))
    # average ranks for ties
    for x in np.unique(v):
        k = v == x
        if k.sum() > 1:
            r[k] = r[k].mean()
    return r


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return {"n": int(len(x)), "pearson": float(np.corrcoef(x, y)[0, 1]), "spearman": float(np.corrcoef(rank(x), rank(y))[0, 1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lab = np.load(os.path.join(a.data, "split_labels.npy"))
    te = np.where(lab == "test")[0]
    idx = np.load(os.path.join(a.data, "pooled", "index.npz"))
    idx_test = {k: idx[k][te] for k in ("task", "rollout_type", "episode", "t")}
    y_test = np.load(os.path.join(a.data, "pooled", "targets.npy"))[te]
    h_test = np.load(os.path.join(a.data, "pooled", "torso_height.npy"))[te]
    emb = json.load(open(a.emb))["embedding_table"]["rows"]

    # row order (PREREG 2): max cosine to the 5 walker embeddings, descending; zero vector last
    names = list(emb.keys())
    key = lambda n: (-round(max(emb[n]["cos_walker"]), 10) if emb[n]["cos_walker"][0] is not None else 9, names.index(n))
    order = sorted(names, key=key)
    rows_all = order + [REF]
    P = {r: dict(np.load(os.path.join(a.rows, fname(r) + ".npz"))) for r in rows_all}
    M = {r: json.load(open(os.path.join(a.rows, fname(r) + ".json"))) for r in rows_all}

    pooled = Sub(np.ones(len(te), bool), idx_test, y_test)
    stand_m = h_test >= 0.6
    notf = Sub(stand_m, idx_test, y_test)
    recs = []
    recs += evaluate("pooled|linear", pooled, {r: [P[r]["lin_main_test"]] for r in rows_all})
    recs += evaluate("pooled|mlp", pooled, {r: list(P[r]["mlp_test"]) for r in rows_all})
    recs += evaluate("not_fallen|linear", notf, {r: [P[r]["lin_main_test"][stand_m]] for r in rows_all})
    recs += evaluate("not_fallen|mlp", notf, {r: [p[stand_m] for p in P[r]["mlp_test"]] for r in rows_all})
    recs += evaluate("pooled|linear_lambda_ext", pooled, {r: [P[r]["lin_ext_test"]] for r in rows_all})
    sp = dict(np.load(os.path.join(a.rows, "standing.npz")))
    sj = json.load(open(os.path.join(a.rows, "standing.json")))
    assert np.array_equal(sp["test_index"], te[stand_m])
    srows = [k[:-5] for k in sp if k.endswith("|test")]
    recs += evaluate("standing_only|linear", notf, {r: [sp[f"{r}|test"]] for r in srows})

    with open(os.path.join(a.out, "cells.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    get = {(c["setting"], c["row"], c["target"]): c for c in recs}

    nonwalker = [n for n in order if n.startswith("task:") and n[5:] not in TASKS]
    rand = [n for n in order if n.startswith("rand:")]
    summary = {"row_order": order, "nonwalker_rows": nonwalker}
    # counts
    counts = {}
    for s in ("pooled|linear", "pooled|mlp", "not_fallen|linear", "not_fallen|mlp", "pooled|linear_lambda_ext", "standing_only|linear"):
        rows_here = sorted({c["row"] for c in recs if c["setting"] == s})
        cnt = {}
        for grp, members in (("nonwalker25", nonwalker), ("zero", ["zero"]), ("random5", rand)):
            mem = [m for m in members if m in rows_here]
            cnt[grp] = {"n_rows": len(mem), **{t: sum(get[(s, m, t)]["verdict"] == "실질적으로 낮음" for m in mem) for t in TASKS},
                        "rows_low": sorted({m for m in mem for t in TASKS if get[(s, m, t)]["verdict"] == "실질적으로 낮음"}),
                        "rmse_ratio_ge_1_5": {t: sum(get[(s, m, t)]["rmse_ratio_ge_1_5"] for m in mem) for t in TASKS},
                        "below_randinit": {t: sum(bool(get[(s, m, t)]["below_randinit"]) for m in mem) for t in TASKS}}
        counts[s] = cnt
    summary["counts"] = counts
    # (linear, MLP) categories
    cats = {}
    for sub in ("pooled", "not_fallen"):
        tab = {}
        for r in order:
            for t in TASKS:
                l = get[(f"{sub}|linear", r, t)]["verdict"] == "실질적으로 낮음"
                m = get[(f"{sub}|mlp", r, t)]["verdict"] == "실질적으로 낮음"
                tab[f"{r}|{t}"] = {(False, False): "둘 다 아님", (True, False): "linear만", (False, True): "MLP만", (True, True): "둘 다"}[(l, m)]
        cats[sub] = tab
    summary["categories"] = cats
    # verdict changes vs linear pooled main
    ch = {}
    for s in ("pooled|linear_lambda_ext", "standing_only|linear"):
        rows_here = sorted({c["row"] for c in recs if c["setting"] == s})
        ch[s] = [{"row": r, "target": t, "main": get[("pooled|linear", r, t)]["verdict"], "this": get[(s, r, t)]["verdict"]}
                 for r in rows_here if ("pooled|linear", r, TASKS[0]) in get for t in TASKS
                 if get[("pooled|linear", r, t)]["verdict"] != get[(s, r, t)]["verdict"]]
    summary["verdict_changes_vs_pooled_linear_main"] = ch
    summary["standing_selection"] = sj["selection"][:5]
    summary["standing_chosen"] = sj["chosen"]
    summary["standing_n_states"] = sj["n_states"]
    # TV vs R2
    tvx = {r: float(np.mean([M[r]["tv_to_walker"][t]["q50"] for t in TASKS])) for r in order}
    rel = {}
    for p in ("linear", "mlp"):
        y = {r: float(np.mean([get[(f"pooled|{p}", r, t)]["r2"] for t in TASKS])) for r in order}
        rel[p] = {"all36": corr([tvx[r] for r in order], [y[r] for r in order]),
                  "nonwalker25": corr([tvx[r] for r in nonwalker], [y[r] for r in nonwalker]), "y": y}
    summary["tv_x"] = tvx
    summary["tv_vs_r2"] = rel
    # run facts
    summary["lambda_main_at_min_1e-4"] = sum(l == 1e-4 for r in rows_all for l in M[r]["lambda_main"])
    summary["lambda_main_at_max_1e4"] = sum(l == 1e4 for r in rows_all for l in M[r]["lambda_main"])
    summary["lambda_ext_at_min_1e-8"] = sum(abs(l - 1e-8) < 1e-20 for r in rows_all for l in M[r]["lambda_ext"])
    summary["lambda_main_cells"] = 5 * len(rows_all)
    summary["standing_lambda_at_min"] = sum(l == 1e-4 for v in sj["rows"].values() for l in v["lambda"])
    ep = [e for r in rows_all for e in M[r]["mlp_epochs"]]
    summary["mlp_epochs"] = {"n": len(ep), "min": min(ep), "median": float(np.median(ep)), "max": max(ep), "n_200": sum(e == 200 for e in ep),
                             "rows_with_200": sorted({r for r in rows_all if 200 in M[r]["mlp_epochs"]})}
    summary["n_const_dims"] = {r: M[r]["n_const_dims"] for r in rows_all if M[r]["n_const_dims"]}
    summary["subsets"] = {"pooled": {"n_states": int(len(te)), "n_episodes": pooled.E, "target_var": [float(v) for v in y_test.var(0)]},
                          "not_fallen": {"n_states": int(stand_m.sum()), "n_episodes": notf.E, "target_var": [float(v) for v in y_test[stand_m].var(0)]}}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)

    # figures
    lbl = lambda r: r.replace("task:", "").replace("rand:", "random ").replace("zero", "zero vector").replace("randinit:walker-stand", "[ref] random-init model")
    for s, rows, title in (("pooled|linear", rows_all, "linear (ridge), pooled test"), ("pooled|mlp", rows_all, "MLP multi-output (mean of 5 seeds), pooled test"),
                           ("not_fallen|linear", rows_all, "linear, test without fallen states"), ("not_fallen|mlp", rows_all, "MLP, test without fallen states"),
                           ("pooled|linear_lambda_ext", rows_all, "linear, lambda grid 10^[-8:0.5:4], pooled test"),
                           ("standing_only|linear", srows, "standing-only linear probe (train/val/test without fallen states)")):
        mat = [[get[(s, r, t)]["r2"] for t in TASKS] for r in rows]
        plot_heat(mat, [lbl(r) for r in rows], f"Experiment 2: {title}\nrows sorted by max cosine to walker embeddings",
                  os.path.join(a.out, f"heatmap_{s.replace('|', '_')}.png"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, p in zip(axs, ("linear", "mlp")):
        for r in order:
            c = "tab:red" if r[5:] in TASKS else ("tab:gray" if not r.startswith("task:") else "tab:blue")
            ax.scatter(tvx[r], rel[p]["y"][r], c=c, s=18)
            ax.annotate(lbl(r), (tvx[r], rel[p]["y"][r]), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel("mean over 5 columns of median d_TV(z(s, row), z(s, diagonal walker)) on test states")
        ax.set_ylabel("5-column mean test R^2 (pooled)")
        ax.set_title(f"{p}: red = walker, blue = other mt30 tasks, gray = zero/random vectors", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "scatter_tv_vs_r2.png"), dpi=130)
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ("counts", "lambda_main_at_min_1e-4", "lambda_ext_at_min_1e-8", "mlp_epochs")}, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
