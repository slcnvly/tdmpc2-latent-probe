"""Step 6 reward probes (EXPERIMENT1_PREREG.md 5 + addendum A1). Encoders are frozen; only probes are trained.

Inputs (each probe independent): learned z(s, tau) (5 rows), raw observation o(s) (1), random-init z (5 rows),
norm-matched random-init z (5 rows, auxiliary). Targets: r_tau'(s_t) for the 5 walker tasks (columns).
Linear: ridge, lambda in 10^[-4:0.5:4] chosen by validation R^2 (ties -> larger lambda), fit on train only.
MLP: in -> 256 -> ReLU -> 256 -> ReLU -> 1, Adam lr 1e-3, wd 0, batch 512, max 200 epochs, early stop on val MSE
     (patience 20, restore best), seeds 0-4 (init + batch order).
Features and target standardized with train statistics (feature std < 1e-8 -> divide by 1).
Saves test and val predictions (original target units) to <out>/pred_<kind>.npz.

  python probe.py cost  --data ../data --out ../results/exp1/probe_cost.json
  python probe.py linear --data ../data --out ../results/exp1/probes
  python probe.py mlp   --data ../data --out ../results/exp1/probes --device cuda --inputs learned:1 ...
"""
import argparse
import json
import os
import time

import numpy as np
import torch

TASKS = ["walker-stand", "walker-walk", "walker-run", "walker-walk-backwards", "walker-run-backwards"]
LAMBDAS = 10.0 ** np.arange(-4, 4.01, 0.5)
LAT_FILES = {"learned": "latent_learned.npy", "random": "latent_random.npy", "normmatched": "latent_normmatched.npy"}


def all_inputs():
    out = [f"learned:{k}" for k in range(5)] + ["raw"] + [f"random:{k}" for k in range(5)] + [f"normmatched:{k}" for k in range(5)]
    return out


class Data:
    def __init__(self, root):
        p = os.path.join(root, "pooled")
        self.lab = np.load(os.path.join(root, "split_labels.npy"))
        self.y = np.load(os.path.join(p, "targets.npy"))
        self.obs = np.load(os.path.join(p, "obs.npy"))
        self.lat = {k: np.load(os.path.join(p, f), mmap_mode="r") for k, f in LAT_FILES.items()}
        self.idx = {s: np.where(self.lab == s)[0] for s in ("train", "val", "test")}

    def X(self, name):
        if name == "raw":
            return np.asarray(self.obs, dtype=np.float32)
        kind, k = name.split(":")
        return np.asarray(self.lat[kind][int(k)], dtype=np.float32)


def standardize(X, tr):
    mu = X[tr].mean(0, dtype=np.float64)
    sd = X[tr].std(0, dtype=np.float64)
    sd[sd < 1e-8] = 1.0
    return ((X - mu) / sd).astype(np.float32), {"n_const_dims": int((X[tr].std(0) < 1e-8).sum())}


def r2(y, p):
    return 1.0 - float(((y - p) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def ridge_all_targets(Xs, Y, idx):
    """Ridge for all 5 target columns; one eigendecomposition of X_tr^T X_tr per input."""
    tr, va = idx["train"], idx["val"]
    Xtr = Xs[tr].astype(np.float64)
    G = Xtr.T @ Xtr
    evals, V = np.linalg.eigh(G)
    out = {}
    for j in range(Y.shape[1]):
        ym, ys = Y[tr, j].mean(), Y[tr, j].std()
        yt = (Y[tr, j] - ym) / ys
        b = V.T @ (Xtr.T @ yt)
        best = None
        for lam in LAMBDAS:  # ascending; '>=' keeps the larger lambda on ties
            w = V @ (b / (evals + lam))
            pv = (Xs[va].astype(np.float64) @ w) * ys + ym
            score = r2(Y[va, j], pv)
            if best is None or score >= best[0]:
                best = (score, lam, w)
        _, lam, w = best
        out[j] = {"lambda": float(lam), "val_r2": float(best[0]),
                  "pred_test": ((Xs[idx["test"]].astype(np.float64) @ w) * ys + ym).astype(np.float32),
                  "pred_val": ((Xs[va].astype(np.float64) @ w) * ys + ym).astype(np.float32)}
    return out


def train_mlp(Xs, y, idx, seed, device, max_epochs=200, patience=20, batch=512, lr=1e-3):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    tr, va, te = idx["train"], idx["val"], idx["test"]
    ym, ys = float(y[tr].mean()), float(y[tr].std())
    Xtr = torch.from_numpy(Xs[tr]).to(device)
    ytr = torch.from_numpy(((y[tr] - ym) / ys).astype(np.float32)).to(device)
    Xva = torch.from_numpy(Xs[va]).to(device)
    yva = torch.from_numpy(((y[va] - ym) / ys).astype(np.float32)).to(device)
    d = Xs.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0)
    best, best_state, bad, ep_run = float("inf"), None, 0, 0
    n = len(tr)
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(n, generator=g).to(device)
        for s in range(0, n, batch):
            b = perm[s:s + batch]
            loss = ((net(Xtr[b]).squeeze(-1) - ytr[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vm = float(((net(Xva).squeeze(-1) - yva) ** 2).mean())
        ep_run = ep + 1
        if vm < best:
            best, bad = vm, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pt = net(torch.from_numpy(Xs[te]).to(device)).squeeze(-1).cpu().numpy() * ys + ym
        pv = net(Xva).squeeze(-1).cpu().numpy() * ys + ym
    return {"pred_test": pt.astype(np.float32), "pred_val": pv.astype(np.float32), "epochs_run": ep_run, "best_val_mse_std_units": best}


def cmd_cost(a):
    D = Data(a.data)
    t = time.time()
    X = D.X("learned:1")
    Xs, info = standardize(X, D.idx["train"])
    t_load = time.time() - t
    t = time.time()
    lin = ridge_all_targets(Xs, D.y[:, [1]], D.idx)
    t_lin = time.time() - t
    torch.set_num_threads(os.cpu_count())
    t = time.time()
    m = train_mlp(Xs, D.y[:, 1], D.idx, 0, a.device)
    t_mlp = time.time() - t
    res = {"cell": "learned z (row walker-walk) -> target walker-walk", "device": a.device, "threads": torch.get_num_threads(),
           "load_standardize_s": t_load, "linear_one_target_s": t_lin, "mlp_one_seed_s": t_mlp, "mlp_epochs_run": m["epochs_run"],
           "trial_linear_test_r2": r2(D.y[D.idx["test"], 1], lin[0]["pred_test"]), "trial_mlp_test_r2": r2(D.y[D.idx["test"], 1], m["pred_test"]),
           "note": "cost-measurement trial only; final results are recomputed in the full run"}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


def cmd_linear(a):
    D = Data(a.data)
    os.makedirs(a.out, exist_ok=True)
    res, preds = {}, {}
    t0 = time.time()
    for name in (a.inputs or all_inputs()):
        Xs, info = standardize(D.X(name), D.idx["train"])
        out = ridge_all_targets(Xs, D.y, D.idx)
        for j in range(5):
            key = f"{name}|{j}"
            preds[f"{key}|test"] = out[j]["pred_test"]
            preds[f"{key}|val"] = out[j]["pred_val"]
            res[key] = {"lambda": out[j]["lambda"], "val_r2": out[j]["val_r2"], **info}
        print(name, {j: (out[j]["lambda"], round(out[j]["val_r2"], 4)) for j in range(5)}, f"{time.time() - t0:.0f}s", flush=True)
    np.savez(os.path.join(a.out, "pred_linear.npz"), **preds)
    json.dump({"cells": res, "seconds": time.time() - t0}, open(os.path.join(a.out, "linear_meta.json"), "w"), indent=1)


def cmd_mlp(a):
    D = Data(a.data)
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(a.threads or os.cpu_count())
    tag = a.tag or "all"
    pfile = os.path.join(a.out, f"pred_mlp_{tag}.npz")
    mfile = os.path.join(a.out, f"mlp_meta_{tag}.json")
    preds = dict(np.load(pfile)) if os.path.exists(pfile) else {}
    meta = json.load(open(mfile)) if os.path.exists(mfile) else {}
    t0 = time.time()
    jobs = [(name, j, seed) for name in (a.inputs or all_inputs()) for j in range(5) for seed in range(5)]
    jobs = [x for i, x in enumerate(jobs) if i % a.num_workers == a.worker]
    cache = {}
    for name, j, seed in jobs:
        if a.deadline and time.time() > a.deadline:
            print("deadline reached", flush=True)
            break
        if name not in cache:
            cache.clear()
            cache[name] = standardize(D.X(name), D.idx["train"])[0]
        Xs = cache[name]
        if True:
            if True:
                key = f"{name}|{j}|{seed}"
                if f"{key}|test" in preds:
                    continue
                t = time.time()
                m = train_mlp(Xs, D.y[:, j], D.idx, seed, a.device)
                preds[f"{key}|test"], preds[f"{key}|val"] = m["pred_test"], m["pred_val"]
                meta[key] = {"epochs_run": m["epochs_run"], "best_val_mse_std_units": m["best_val_mse_std_units"], "seconds": time.time() - t}
                np.savez(pfile, **preds)  # resumable
                json.dump(meta, open(mfile, "w"), indent=1)
                print(key, f"epochs {m['epochs_run']} {meta[key]['seconds']:.0f}s total {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("cost", "linear", "mlp"):
        s = sub.add_parser(c)
        s.add_argument("--data", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--device", default="cpu")
        s.add_argument("--inputs", nargs="*")
        s.add_argument("--threads", type=int, default=0)
        s.add_argument("--tag", default="")
        s.add_argument("--worker", type=int, default=0)
        s.add_argument("--num-workers", type=int, default=1)
        s.add_argument("--deadline", type=float, default=0)
    a = ap.parse_args()
    {"cost": cmd_cost, "linear": cmd_linear, "mlp": cmd_mlp}[a.cmd](a)


if __name__ == "__main__":
    main()
