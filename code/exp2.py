"""Experiment 2 (EXPERIMENT2_PREREG.md): extreme task swap. Encoders frozen; only probes are trained.

Rows: 30 mt30 task embeddings (official WorldModel.encode), zero vector + 5 random directions (inject()),
random-init reference (Exp1 random-init model, task id 0, official encode). One row at a time:
encode (CPU float32) -> ridge (main grid 10^[-4:0.5:4] and extended 10^[-8:0.5:4]) -> multi-output MLP (5 seeds)
-> d_TV to the 5 walker-embedding z on test states -> save predictions -> drop the latent.

  python exp2.py check    --data ../data                                    # inject() == official encode / Exp1 latent
  python exp2.py run      --data DIR --out OUT --device cuda --worker K --num-workers 4
  python exp2.py standing --data DIR --out OUT                              # after all rows (selection rule PREREG 4)
DIR holds pooled/{obs,targets,torso_height}.npy and split_labels.npy.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from common_setup import WALKER_TASKS, load_world_model, make_env_and_cfg
from probe import r2, standardize
from step5_latents_targets import RANDOM_INIT_SEED

LAM_MAIN = 10.0 ** np.arange(-4, 4.01, 0.5)
LAM_EXT = 10.0 ** np.arange(-8, 4.01, 0.5)
RAND_SEED, N_RAND = 20261102, 5
CHECK_IDX = np.arange(0, 150000, 50)
TOL_INJECT = 1e-5


def build():
    env, cfg = make_env_and_cfg(48)
    model = load_world_model(cfg)
    torch.manual_seed(RANDOM_INIT_SEED)
    from common.world_model import WorldModel

    rnd = WorldModel(cfg).eval()
    return cfg, model, rnd


def control_vectors(model):
    W = model._task_emb.weight.detach().numpy().astype(np.float64)
    c30 = float(np.linalg.norm(W, axis=1).mean())
    g = np.random.default_rng(RAND_SEED).standard_normal((N_RAND, W.shape[1]))
    g = g / np.linalg.norm(g, axis=1, keepdims=True) * c30
    return c30, {"zero": np.zeros(W.shape[1], np.float32), **{f"rand:{i}": g[i].astype(np.float32) for i in range(N_RAND)}}


def row_names(cfg):
    return [f"task:{t}" for t in cfg.tasks] + ["zero"] + [f"rand:{i}" for i in range(N_RAND)] + ["randinit:walker-stand"]


@torch.no_grad()
def inject(model, obs, v):
    """Official encode (world_model.py:105-114) minus the embedding lookup: the vector v is concatenated directly."""
    x = torch.as_tensor(obs)
    return model._encoder[model.cfg.obs](torch.cat([x, torch.as_tensor(v).expand(x.shape[0], -1)], dim=-1))


@torch.no_grad()
def encode_row(name, obs, cfg, model, rnd, ctrl, batch=4096):
    out = np.empty((obs.shape[0], cfg.latent_dim), np.float32)
    for s in range(0, obs.shape[0], batch):
        xb = torch.from_numpy(obs[s:s + batch])
        if name.startswith("task:"):
            k = cfg.tasks.index(name[5:])
            out[s:s + batch] = model.encode(xb, torch.full((xb.shape[0],), k, dtype=torch.long)).numpy()
        elif name.startswith("randinit:"):
            k = cfg.tasks.index(name[9:])
            out[s:s + batch] = rnd.encode(xb, torch.full((xb.shape[0],), k, dtype=torch.long)).numpy()
        else:
            out[s:s + batch] = inject(model, xb, ctrl[name]).numpy()
    return out


def tv(a, b):
    """mean over the 96 SimNorm groups of 0.5 * L1 (Experiment 1 definition)."""
    return 0.5 * np.abs(a - b).reshape(a.shape[0], -1, 8).sum(-1).mean(-1)


def load_data(root):
    p = os.path.join(root, "pooled")
    lab = np.load(os.path.join(root, "split_labels.npy"))
    return {"obs": np.load(os.path.join(p, "obs.npy")), "y": np.load(os.path.join(p, "targets.npy")),
            "h": np.load(os.path.join(p, "torso_height.npy")), "idx": {s: np.where(lab == s)[0] for s in ("train", "val", "test")}}


def ridge(Xs, Y, idx, grids):
    """Ridge per target column; one eigendecomposition; selection by val R^2 (ties -> larger lambda) per grid."""
    tr, va, te = idx["train"], idx["val"], idx["test"]
    Xtr = Xs[tr].astype(np.float64)
    evals, V = np.linalg.eigh(Xtr.T @ Xtr)
    Xva, Xte = Xs[va].astype(np.float64), Xs[te].astype(np.float64)
    res = {g: {"test": np.zeros((len(te), Y.shape[1]), np.float32), "val": np.zeros((len(va), Y.shape[1]), np.float32), "lambda": [], "val_r2": []} for g in grids}
    for j in range(Y.shape[1]):
        ym, ys = Y[tr, j].mean(), Y[tr, j].std()
        b = V.T @ (Xtr.T @ ((Y[tr, j] - ym) / ys))
        for g, lams in grids.items():
            best = None
            for lam in lams:  # ascending; '>=' keeps the larger lambda on ties
                w = V @ (b / (evals + lam))
                sc = r2(Y[va, j], (Xva @ w) * ys + ym)
                if best is None or sc >= best[0]:
                    best = (sc, lam, w)
            sc, lam, w = best
            res[g]["test"][:, j] = (Xte @ w) * ys + ym
            res[g]["val"][:, j] = (Xva @ w) * ys + ym
            res[g]["lambda"].append(float(lam))
            res[g]["val_r2"].append(float(sc))
    return res


def mlp_multi(Xs, Y, idx, seed, device, max_epochs=200, patience=20, batch=512, lr=1e-3):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    tr, va, te = idx["train"], idx["val"], idx["test"]
    ym, ys = Y[tr].mean(0), Y[tr].std(0)
    Xtr = torch.from_numpy(Xs[tr]).to(device)
    ytr = torch.from_numpy(((Y[tr] - ym) / ys).astype(np.float32)).to(device)
    Xva = torch.from_numpy(Xs[va]).to(device)
    yva = torch.from_numpy(((Y[va] - ym) / ys).astype(np.float32)).to(device)
    net = torch.nn.Sequential(torch.nn.Linear(Xs.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(),
                              torch.nn.Linear(256, Y.shape[1])).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0)
    best, best_state, bad, ep_run = float("inf"), None, 0, 0
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(len(tr), generator=g).to(device)
        for s in range(0, len(tr), batch):
            b = perm[s:s + batch]
            loss = ((net(Xtr[b]) - ytr[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vm = float(((net(Xva) - yva) ** 2).mean())
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
        pt = net(torch.from_numpy(Xs[te]).to(device)).cpu().numpy() * ys + ym
        pv = net(Xva).cpu().numpy() * ys + ym
    return pt.astype(np.float32), pv.astype(np.float32), ep_run, best


def embedding_table(cfg, model, ctrl, c30):
    W = model._task_emb.weight.detach().numpy().astype(np.float64)
    walk = [W[cfg.tasks.index(t)] for t in WALKER_TASKS]
    rows = {}
    for name in row_names(cfg)[:-1]:
        v = W[cfg.tasks.index(name[5:])] if name.startswith("task:") else ctrl[name].astype(np.float64)
        n = float(np.linalg.norm(v))
        cos = [None if n == 0 else float(v @ w / (n * np.linalg.norm(w))) for w in walk]
        rows[name] = {"l2_norm": n, "cos_walker": cos}
    return {"c30_mean_norm": c30, "random_seed": RAND_SEED, "rows": rows}


def cmd_check(a):
    torch.set_num_threads(os.cpu_count())
    cfg, model, rnd = build()
    D = load_data(a.data)
    obs = D["obs"][CHECK_IDX]
    lat_path = os.path.join(a.data, "pooled", "latent_learned.npy")
    stored = np.load(lat_path, mmap_mode="r") if os.path.exists(lat_path) else None
    res = {"n_states": int(len(CHECK_IDX)), "state_index": "0:150000:50", "tol": TOL_INJECT, "rows": {}}
    for i, t in enumerate(WALKER_TASKS):
        k = cfg.tasks.index(t)
        zi = inject(model, obs, model._task_emb.weight[k].detach()).numpy()
        zo = model.encode(torch.from_numpy(obs), torch.full((len(obs),), k, dtype=torch.long)).detach().numpy()
        r = {"max_abs_inject_vs_official_encode": float(np.abs(zi - zo).max())}
        if stored is not None:
            r["max_abs_inject_vs_exp1_latent_learned"] = float(np.abs(zi - np.asarray(stored[i, CHECK_IDX])).max())
        res["rows"][t] = r
    res["pass"] = all(v <= TOL_INJECT for r in res["rows"].values() for v in r.values())
    c30, ctrl = control_vectors(model)
    res["embedding_table"] = embedding_table(cfg, model, ctrl, c30)
    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    json.dump(res, open(a.out_json, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "embedding_table"}, indent=1))
    if not res["pass"]:
        raise SystemExit("INJECT CHECK FAILED")


def cmd_run(a):
    torch.set_num_threads(a.threads or os.cpu_count())
    cfg, model, rnd = build()
    c30, ctrl = control_vectors(model)
    D = load_data(a.data)
    os.makedirs(a.out, exist_ok=True)
    te = D["idx"]["test"]
    walker_z = [encode_row(f"task:{t}", D["obs"][te], cfg, model, rnd, ctrl) for t in WALKER_TASKS]
    names = [n for i, n in enumerate(a.rows or row_names(cfg)) if i % a.num_workers == a.worker]
    for name in names:
        f = os.path.join(a.out, f"row_{name.replace(':', '__')}.npz")
        if os.path.exists(f):
            continue
        t0 = time.time()
        Z = encode_row(name, D["obs"], cfg, model, rnd, ctrl)
        t_enc = time.time() - t0
        tvs = [tv(Z[te], wz) for wz in walker_z]
        tv_stats = [{"q05": float(np.quantile(d, 0.05)), "q50": float(np.quantile(d, 0.5)), "q95": float(np.quantile(d, 0.95))} for d in tvs]
        Xs, info = standardize(Z, D["idx"]["train"])
        del Z
        t1 = time.time()
        lin = ridge(Xs, D["y"], D["idx"], {"main": LAM_MAIN, "ext": LAM_EXT})
        t_lin = time.time() - t1
        mt, mv, mep, mbest = [], [], [], []
        t2 = time.time()
        for seed in range(5):
            pt, pv, ep, b = mlp_multi(Xs, D["y"], D["idx"], seed, a.device, max_epochs=a.max_epochs)
            mt.append(pt); mv.append(pv); mep.append(ep); mbest.append(b)
        t_mlp = time.time() - t2
        del Xs
        np.savez(f, lin_main_test=lin["main"]["test"], lin_main_val=lin["main"]["val"], lin_ext_test=lin["ext"]["test"],
                 lin_ext_val=lin["ext"]["val"], mlp_test=np.stack(mt), mlp_val=np.stack(mv))
        meta = {"row": name, **info, "lambda_main": lin["main"]["lambda"], "val_r2_main": lin["main"]["val_r2"],
                "lambda_ext": lin["ext"]["lambda"], "val_r2_ext": lin["ext"]["val_r2"], "mlp_epochs": mep, "mlp_best_val_mse_std": mbest,
                "tv_to_walker": dict(zip(WALKER_TASKS, tv_stats)), "seconds": {"encode": t_enc, "linear": t_lin, "mlp": t_mlp},
                "max_epochs": a.max_epochs, "device": a.device}
        json.dump(meta, open(f[:-4] + ".json", "w"), indent=1)
        print(name, f"enc {t_enc:.0f}s lin {t_lin:.0f}s mlp {t_mlp:.0f}s epochs {mep}", flush=True)


def cmd_standing(a):
    torch.set_num_threads(a.threads or os.cpu_count())
    cfg, model, rnd = build()
    c30, ctrl = control_vectors(model)
    D = load_data(a.data)
    te = D["idx"]["test"]
    # selection rule (PREREG 4): 25 non-walker task rows, lowest 5-column mean linear main pooled test R^2, ties -> lower task id
    cand = []
    for k, t in enumerate(cfg.tasks):
        if t in WALKER_TASKS:
            continue
        p = np.load(os.path.join(a.out, f"row_task__{t}.npz"))["lin_main_test"]
        cand.append((float(np.mean([r2(D["y"][te, j], p[:, j]) for j in range(5)])), k, t))
    cand.sort()
    chosen = [t for _, _, t in cand[:3]]
    stand = D["h"] >= 0.6
    idx = {s: v[stand[v]] for s, v in D["idx"].items()}
    rows = [f"task:{t}" for t in WALKER_TASKS] + ["raw"] + [f"randinit:{t}" for t in WALKER_TASKS] + [f"task:{t}" for t in chosen]
    preds, meta = {}, {"selection": [{"task": t, "task_id": k, "mean_r2_lin_main_pooled": m} for m, k, t in cand],
                       "chosen": chosen, "n_states": {s: int(len(v)) for s, v in idx.items()}, "rows": {}}
    for name in rows:
        Z = D["obs"].astype(np.float32) if name == "raw" else encode_row(name, D["obs"], cfg, model, rnd, ctrl)
        Xs, info = standardize(Z, idx["train"])
        lin = ridge(Xs, D["y"], idx, {"main": LAM_MAIN})
        preds[f"{name}|test"] = lin["main"]["test"]
        preds[f"{name}|val"] = lin["main"]["val"]
        meta["rows"][name] = {**info, "lambda": lin["main"]["lambda"], "val_r2": lin["main"]["val_r2"]}
        print(name, lin["main"]["lambda"], flush=True)
    np.savez(os.path.join(a.out, "standing.npz"), test_index=idx["test"], **preds)
    json.dump(meta, open(os.path.join(a.out, "standing.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("check", "run", "standing"):
        s = sub.add_parser(c)
        s.add_argument("--data", required=True)
        s.add_argument("--out", default="")
        s.add_argument("--out-json", default="../results/exp2/inject_check_local.json")
        s.add_argument("--device", default="cpu")
        s.add_argument("--threads", type=int, default=0)
        s.add_argument("--worker", type=int, default=0)
        s.add_argument("--num-workers", type=int, default=1)
        s.add_argument("--rows", nargs="*")
        s.add_argument("--max-epochs", type=int, default=200)
    a = ap.parse_args()
    {"check": cmd_check, "run": cmd_run, "standing": cmd_standing}[a.cmd](a)


if __name__ == "__main__":
    main()
