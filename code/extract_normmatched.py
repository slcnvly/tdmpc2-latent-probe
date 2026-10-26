"""Auxiliary control A1 (EXPERIMENT1_PREREG.md addendum): norm-matched random-init latents.
Same random-init model as step5 (seed 20261028), walker task-embedding rows rescaled to norm c = 0.994675
(mean norm of the learned mt30-48M walker rows); direction kept. Extraction = step5's encode_all.
Path check: the unmodified random-init model re-encoded with the same function must reproduce latent_random.npy.
  python extract_normmatched.py --pooled ../data/pooled
"""
import argparse
import json
import os

import numpy as np
import torch

from common_setup import ROOT, WALKER_TASKS, load_world_model, make_env_and_cfg
from step5_latents_targets import RANDOM_INIT_SEED, encode_all

ap = argparse.ArgumentParser()
ap.add_argument("--pooled", required=True)
a = ap.parse_args()
torch.set_num_threads(os.cpu_count())
env, cfg = make_env_and_cfg(48)
ids = [cfg.tasks.index(t) for t in WALKER_TASKS]
learned = load_world_model(cfg)
c = float(np.mean([float(learned._task_emb.weight[k].norm()) for k in ids]))
obs = np.load(os.path.join(a.pooled, "obs.npy"))
from common.world_model import WorldModel

torch.manual_seed(RANDOM_INIT_SEED)
rnd = WorldModel(cfg).eval()
# path check on a subset: unmodified random model through encode_all == stored latent_random
sub = np.arange(0, len(obs), 50)
tmp = os.path.join(a.pooled, "_check_random_subset.npy")
chk = encode_all(rnd, obs[sub], ids, tmp)
stored = np.load(os.path.join(a.pooled, "latent_random.npy"), mmap_mode="r")
path_diff = float(np.abs(np.asarray(chk) - np.asarray(stored[:, sub])).max())
del chk
os.remove(tmp)
before = [float(rnd._task_emb.weight[k].norm()) for k in ids]
with torch.no_grad():
    for k in ids:
        rnd._task_emb.weight[k] *= c / rnd._task_emb.weight[k].norm()
after = [float(rnd._task_emb.weight[k].norm()) for k in ids]
lookup = [float(rnd._task_emb(torch.tensor([k])).norm()) for k in ids]  # norm seen by encode (max_norm=1 not triggered)
encode_all(rnd, obs, ids, os.path.join(a.pooled, "latent_normmatched.npy"))
meta = {"c_target_norm": c, "random_norms_before": before, "norms_after": after, "norms_at_lookup": lookup,
        "path_check_max_abs_vs_latent_random_subset": path_diff, "subset_size": int(len(sub)), "random_init_seed": RANDOM_INIT_SEED}
json.dump(meta, open(os.path.join(ROOT, "results", "exp1", "normmatched_meta.json"), "w"), indent=1)
print(json.dumps(meta, indent=1))
