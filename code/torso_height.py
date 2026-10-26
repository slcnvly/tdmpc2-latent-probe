"""Torso height (m) of every pooled state s_t, by restoring s_t (physics.torso_height(), dm_control/suite/walker.py:88-90).
Used for the fallen definition (torso_height < 0.6 m, EXPERIMENT1_PREREG.md 2).
  python torso_height.py --pooled ../data/pooled
"""
import argparse
import os

import numpy as np

from collect_rollouts import raw_dm_env
from common_setup import make_env_and_cfg

ap = argparse.ArgumentParser()
ap.add_argument("--pooled", required=True)
a = ap.parse_args()
env, cfg = make_env_and_cfg(48)
dm = raw_dm_env(env, cfg.tasks.index("walker-stand"))
state = np.load(os.path.join(a.pooled, "state.npy"))
h = np.empty(len(state))
for n, s in enumerate(state):
    dm.physics.set_state(s); dm.physics.forward()
    h[n] = dm.physics.torso_height()
np.save(os.path.join(a.pooled, "torso_height.npy"), h)
print("fallen (<0.6 m) fraction", float((h < 0.6).mean()))
