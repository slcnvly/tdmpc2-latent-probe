"""EXPERIMENT3_AUDIT.md items 2-7 (local CPU, Meta-World venv). No training, no planning.

  source mwenv.sh && ../venv_mw/bin/python exp3_audit_local.py --out ../results/exp3_audit/local_audit.json
"""
import argparse
import hashlib
import json
import os

import numpy as np

from common_setup import CKPT_DIR, load_world_model, make_cfg, patch_timeout_setter

MW = ["mw-reach", "mw-push", "mw-pick-place"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def unwrap_mw(e):
    while not hasattr(e, "_get_curr_obs_combined_no_goal"):
        e = e.env
    return e


def model_summary(m):
    """Compiled MjModel contents that define the scene (bodies, geoms, sites, joints)."""
    out = {"nbody": m.nbody, "ngeom": m.ngeom, "nsite": m.nsite, "njnt": m.njnt, "nq": m.nq, "nv": m.nv, "nu": m.nu,
           "bodies": [m.body_id2name(i) for i in range(m.nbody)], "geoms": [], "sites": []}
    for i in range(m.ngeom):
        out["geoms"].append({"name": m.geom_id2name(i), "type": int(m.geom_type[i]), "size": np.round(m.geom_size[i], 6).tolist(),
                             "body": int(m.geom_bodyid[i]), "contype": int(m.geom_contype[i]), "rgba": np.round(m.geom_rgba[i], 4).tolist()})
    for i in range(m.nsite):
        out["sites"].append({"name": m.site_id2name(i), "pos": np.round(m.site_pos[i], 6).tolist(), "rgba": np.round(m.site_rgba[i], 4).tolist()})
    out["body_mass"] = np.round(m.body_mass, 6).tolist()
    return out


def diff_models(a, b):
    d = []
    for k in ("nbody", "ngeom", "nsite", "njnt", "nq", "nv", "nu", "bodies", "body_mass"):
        if a[k] != b[k]:
            d.append({"field": k, "a": a[k], "b": b[k]})
    for k in ("geoms", "sites"):
        for x, y in zip(a[k], b[k]):
            if x != y:
                d.append({"field": k, "a": x, "b": y})
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    import torch

    from common import TASK_SET
    from common.seed import set_seed
    from envs import make_env

    patch_timeout_setter()  # see common_setup.patch_timeout_setter (official make_env fails without it at 332f19e)
    res = {"timeout_setter_patch": True}
    ck = os.path.join(CKPT_DIR, "mt80-48M.pt")
    cfg = make_cfg(48, task="mt80", checkpoint=ck)
    set_seed(cfg.seed)  # evaluate.py:45, before make_env
    env = make_env(cfg)
    model = load_world_model(cfg, checkpoint=ck)
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    sd = sd["model"] if "model" in sd else sd
    res["item2_checkpoint"] = {
        "file": "mt80-48M.pt", "sha256": sha256(ck), "cfg_latent_dim": cfg.latent_dim, "cfg_task_dim": cfg.task_dim,
        "parser_rule": "common/parser.py:75 task_dim = 96 if task == 'mt80' ...; config.yaml:60 latent_dim 512, MODEL_SIZE[48] overrides",
        "ckpt_task_emb_shape": list(sd["_task_emb.weight"].shape),
        "ckpt_encoder_in_features": int(sd["_encoder.state.0.weight"].shape[1]),
        "ckpt_encoder_out_features": int(sd[[k for k in sd if k.startswith("_encoder.state.3") and k.endswith("weight") and sd[k].ndim == 2][0]].shape[0]),
        "obs_shape": list(cfg.obs_shape["state"]), "action_dim": cfg.action_dim, "n_tasks": len(cfg.tasks),
        "obs_dims_per_task": dict(zip(cfg.tasks, env._obs_dims)), "action_dims_per_task": dict(zip(cfg.tasks, env._action_dims)),
        "episode_lengths_per_task": dict(zip(cfg.tasks, env._episode_lengths)),
        "load_world_model_ok": True,
        "task_emb_norms": {t: float(model._task_emb.weight[cfg.tasks.index(t)].norm()) for t in MW},
    }
    # item 3: task ids by three paths
    ids = {}
    for t in MW:
        k_set = TASK_SET["mt80"].index(t)
        k_cfg = cfg.tasks.index(t)
        e = unwrap_mw(env.envs[k_cfg])
        ids[t] = {"a_TASK_SET_index": k_set, "b_cfg_tasks_index_and_env_class": [k_cfg, type(e).__name__, os.path.basename(e.model_name)],
                  "c_task_emb_rows": int(sd["_task_emb.weight"].shape[0]), "same": k_set == k_cfg}
    res["item3_task_ids"] = ids
    # item 4: scene identity on the compiled models
    summ = {t: model_summary(unwrap_mw(env.envs[cfg.tasks.index(t)]).sim.model) for t in MW}
    res["item4_scene"] = {"reach_vs_push": diff_models(summ["mw-reach"], summ["mw-push"]),
                          "reach_vs_pick_place": diff_models(summ["mw-reach"], summ["mw-pick-place"]),
                          "counts": {t: {k: summ[t][k] for k in ("nbody", "ngeom", "nsite", "njnt", "nq")} for t in MW},
                          "object_bodies": {t: [b for b in summ[t]["bodies"] if b and b.startswith("obj")] for t in MW}}
    # items 5 and 6: observation layout check and reset randomization through the TD-MPC2 path (env.reset(task_idx))
    lay, rnd = {}, {}
    for t in MW:
        k = cfg.tasks.index(t)
        e = unwrap_mw(env.envs[k])
        recs, checks = [], []
        for ep in range(10):
            obs = env.reset(task_idx=k).numpy().astype(np.float64)  # MetaWorldWrapper.reset: obs of reset(), then one zero-action step
            o_after = e._get_obs()  # current state after the wrapper's extra step (also advances _prev_obs: check only)
            recs.append({"obj_pos_obs": obs[4:7].tolist(), "obj_init_pos": np.asarray(e.obj_init_pos, float).tolist(),
                         "goal_obs": obs[36:39].tolist(), "target_pos": np.asarray(e._target_pos, float).tolist(), "hand_obs": obs[0:3].tolist(),
                         "max_abs_obs_returned_vs_after_extra_step": float(np.abs(obs[:39] - o_after).max())})
            if ep == 0:
                act = np.zeros(cfg.action_dim, np.float32)
                o1, r, d, info = env.step(torch.from_numpy(act))
                o1 = o1.numpy().astype(np.float64)
                checks = {"hand_eq_endeff": float(np.abs(o1[0:3] - e.get_endeff_pos()).max()),
                          "obj_eq_body_com": float(np.abs(o1[4:7] - e.get_body_com("obj")).max()),
                          "pad_11_18_all_zero": bool(np.all(o1[11:18] == 0)),
                          "goal_eq_target_pos": float(np.abs(o1[36:39] - e._target_pos).max()),
                          "partially_observable": bool(e._partially_observable), "freeze_rand_vec": bool(e._freeze_rand_vec),
                          "isV2": bool(e.isV2), "frame_skip": int(e.frame_skip), "max_path_length": int(e.max_path_length),
                          "obs_len": int(len(o1)), "gripper_obs_3": float(o1[3])}
        P = lambda key: np.array([r[key] for r in recs])
        rnd[t] = {"n_resets": 10, "obj_start_std_m_xyz": P("obj_pos_obs").std(0).tolist(), "goal_std_m_xyz": P("goal_obs").std(0).tolist(),
                  "hand_start_std_m_xyz": P("hand_obs").std(0).tolist(), "obj_start_min": P("obj_pos_obs").min(0).tolist(),
                  "obj_start_max": P("obj_pos_obs").max(0).tolist(), "goal_min": P("goal_obs").min(0).tolist(), "goal_max": P("goal_obs").max(0).tolist(),
                  "max_abs_obs_returned_vs_after_extra_step": max(r["max_abs_obs_returned_vs_after_extra_step"] for r in recs), "resets": recs}
        lay[t] = checks
    res["item5_obs_layout_checks"] = lay
    res["item6_reset_randomization"] = rnd
    # item 7: reward dependence on the object position (perturbation: move the object, re-evaluate the same obs/tcp)
    dep = {}
    for t in MW:
        k = cfg.tasks.index(t)
        e = unwrap_mw(env.envs[k])
        env.reset(task_idx=k)
        o = e._get_obs()
        r0, _ = e.evaluate_state(o, np.zeros(4))
        o2 = o.copy()
        o2[4:7] = o[4:7] + np.array([0.05, 0.05, 0.0])  # object 5 cm away in x and y, hand and target unchanged
        r1, _ = e.evaluate_state(o2, np.zeros(4))
        dep[t] = {"reward_obs": float(r0), "reward_obj_moved_5cm_xy": float(r1), "changed": bool(abs(r1 - r0) > 1e-12)}
    res["item7_reward_obj_dependence"] = dep
    res["item7_lengths"] = {"timeout_agent_steps": 100, "action_repeat": 2, "frame_skip": int(unwrap_mw(env.envs[cfg.tasks.index("mw-reach")]).frame_skip),
                            "source": "envs/metaworld.py:22-28 (2 env.step per agent step), :50 Timeout(max_episode_steps=100)"}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("item6_reset_randomization",)}, indent=1)[:6000])
    print(json.dumps({t: {k: v for k, v in r.items() if k != "resets"} for t, r in rnd.items()}, indent=1))


if __name__ == "__main__":
    main()
