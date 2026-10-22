"""CPU checks behind BASELINE_AUDIT.md (items 1-8). Writes ../results/audit_checks.json.

  ../venv/bin/python audit_checks.py
"""
import json
import os

import numpy as np
import torch

from common_setup import ROOT, WALKER_TASKS, load_world_model, make_env_and_cfg

OUT = os.path.join(ROOT, "results", "audit_checks.json")
res = {}

env, cfg = make_env_and_cfg(5)
res["mt30_tasks"] = list(cfg.tasks)
res["obs_dims_per_task"] = dict(zip(cfg.tasks, env._obs_dims))
res["action_dims_per_task"] = dict(zip(cfg.tasks, env._action_dims))
res["episode_lengths_per_task"] = dict(zip(cfg.tasks, env._episode_lengths))
res["padded_obs_dim"] = env._obs_shape[0]
res["padded_action_dim"] = env._action_dim
res["walker_task_ids"] = {t: cfg.tasks.index(t) for t in WALKER_TASKS}

# --- item 1: padding; item 7: state save/restore and observation / reward recomputation
checks = {}
for t in WALKER_TASKS:
    k = cfg.tasks.index(t)
    obs = env.reset(task_idx=k)
    dm = env.env.env.env.unwrapped if hasattr(env.env, "env") else None
    base = env.envs[k]  # Timeout -> DMControlWrapper -> action_scale.Wrapper -> control.Environment
    dmenv = base.env.env.env if hasattr(base.env, "env") else None
    # walk down the wrapper chain to the dm_control Environment
    e = env.envs[k]
    chain = []
    while True:
        chain.append(type(e).__name__)
        if hasattr(e, "physics") and hasattr(e, "task"):
            break
        e = getattr(e, "env")
    phys, task = e.physics, e.task
    rng = np.random.default_rng(0)
    max_obs_err, max_rew_err, max_rew2_err = 0.0, 0.0, 0.0
    obs_list = []
    for step in range(50):
        a = torch.from_numpy(rng.uniform(-1, 1, size=env._action_dim).astype(np.float32))
        state = phys.get_state().copy()
        obs_next, reward, done, info = env.step(a)
        # substep rewards: the wrapper repeats the action twice (dmcontrol.py:57-59) and sums rewards
        # recompute from the saved state: restore, then step twice with the same scaled action
        st_after = phys.get_state().copy()
        r_last = task.get_reward(phys)
        ob_now = np.concatenate([v.flatten() for v in task.get_observation(phys).values()]).astype(np.float32)
        max_obs_err = max(max_obs_err, float(np.abs(ob_now - obs_next[:len(ob_now)].numpy()).max()))
        # restore and recompute observation and reward at the restored state
        phys.set_state(st_after)
        phys.forward()
        ob_re = np.concatenate([v.flatten() for v in task.get_observation(phys).values()]).astype(np.float32)
        max_rew_err = max(max_rew_err, abs(float(task.get_reward(phys)) - float(r_last)))
        max_obs_err = max(max_obs_err, float(np.abs(ob_re - obs_next[:len(ob_re)].numpy()).max()))
        obs_list.append(obs_next.numpy())
    padded_tail = np.array(obs_list)[:, env._obs_dims[k]:]
    checks[t] = {"wrapper_chain": chain, "obs_dim": env._obs_dims[k], "padding_len": env._obs_shape[0] - env._obs_dims[k],
                 "padding_all_zero": bool(np.all(padded_tail == 0)) if padded_tail.size else None,
                 "max_abs_obs_err_after_restore": max_obs_err, "max_abs_reward_err_after_restore": max_rew_err,
                 "observation_keys": {kk: list(np.shape(v)) for kk, v in task.get_observation(phys).items()},
                 "task_class": type(task).__name__, "move_speed": getattr(task, "_move_speed", None)}
res["walker_state_restore_checks"] = checks

# --- does the wrapper-step reward equal the sum of the two substep rewards, and r(s') of the final substep?
k = cfg.tasks.index("walker-walk")
env.reset(task_idx=k)
e = env.envs[k]
while not (hasattr(e, "physics") and hasattr(e, "task")):
    e = e.env
phys, task = e.physics, e.task
rows = []
for step in range(20):
    a = np.random.default_rng(step).uniform(-1, 1, size=6).astype(np.float32)
    s0 = phys.get_state().copy()
    _, r_wrap, _, _ = env.step(torch.from_numpy(a))
    s2 = phys.get_state().copy()
    # replay the two substeps from s0 on the raw dm_control env (action already in [-1, 1]; action_scale maps to bounds)
    phys.set_state(s0); phys.forward()
    sub = []
    for _ in range(2):
        ts = e.step(np.asarray(a, dtype=np.float64) * 1.0)  # walker actuator bounds are [-1, 1], scale is identity
        sub.append(float(ts.reward))
    rows.append({"r_wrapper": float(r_wrap), "sum_substeps": sum(sub), "r_last_substep": sub[1],
                 "state_after_replay_equal": bool(np.allclose(phys.get_state(), s2))})
res["walker_walk_reward_composition"] = {
    "max_abs(r_wrapper - sum_substeps)": max(abs(r["r_wrapper"] - r["sum_substeps"]) for r in rows),
    "all_replayed_states_equal": all(r["state_after_replay_equal"] for r in rows),
    "example": rows[:3],
    "actuator_ctrlrange": phys.model.actuator_ctrlrange.tolist(),
}

# --- items 2-4: task embedding, encoder structure, official encode
arch = {}
for size in (1, 5, 19, 48):
    _, c = make_env_and_cfg(size)
    m = load_world_model(c)
    emb = m._task_emb.weight.detach()
    enc = m._encoder["state"]
    walker_ids = [c.tasks.index(t) for t in WALKER_TASKS]
    obs = torch.randn(4, c.obs_shape["state"][0])
    z = {t: m.encode(obs, torch.full((4,), c.tasks.index(t), dtype=torch.long)) for t in WALKER_TASKS}
    zt = z["walker-walk"]
    groups = zt.view(4, -1, c.simnorm_dim)
    # re-implementation: concat embedding (after max_norm renorm done by nn.Embedding lookup) then the encoder MLP
    e_w = m._task_emb(torch.full((4,), c.tasks.index("walker-walk"), dtype=torch.long))
    z_manual = enc(torch.cat([obs, e_w], -1))
    arch[f"{size}M"] = {
        "task_dim": c.task_dim, "latent_dim": c.latent_dim, "enc_dim": c.enc_dim, "num_enc_layers": c.num_enc_layers,
        "mlp_dim": c.mlp_dim, "num_q": c.num_q, "simnorm_dim": c.simnorm_dim,
        "encoder_repr": repr(enc), "encoder_first_layer_in_features": enc[0].in_features,
        "n_params_total": int(sum(p.numel() for p in m.parameters())), "n_params_encoder": int(sum(p.numel() for p in enc.parameters())),
        "task_emb_num": int(emb.shape[0]), "task_emb_max_norm": m._task_emb.max_norm,
        "task_emb_norms_walker": {t: float(emb[c.tasks.index(t)].norm()) for t in WALKER_TASKS},
        "task_emb_norm_min_max_all": [float(emb.norm(dim=1).min()), float(emb.norm(dim=1).max())],
        "simnorm_group_sums_close_to_1": bool(torch.allclose(groups.sum(-1), torch.ones_like(groups.sum(-1)), atol=1e-5)),
        "encode_differs_across_walker_tasks": float((z["walker-walk"] - z["walker-run"]).abs().max()),
        "manual_reimplementation_max_abs_diff": float((z_manual - zt).abs().max()),
        "pairwise_cos_walker_embeddings": [[round(float(torch.nn.functional.cosine_similarity(emb[i], emb[j], dim=0)), 4) for j in walker_ids] for i in walker_ids],
    }
res["architecture"] = arch
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k != "architecture"}, indent=1, default=str)[:6000])
for s, a in arch.items():
    print(s, {k: v for k, v in a.items() if k not in ("encoder_repr", "pairwise_cos_walker_embeddings")})
print(arch["5M"]["encoder_repr"]); print(arch["5M"]["pairwise_cos_walker_embeddings"])
