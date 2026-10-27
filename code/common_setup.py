"""Shared setup: build the official TD-MPC2 config / env / world model without modifying the repo.

The official parse_cfg() calls hydra.utils.get_original_cwd(), which only works inside a @hydra.main
app; we compose the same config.yaml with Hydra's compose API and patch that one helper to os.getcwd.
Nothing in the tdmpc2 repository is edited.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TDMPC2_DIR = os.path.join(ROOT, "tdmpc2", "tdmpc2")
CKPT_DIR = os.path.join(ROOT, "ckpts", "multitask")
sys.path.insert(0, TDMPC2_DIR)
os.environ.setdefault("MUJOCO_GL", "egl")

WALKER_TASKS = ["walker-stand", "walker-walk", "walker-run", "walker-walk-backwards", "walker-run-backwards"]


def make_cfg(model_size, task="mt30", **overrides):
    import hydra
    from hydra import compose, initialize_config_dir

    from common.parser import parse_cfg

    hydra.utils.get_original_cwd = os.getcwd  # parse_cfg uses it only to build cfg.work_dir
    ov = [f"task={task}", f"model_size={model_size}", f"checkpoint={os.path.join(CKPT_DIR, f'{task}-{model_size}M.pt')}"]
    ov += [f"{k}={v}" for k, v in overrides.items()]
    with initialize_config_dir(config_dir=TDMPC2_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=ov)
    return parse_cfg(cfg)


def make_env_and_cfg(model_size, **overrides):
    """Official make_env (envs/__init__.py): builds all tasks of the set and fills obs/action dims into cfg."""
    from envs import make_env

    cfg = make_cfg(model_size, **overrides)
    env = make_env(cfg)
    return env, cfg


def load_world_model(cfg, device="cpu", checkpoint=None):
    """Same load path as TDMPC2.load (tdmpc2.py:81-95) but without constructing the CUDA-bound agent."""
    import torch

    from common.layers import api_model_conversion
    from common.world_model import WorldModel

    model = WorldModel(cfg).to(device)
    sd = torch.load(checkpoint or cfg.checkpoint, map_location=device, weights_only=False)
    sd = sd["model"] if "model" in sd else sd
    sd = api_model_conversion(model.state_dict(), sd)
    model.load_state_dict(sd)
    model.eval()
    return model


def patch_timeout_setter():
    """Meta-World / ManiSkill / MyoSuite make_env (envs/metaworld.py:51) assign env.max_episode_steps, but Timeout
    (envs/wrappers/timeout.py:13-15, since commit 6117bc4) defines it as a read-only property -> AttributeError at 332f19e.
    This adds a setter that only accepts the value already stored (the assignment there re-assigns _max_episode_steps),
    so behaviour is unchanged. Used for Experiment 3 audit only (EXPERIMENT3_AUDIT.md); the repo itself is not edited."""
    from envs.wrappers.timeout import Timeout

    def _set(self, v):
        if v != self._max_episode_steps:
            raise ValueError(f"max_episode_steps {v} != {self._max_episode_steps}")

    Timeout.max_episode_steps = property(lambda self: self._max_episode_steps, _set)
