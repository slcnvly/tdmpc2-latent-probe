import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
which = sys.argv[2] if len(sys.argv) > 2 else "gate1"
consts = {"PROBE_COMMIT": sys.argv[1], "TDMPC2_COMMIT": "332f19e4aba4bffe2dc736f45ea7f7bdc585903a",
          "SIZES": [1, 5, 19, 48, 317], "COST_STEPS": 200, "EPISODES": 5, "EVAL_SEED": 1, "DEADLINE_H": 10.5}
d = os.path.join(HERE, f"kernel_{which}"); os.makedirs(d, exist_ok=True)
open(os.path.join(d, "run.py"), "w").write("".join(f"{k} = {v!r}\n" for k, v in consts.items()) + open(os.path.join(HERE, f"{which}_body.py")).read())
json.dump({"id": f"seungwonryoo/tdmpc2-probe-{which}", "title": f"TDMPC2 Probe {which.title()}", "code_file": "run.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [],
           "competition_sources": [], "kernel_sources": [], "model_sources": [], **({"dataset_sources": ["seungwonryoo/tdmpc2-probe-exp1-rollouts"]} if which == "exp1_probe" else {})}, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
print(d)
