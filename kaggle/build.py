import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
consts = {"PROBE_COMMIT": sys.argv[1], "TDMPC2_COMMIT": "332f19e4aba4bffe2dc736f45ea7f7bdc585903a",
          "SIZES": [1, 5, 19, 48, 317], "COST_STEPS": 200, "EPISODES": 5, "EVAL_SEED": 1}
d = os.path.join(HERE, "kernel_gate1"); os.makedirs(d, exist_ok=True)
open(os.path.join(d, "run.py"), "w").write("".join(f"{k} = {v!r}\n" for k, v in consts.items()) + open(os.path.join(HERE, "gate1_body.py")).read())
json.dump({"id": "seungwonryoo/tdmpc2-probe-gate1", "title": "TDMPC2 Probe Gate1", "code_file": "run.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [],
           "competition_sources": [], "kernel_sources": [], "model_sources": []}, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
print(d)
