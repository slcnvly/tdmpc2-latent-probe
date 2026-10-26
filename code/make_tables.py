"""Markdown tables and summary facts for EXPERIMENT_RESULTS.md from results/exp1/eval (no new computation of R^2).
  python make_tables.py --eval ../results/exp1/eval --probes ../results/exp1/probes
"""
import argparse
import csv
import json
import os

import numpy as np

T = ["walker-stand", "walker-walk", "walker-run", "walker-walk-backwards", "walker-run-backwards"]
S = ["stand", "walk", "run", "walk-bwd", "run-bwd"]

ap = argparse.ArgumentParser()
ap.add_argument("--eval", required=True)
ap.add_argument("--probes", required=True)
a = ap.parse_args()
cells = list(csv.DictReader(open(os.path.join(a.eval, "cells.csv"))))
diffs = list(csv.DictReader(open(os.path.join(a.eval, "paired_differences.csv"))))
sub = json.load(open(os.path.join(a.eval, "subsets.json")))["subsets"]
C = {(c["subset"], c["probe"], c["input"], c["target"]): c for c in cells}
f = lambda x: "n/a" if x in ("", None) else f"{float(x):.3f}"
out = []


def r(s, p, i, t):
    c = C.get((s, p, i, t))
    return None if c is None or c["r2"] == "" else float(c["r2"])


def grid(s, p, kind, ci=True):
    lines = ["| encoder 입력 τ \\ target τ′ | " + " | ".join(T) + " |", "|---|" + "---|" * 5]
    for k in range(5):
        row = []
        for t in T:
            c = C[(s, p, f"{kind}:{k}", t)]
            row.append(f"{f(c['r2'])} [{f(c['ci_lo'])}, {f(c['ci_hi'])}]" if ci else f(c["r2"]))
        lines.append(f"| {T[k]} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def raw_row(s):
    lines = ["| probe | " + " | ".join(T) + " |", "|---|" + "---|" * 5]
    for p in ("linear", "mlp"):
        lines.append(f"| {p} | " + " | ".join(f"{f(C[(s, p, 'raw', t)]['r2'])} [{f(C[(s, p, 'raw', t)]['ci_lo'])}, {f(C[(s, p, 'raw', t)]['ci_hi'])}]" for t in T) + " |")
    return "\n".join(lines)


def subset_desc(s):
    d = sub[s]
    return (f"test 상태 {d['n_states']:,}개, 에피소드 {d['n_episodes']}개. 열별 target 평균: " + ", ".join(f"{S[j]} {d['target_mean'][j]:.4f}" for j in range(5))
            + ". 분산: " + ", ".join(f"{S[j]} {d['target_var'][j]:.4f}" for j in range(5)) + ".")


for s, title in (("pooled", "pooled test (주 결과)"), ("not_fallen", "fallen 상태 제외 pooled test (보조 1)")):
    out.append(f"\n### {title}\n\n{subset_desc(s)}\n")
    for p, pn in (("linear", "Linear (ridge)"), ("mlp", "MLP (seed 5개 평균)")):
        for kind, kn in (("learned", "학습된 encoder z(s, τ)"), ("random", "random-init z(s, τ) (사전 등록 대조군)"), ("normmatched", "norm-matched random-init z(s, τ) (보조 대조군)")):
            out.append(f"\n**{pn}, {kn}** (칸: test R² [95% bootstrap 구간])\n\n" + grid(s, p, kind))
    out.append(f"\n**raw observation o(s)** (열마다 1개, 행 없음)\n\n" + raw_row(s))

out.append("\n### in-distribution test (보조 2)\n\n**이 표 안에서는 행끼리 비교하지 않는다. 행마다 test 상태 집합이 다르다.**\n")
for d, dn in (("indist_A", "주 정의: source task = τ인 A 상태"), ("indist_AB", "보조 정의: source task = τ인 A와 B 상태")):
    for p in ("linear", "mlp"):
        lines = [f"\n**{dn}, {p}, 학습된 z(s, τ)** (행 τ의 test 상태 수·에피소드 수는 오른쪽 열)\n",
                 "| 행 τ | " + " | ".join(T) + " | 상태 / 에피소드 |", "|---|" + "---|" * 6]
        for k in range(5):
            s = f"{d}_row{k}"
            lines.append(f"| {T[k]} | " + " | ".join(f"{f(C[(s, p, f'learned:{k}', t)]['r2'])} [{f(C[(s, p, f'learned:{k}', t)]['ci_lo'])}, {f(C[(s, p, f'learned:{k}', t)]['ci_hi'])}]" for t in T)
                         + f" | {sub[s]['n_states']:,} / {sub[s]['n_episodes']} |")
        out.append("\n".join(lines))
    lines = [f"\n{dn}: 행별 test 집합의 target 분산\n", "| 행 τ | " + " | ".join(T) + " |", "|---|" + "---|" * 5]
    for k in range(5):
        lines.append(f"| {T[k]} | " + " | ".join(f"{v:.4f}" for v in sub[f'{d}_row{k}']['target_var']) + " |")
    out.append("\n".join(lines))

# paired differences summary
out.append("\n### paired bootstrap difference 판정 개수 (주 기준)\n")
lines = ["| 결과 구분 | probe | 비교 | 차이 있음 | 구분되지 않음 | 계산 안 됨 |", "|---|---|---|---|---|---|"]
for s in ("pooled", "not_fallen", "indist_A", "indist_AB"):
    for p in ("linear", "mlp"):
        for cmp in ("a_learned_vs_raw", "b_learned_vs_random", "c_diag_vs_row", "d_learned_vs_normmatched"):
            ds = [d for d in diffs if (d["subset"] == s or d["subset"].startswith(s + "_row")) and d["probe"] == p and d["comparison"] == cmp]
            if not ds:
                continue
            n1 = sum(d["verdict"] == "difference" for d in ds); n2 = sum(d["verdict"] == "not distinguishable" for d in ds); n3 = len(ds) - n1 - n2
            lines.append(f"| {s} | {p} | {cmp} | {n1} | {n2} | {n3} |")
out.append("\n".join(lines))
for s in ("pooled",):
    for p in ("linear", "mlp"):
        for cmp, cn in (("a_learned_vs_raw", "(a) 학습된 z(τ) − raw observation"), ("b_learned_vs_random", "(b) 학습된 z(τ) − random-init z(τ)"),
                        ("d_learned_vs_normmatched", "(d) 학습된 z(τ) − norm-matched random-init z(τ)"), ("c_diag_vs_row", "(c) 대각선 칸 − 같은 열의 다른 행")):
            lines = [f"\n**{s}, {p}, {cn}** (R² 차이 [95% 구간], 판정)\n"]
            if cmp == "c_diag_vs_row":
                lines += ["| target 열 (= 대각선 행) | 비교 행 | 차이 | 판정 | 보조: 개별 구간 겹침 |", "|---|---|---|---|---|"]
                for d in diffs:
                    if d["subset"] == s and d["probe"] == p and d["comparison"] == cmp:
                        lines.append(f"| {d['target']} | {T[int(d['x2'].split(':')[1])]} | {float(d['diff']):+.5f} [{float(d['ci_lo']):+.5f}, {float(d['ci_hi']):+.5f}] | {'차이 있음' if d['verdict'] == 'difference' else '구분되지 않음'} | {'겹침' if d['aux_ci_overlap'] == 'True' else '안 겹침'} |")
            else:
                lines += ["| 행 τ \\ 열 τ′ | " + " | ".join(T) + " |", "|---|" + "---|" * 5]
                for k in range(5):
                    row = []
                    for t in T:
                        d = next(x for x in diffs if x["subset"] == s and x["probe"] == p and x["comparison"] == cmp and x["x1"] == f"learned:{k}" and x["target"] == t)
                        mark = "" if d["verdict"] == "difference" else " (구분 안 됨)"
                        row.append(f"{float(d['diff']):+.4f} [{float(d['ci_lo']):+.4f}, {float(d['ci_hi']):+.4f}]{mark}")
                    lines.append(f"| {T[k]} | " + " | ".join(row) + " |")
            out.append("\n".join(lines))

# facts to report as numbers
facts = {}
for p in ("linear", "mlp"):
    raw_ok = [r("pooled", p, "raw", t) >= 0.99 for t in T]
    lrn_ok = [all(r("pooled", p, f"learned:{k}", t) >= 0.99 for k in range(5)) for t in T]
    facts[f"{p}_ceiling"] = {t: {"raw_r2": r("pooled", p, "raw", t), "learned_min_r2": min(r("pooled", p, f"learned:{k}", t) for k in range(5)),
                                 "both_ge_0.99": bool(raw_ok[j] and lrn_ok[j])} for j, t in enumerate(T)}
    for kind in ("random", "normmatched"):
        facts[f"{p}_{kind}_within_column_max_minus_min"] = {t: max(r("pooled", p, f"{kind}:{k}", t) for k in range(5)) - min(r("pooled", p, f"{kind}:{k}", t) for k in range(5)) for t in T}
    facts[f"{p}_learned_within_column_max_minus_min"] = {t: max(r("pooled", p, f"learned:{k}", t) for k in range(5)) - min(r("pooled", p, f"learned:{k}", t) for k in range(5)) for t in T}
    for kind in ("learned", "random", "normmatched"):
        facts[f"{p}_{kind}_pooled_minus_not_fallen"] = [[r("pooled", p, f"{kind}:{k}", t) - r("not_fallen", p, f"{kind}:{k}", t) for t in T] for k in range(5)]
    facts[f"{p}_raw_pooled_minus_not_fallen"] = [r("pooled", p, "raw", t) - r("not_fallen", p, "raw", t) for t in T]
mlp_std = [float(c["seed_std"]) for c in cells if c["probe"] == "mlp" and c["seed_std"] not in ("", None) and c["subset"] == "pooled"]
facts["mlp_seed_std_pooled_max"] = max(mlp_std)
facts["mlp_seed_std_pooled_median"] = float(np.median(mlp_std))
meta = {}
for w in range(4):
    meta.update(json.load(open(os.path.join(a.probes, f"mlp_meta_w{w}.json"))))
ep = [v["epochs_run"] for v in meta.values()]
facts["mlp_epochs"] = {"n": len(ep), "min": min(ep), "median": float(np.median(ep)), "max": max(ep), "n_reached_200": sum(e == 200 for e in ep),
                       "reached_200_keys": sorted(k for k, v in meta.items() if v["epochs_run"] == 200)}
lin = json.load(open(os.path.join(a.probes, "linear_meta.json")))["cells"]
lams = [v["lambda"] for v in lin.values()]
facts["ridge_lambda"] = {"n": len(lams), "at_min_1e-4": sum(abs(l - 1e-4) < 1e-12 for l in lams), "at_max_1e4": sum(abs(l - 1e4) < 1e-6 for l in lams),
                         "values": {k: v["lambda"] for k, v in lin.items()}}
facts["ridge_const_dims"] = {k: v["n_const_dims"] for k, v in lin.items() if v["n_const_dims"]}
json.dump(facts, open(os.path.join(a.eval, "facts.json"), "w"), indent=1)
open(os.path.join(a.eval, "tables.md"), "w").write("\n".join(out) + "\n")
print(json.dumps({k: v for k, v in facts.items() if "minus_not_fallen" not in k and k != "ridge_lambda"}, indent=1)[:6000])
print("lambda at min", facts["ridge_lambda"]["at_min_1e-4"], "at max", facts["ridge_lambda"]["at_max_1e4"])
