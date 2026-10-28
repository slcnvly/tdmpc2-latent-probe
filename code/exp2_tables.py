"""Markdown tables for Experiment 2 from exp2_evaluate.py output (cells.csv, summary.json) and the embedding table.

  python exp2_tables.py --eval ../results/exp2/eval --emb ../results/exp2/kaggle/inject_check_kaggle.json
"""
import argparse
import csv
import json
import os
from collections import Counter

TASKS = ["walker-stand", "walker-walk", "walker-run", "walker-walk-backwards", "walker-run-backwards"]
SHORT = ["stand", "walk", "run", "walk-bwd", "run-bwd"]
LOW = "실질적으로 낮음"
REF = "randinit:walker-stand"


def lbl(r):
    return {"zero": "영벡터", REF: "[참고] random-init 모델"}.get(r, r.replace("task:", "").replace("rand:", "무작위 "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--emb", required=True)
    a = ap.parse_args()
    cells = list(csv.DictReader(open(os.path.join(a.eval, "cells.csv"), encoding="utf-8")))
    s = json.load(open(os.path.join(a.eval, "summary.json"), encoding="utf-8"))
    emb = json.load(open(a.emb, encoding="utf-8"))["embedding_table"]
    get = {(c["setting"], c["row"], c["target"]): c for c in cells}
    order = s["row_order"]
    L = []
    w = L.append

    w("# Experiment 2 표 (자동 생성: `code/exp2_tables.py`)\n")
    w("칸 표기: test R² [95% bootstrap 구간]. ▼ = 판정 \"실질적으로 낮음\"(Δ 구간 하한 > 0, Δ ≥ 0.02). "
      "† = RMSE 비율 ≥ 1.5. ‡ = random-init 기준선 행보다 R² 점추정이 낮음. 행 순서는 PREREG 2절 정렬 규칙.\n")

    w("## 행 (embedding) 목록\n")
    w(f"c30(학습된 30개 embedding L2 norm 평균) = {emb['c30_mean_norm']:.6f}. 무작위 벡터 seed {emb['random_seed']}.\n")
    w("| 행 | L2 norm | " + " | ".join(f"cos({t})" for t in SHORT) + " |")
    w("|---|---|" + "---|" * 5)
    for r in order:
        e = emb["rows"][r]
        cos = ["정의 안 됨" if c is None else f"{c:.3f}" for c in e["cos_walker"]]
        w(f"| {lbl(r)} | {e['l2_norm']:.4f} | " + " | ".join(cos) + " |")
    w("")

    def table(setting, rows, title):
        w(f"## {title} (`{setting}`)\n")
        c0 = get[(setting, rows[0], TASKS[0])]
        w(f"상태 {c0['n_states']}개, 에피소드 {c0['n_episodes']}개.\n")
        w("| 행 \\ target | " + " | ".join(TASKS) + " |")
        w("|---|" + "---|" * 5)
        for r in rows:
            out = []
            for t in TASKS:
                c = get[(setting, r, t)]
                m = ("▼" if c["verdict"] == LOW else "") + ("†" if c["rmse_ratio_ge_1_5"] == "True" else "") + \
                    ("‡" if c["below_randinit"] == "True" else "")
                out.append(f"{float(c['r2']):.3f} [{float(c['r2_lo']):.3f}, {float(c['r2_hi']):.3f}]{m}")
            w(f"| {lbl(r)} | " + " | ".join(out) + " |")
        w("")

    rows_all = order + [REF]
    table("pooled|linear", rows_all, "Linear (ridge), pooled test — 주 결과")
    table("pooled|mlp", rows_all, "MLP multi-output (seed 5개 평균), pooled test — 주 결과")
    table("not_fallen|linear", rows_all, "Linear, fallen 제외 test (보조)")
    table("not_fallen|mlp", rows_all, "MLP, fallen 제외 test (보조)")
    table("pooled|linear_lambda_ext", rows_all, "Linear, λ 확장 10^[−8:0.5:4], pooled test (보조)")
    srows = sorted({c["row"] for c in cells if c["setting"] == "standing_only|linear"},
                   key=lambda r: (0 if r.startswith("task:walker") else 1 if r == "raw" else 2 if r.startswith("randinit") else 3, r))
    table("standing_only|linear", srows, "서 있는 상태 전용 linear probe (보조)")

    w("## \"실질적으로 낮음\" 칸 수 (열별)\n")
    w("| 결과 구분 | 행 묶음 | 행 수 | " + " | ".join(SHORT) + " |")
    w("|---|---|---|" + "---|" * 5)
    for st, c in s["counts"].items():
        for g, name in (("nonwalker25", "비-walker task"), ("zero", "영벡터"), ("random5", "무작위 벡터")):
            if c[g]["n_rows"]:
                w(f"| {st} | {name} | {c[g]['n_rows']} | " + " | ".join(str(c[g][t]) for t in TASKS) + " |")
    w("")
    w("RMSE 비율 ≥ 1.5 칸 수와 random-init 기준선 미만 칸 수:\n")
    w("| 결과 구분 | 행 묶음 | 지표 | " + " | ".join(SHORT) + " |")
    w("|---|---|---|" + "---|" * 5)
    for st, c in s["counts"].items():
        for g, name in (("nonwalker25", "비-walker task"), ("zero", "영벡터"), ("random5", "무작위 벡터")):
            if c[g]["n_rows"]:
                w(f"| {st} | {name} | RMSE 비율 ≥ 1.5 | " + " | ".join(str(c[g]["rmse_ratio_ge_1_5"][t]) for t in TASKS) + " |")
                w(f"| {st} | {name} | random-init 미만 | " + " | ".join(str(c[g]["below_randinit"][t]) for t in TASKS) + " |")
    w("")

    w("## 칸 분류 (linear 판정, MLP 판정)\n")
    for sub, cat in s["categories"].items():
        w(f"### {sub}\n")
        names = ["둘 다 아님", "linear만", "MLP만", "둘 다"]
        w("| 행 묶음 | target | " + " | ".join(names) + " |")
        w("|---|---|" + "---|" * 4)
        for gname, members in (("비-walker task 25", s["nonwalker_rows"]), ("영벡터", ["zero"]),
                               ("무작위 벡터 5", [r for r in order if r.startswith("rand:")]),
                               ("walker 5", [r for r in order if r.startswith("task:walker")])):
            for t in TASKS:
                cnt = Counter(cat[f"{r}|{t}"] for r in members)
                w(f"| {gname} | {t} | " + " | ".join(str(cnt[n]) for n in names) + " |")
        w("")

    w("## 판정이 바뀐 칸 (linear pooled 주 결과 대비)\n")
    for st, ch in s["verdict_changes_vs_pooled_linear_main"].items():
        w(f"- `{st}`: {len(ch)}칸")
        for x in ch:
            w(f"  - {lbl(x['row'])} → {x['target']}: 주 결과 \"{x['main']}\" → 이 구분 \"{x['this']}\"")
    w("")
    w("서 있는 상태 전용 probe의 비-walker 행 선택 (PREREG 4절, linear pooled 5열 평균 R² 낮은 순):\n")
    w("| 순위 | task | task id | 5열 평균 R² |")
    w("|---|---|---|---|")
    for i, x in enumerate(s["standing_selection"], 1):
        w(f"| {i} | {x['task']} | {x['task_id']} | {x['mean_r2_lin_main_pooled']:.4f} |")
    w(f"\n선택: {', '.join(s['standing_chosen'])}. 상태 수: {s['standing_n_states']}.\n")

    w("## d_TV와 R² (서술용, 판정 없음)\n")
    w("| probe | 행 집합 | n | Pearson | Spearman |")
    w("|---|---|---|---|---|")
    for p, v in s["tv_vs_r2"].items():
        for k, name in (("all36", "36행 전체"), ("nonwalker25", "비-walker 25행")):
            w(f"| {p} | {name} | {v[k]['n']} | {v[k]['pearson']:.3f} | {v[k]['spearman']:.3f} |")
    w("")
    w("| 행 | x: 대각선 z와의 d_TV 중앙값 (5열 평균) | linear 5열 평균 R² | MLP 5열 평균 R² |")
    w("|---|---|---|---|")
    for r in order:
        w(f"| {lbl(r)} | {s['tv_x'][r]:.4f} | {s['tv_vs_r2']['linear']['y'][r]:.4f} | {s['tv_vs_r2']['mlp']['y'][r]:.4f} |")
    w("")
    open(os.path.join(a.eval, "tables.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print(os.path.join(a.eval, "tables.md"), len(L), "lines")


if __name__ == "__main__":
    main()
