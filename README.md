# tdmpc2-latent-probe

TD-MPC2 multi-task (mt30) checkpoint의 task-conditioned encoder가 task에 따라 정보를 어떻게 다르게 보존하는지 진단하기 위한 작업 저장소.

현재 단계: **GATE 3** (Experiment 1 데이터 수집과 latent·reward target 준비 완료). Step 6(probe) 이후는 사용자 승인 전까지 실행하지 않는다.

| 문서 | 내용 |
|---|---|
| [BASELINE_AUDIT.md](BASELINE_AUDIT.md) | 코드 근거(파일·줄 번호)가 붙은 baseline 확인 |
| [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | 실험 계획 초안 (가설은 사용자 작성) |
| [INITIAL_RESULTS.md](INITIAL_RESULTS.md) | GATE 1까지 실제로 실행한 것과 그 결과 |
| [VERIFICATION_PLAN.md](VERIFICATION_PLAN.md) | GATE 2 검증의 판정 기준 (실행 전 기록) |
| [VERIFICATION_RESULTS.md](VERIFICATION_RESULTS.md) | V1~V5 결과와 판정 |
| [EXPERIMENT1_PREREG.md](EXPERIMENT1_PREREG.md) | Experiment 1 사전 고정 (Step P0) |
| [DATA_SUMMARY.md](DATA_SUMMARY.md) | Experiment 1 수집 데이터 요약 |
| [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) | Experiment 1 결과 (GATE 3 시점, 진행 중) |

데이터: rollout(110 MB)과 latent(4.6 GB)는 현재 작업 컴퓨터의 `data/`에만 있다 (공유 위치는 미정). rollout은 `code/collect_rollouts.py`, pooled 데이터와 latent는 `code/step5_latents_targets.py`로 다시 만든다.

## 재현

```bash
git clone https://github.com/nicklashansen/tdmpc2 && git -C tdmpc2 checkout 332f19e4aba4bffe2dc736f45ea7f7bdc585903a
uv venv --python 3.11 venv && uv pip install --python venv/bin/python -r requirements.txt
hf download nicklashansen/tdmpc2 multitask/mt30-5M.pt --local-dir ckpts   # 필요한 크기만
cd code && ../venv/bin/python audit_checks.py                               # CPU
../venv/bin/python eval_single_task.py cost --out ../results/cost.json      # GPU
```

TD-MPC2 저장소 코드는 수정하지 않는다. `code/common_setup.py`가 공식 config·env·model을 그대로 불러온다.
