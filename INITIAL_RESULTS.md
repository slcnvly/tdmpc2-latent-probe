# INITIAL_RESULTS (GATE 1까지)

작성: 2026-10-22. 실제로 실행한 것만 적는다. **Step 4 이후(데이터 수집, latent 추출, probe, task swap)는 실행하지 않았다.** GATE 1에서 멈췄다.

## 실행한 것

| 단계 | 내용 | 장소 | 결과 파일 |
|---|---|---|---|
| Step 1 (1~8) | 코드 확인, CPU 검증 (padding, embedding, encoder, encode 일치, reward, 상태 복원) | 로컬 CPU | `results/audit_checks.json`, `results/reward_obs_mapping.json` |
| Step 1 (9) | 크기별 비용 실측 | Kaggle T4 | `results/gate1/cost.json` |
| Step 2 | 체크포인트 크기 결정 (규칙은 측정 전에 고정) | Kaggle T4 | `results/gate1/size_choice.json` |
| Step 3 | Experiment 0: mt30-48M을 walker 5개 task에서 평가 | Kaggle T4 | `results/gate1/exp0_eval.json` |

- 체크포인트: HF `nicklashansen/tdmpc2`의 `multitask/mt30-{1,5,19,48,317}M.pt`. 로컬에 받은 네 파일은 sha256 앞자리가 HF LFS oid와 일치.
- TD-MPC2 코드: 커밋 `332f19e`, 수정 없음. 환경: Python 3.11, torch 2.7.1(+cu128 / +cpu), dm-control 1.0.16, mujoco 3.1.2, tensordict 0.8.3 (`results/gate1/pip_freeze.txt`).

## Experiment 0: mt30-48M 단일 task 평가

설정: `code/eval_single_task.py eval --size 48 --episodes 5 --seed 1`.
공식 `evaluate.py`의 루프(30개 env를 가진 `MultitaskWrapper`, `agent.act(obs, t0, task)`, `eval_mode=False`, `compile=true`)를 walker task index에만 돌렸다.
seed는 1 하나(`set_seed(1)` 한 번, env seed도 1)이고, 다섯 task를 순서대로 각 5 에피소드(500 step) 돌렸다.

| task | index | 에피소드 return (5개) | 평균 | 표준편차 | 단일 task TD-MPC2 참고값 (4M step, seed 1·2·3) |
|---|---|---|---|---|---|
| walker-stand | 0 | 996.8, 997.0, 981.7, 982.7, 983.2 | 988.3 | 7.9 | 992.2, 992.0, 993.9 |
| walker-walk | 1 | 988.8, 977.7, 974.3, 967.2, 974.4 | 976.4 | 7.9 | 977.4, 982.9, 983.1 |
| walker-run | 2 | 654.2, 619.6, 604.4, 565.7, 638.5 | 616.5 | 34.1 | 879.3, 868.1, 884.2 |
| walker-walk-backwards | 19 | 982.8, 977.7, 955.3, 958.2, 944.8 | 963.8 | 16.0 | 980.4, 984.9, 985.2 |
| walker-run-backwards | 20 | 166.7, 189.3, 163.8, 191.5, 163.3 | 174.9 | 14.2 | 939.3, 936.5, 941.0 |

- 에피소드당 시간 약 80초(첫 에피소드는 compile 포함 174초). 최대 GPU 메모리 562 MB.

### 비교 기준에 관한 사실

- 지시는 "저장소 results 폴더의 공식 task별 점수와 비교"였다. 그런데 `tdmpc2/results/`에는 **mt30 체크포인트의 task별 점수가 없다.**
  있는 것은 단일 task online RL 학습 곡선(`results/tdmpc2/<task>.csv`, 열: step, reward, seed)이다.
- 논문(arXiv 2310.16828)도 mt30은 30개 task 평균 normalized score만 보고한다 (Figure 7, 22: 48M은 59.4).
  task별로 나누어 볼 수 없다.
- 위 표의 오른쪽 열은 **다른 agent**(task 하나만으로 학습한 5M 모델)의 마지막 기록값이다. 공식 mt30 점수가 아니다.
- 30개 task 전체를 평가하면 normalized score와는 비교할 수 있지만, 지시가 30개 task 전체 재현을 금지해 하지 않았다.

### 관찰된 사실

- walker-stand, walker-walk, walker-walk-backwards: 단일 task 참고값과 1~2% 차이.
- walker-run: 평균 616.5. 참고값(868~884)보다 약 260 낮다.
- walker-run-backwards: 평균 174.9. 참고값(937~941)보다 약 765 낮다.
- 두 run task의 차이는 다섯 에피소드의 표준편차(34.1, 14.2)보다 훨씬 크다.

### 원인 조사 (지시에 따라 GATE 1 전 진행 여부 판단용)

확인한 것:
- 다섯 task는 같은 스크립트, 같은 env 객체, 같은 agent로 돌았고 다른 것은 task index뿐이다.
  index는 `TASK_SET['mt30']`의 순서에서 얻었다 (`common/__init__.py:29, 34`: stand 0, walk 1, run 2, walk-backwards 19, run-backwards 20).
- walker 관측·행동에는 padding과 masking이 없다 (BASELINE_AUDIT.md 1절). 따라서 padding 처리 오류로 이 task들만 틀어질 경로는 없다.
- walker-run-backwards env의 reward 함수와 물리는 audit에서 확인한 것과 같다 (BASELINE_AUDIT.md 6절).

확인하지 못한 것:
- 이 두 값이 mt30-48M의 실제 성능인지, 이 스크립트의 문제인지 가를 공식 기준값이 없다.
- 공식 `evaluate.py`를 mt30 전체로 돌려 walker 값을 비교하는 방법이 있지만, 30개 task 전체 평가라 하지 않았다.
- 다른 크기(19M, 317M)의 walker-run 계열 성능은 재지 않았다.

**따라서 Experiment 0의 판정 기준("공식 점수와 noise 수준을 넘어 어긋나면 멈춤")은 그대로 적용할 수 없다.**
단일 task 참고값을 기준으로 보면 두 task가 noise를 넘어 어긋난다. 이 상태로 GATE 1에서 멈추고 결정을 기다린다.

## 계획 대비 변경점

| 계획 | 실제 | 이유 |
|---|---|---|
| "이 컴퓨터"의 GPU | Kaggle Tesla T4 | 로컬에 GPU가 없다 (`nvidia-smi` 없음, WSL) |
| conda 환경 | uv venv (Python 3.11) | conda 없이 같은 버전 고정 |
| 공식 task별 점수와 비교 | 단일 task 학습 곡선과 나란히 적음 | 공식 mt30 task별 점수가 저장소·논문에 없다 |
| 19M latent_dim 768 (사전 정보) | 512 | `parser.py:67-68`이 mt30 19M을 512로 덮어쓴다 |

## 이 결과로 결론 낼 수 없는 것

- mt30-48M이 walker-run, walker-run-backwards에서 "원래" 이 정도 성능인지. 비교할 공식 값이 없다.
- 다섯 task 결과가 seed 하나(1), 에피소드 5개라 seed 사이 변동은 모른다.
- 비용은 T4 한 장, walker-walk 200 step 기준이다. 다른 GPU나 task에서는 다를 수 있다.
- Step 4 이후의 어떤 질문(정보 보존, probe R², task swap 거리)에도 아직 답하지 않았다.

## 세 질문에 대한 현재 답 (GATE 1 시점)

1. **pretrained encoder는 task-independent인가, task-conditioned인가?**
   task-conditioned다. encoder 입력이 `[관측, task embedding]`이다 (`common/world_model.py:105-114`, `common/layers.py:159`).
   같은 관측에 walker task id만 바꾸면 encode 출력이 달라진다 (최대 절대 차이 0.17~0.30, `results/audit_checks.json`).
2. **선택한 task family에서 표현 비교 실험이 기술적으로 타당한가?**
   기술적 전제는 대부분 확인됐다: 관측·행동 차원이 같고 padding이 없으며, 상태를 복원해 관측과 다섯 reward를 오차 0으로 재계산할 수 있다.
   타당성을 해칠 수 있는 요인:
   - 기록 reward가 두 substep의 합이라 "상태의 reward"와 다르다.
   - horizontal_velocity가 관측 성분에 직접 없다.
   - backwards 두 task는 XML이 다르다(동역학 차이는 관찰되지 않음).
   - mt30-48M이 두 run task를 단일 task 수준으로 수행하지 못한다. 이 경우 (a) rollout의 상태 분포가 task마다 크게 다르다.
3. **첫 검증 가능한 실험을 실제로 실행했는가?**
   아니다. Experiment 0(체크포인트 동작 확인)과 비용 실측까지만 실행했다. representation 분석 실험(Step 4~7)은 GATE 1 승인 전이라 실행하지 않았다.
