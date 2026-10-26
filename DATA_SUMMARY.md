# DATA_SUMMARY (Experiment 1, Step 4 수집 결과)

작성: 2026-10-26. 수집 데이터의 기술 통계만 적는다. 해석은 쓰지 않는다.
설정은 [EXPERIMENT1_PREREG.md](EXPERIMENT1_PREREG.md)(커밋 `6b9c0f1`)를 따른다.

## 수집 개요

| 항목 | 값 |
|---|---|
| checkpoint | `mt30-48M.pt`, sha256 `bacb9ae1891a9b03dc3942a287080beb573220ea9d0263c5efd7a94a5c2ec419` (수집 시작 시 다시 계산해 GATE 2 값과 같음) |
| 수집 코드 | `code/collect_rollouts.py`, 저장소 커밋 `9d13467` (Kaggle 커널이 이 커밋을 받아 실행) |
| 완료 | **15개 조건 × 20 에피소드 = 300 에피소드 전부, 150,000 상태 (t = 0~499)**. 미완료 없음 |
| A, B | Kaggle Tesla T4 2장에 프로세스 하나씩. 에피소드를 번갈아 나눴다 (수집 장치는 아래 표) |
| C | 같은 Kaggle 세션의 CPU 프로세스. planner를 부르지 않았다 |
| 무결성 검사 | 300 에피소드 모두 통과 (`results/exp1/check_all.json`). 아래 "무결성 검사" |

- **C의 source task 꼬리표는 의미가 없다.** C의 행동은 task와 관계없는 U[−1, 1]⁶ 독립 표본이다.
  꼬리표에 따라 달라지는 것은 에피소드 seed(따라서 초기 상태와 행동 표본)와, return을 계산한 reward 함수뿐이다.
- 같은 task·같은 에피소드 번호의 A, B, C는 같은 seed로 시작하므로 초기 상태가 같다 (PREREG 1).

### 조건별 개수와 return

| source task | 종류 | 에피소드 | 상태 | return 평균 | return 표준편차 | 수집 장치 |
|---|---|---|---|---|---|---|
| walker-stand | A | 20 | 10,000 | 991.2 | 6.7 | Tesla T4 / GPU 0 |
| walker-stand | B | 20 | 10,000 | 991.4 | 5.8 | Tesla T4 / GPU 1 |
| walker-stand | C | 20 | 10,000 | 136.2 | 13.9 | cpu |
| walker-walk | A | 20 | 10,000 | 981.8 | 8.5 | Tesla T4 / GPU 1 |
| walker-walk | B | 20 | 10,000 | 974.5 | 9.7 | Tesla T4 / GPU 0 |
| walker-walk | C | 20 | 10,000 | 39.5 | 8.0 | cpu |
| walker-run | A | 20 | 10,000 | 622.9 | 31.1 | Tesla T4 / GPU 0 |
| walker-run | B | 20 | 10,000 | 529.6 | 24.5 | Tesla T4 / GPU 1 |
| walker-run | C | 20 | 10,000 | 26.4 | 3.9 | cpu |
| walker-walk-backwards | A | 20 | 10,000 | 963.1 | 12.0 | Tesla T4 / GPU 1 |
| walker-walk-backwards | B | 20 | 10,000 | 952.5 | 17.4 | Tesla T4 / GPU 0 |
| walker-walk-backwards | C | 20 | 10,000 | 40.7 | 13.3 | cpu |
| walker-run-backwards | A | 20 | 10,000 | 195.1 | 32.5 | Tesla T4 / GPU 0 |
| walker-run-backwards | B | 20 | 10,000 | 177.1 | 14.4 | Tesla T4 / GPU 1 |
| walker-run-backwards | C | 20 | 10,000 | 25.5 | 4.3 | cpu |

### 몸통 높이 torso_height (m)

평균 / 표준편차 / 5% / 50% / 95% 분위수, 상태 단위(조건당 10,000개).

| source task | A | B | C |
|---|---|---|---|
| walker-stand | 1.282 / 0.062 / 1.266 / 1.287 / 1.303 | 1.278 / 0.060 / 1.219 / 1.288 / 1.311 | 0.238 / 0.181 / 0.071 / 0.192 / 0.560 |
| walker-walk | 1.272 / 0.066 / 1.228 / 1.280 / 1.318 | 1.263 / 0.071 / 1.196 / 1.274 / 1.318 | 0.232 / 0.178 / 0.071 / 0.186 / 0.552 |
| walker-run | 1.212 / 0.067 / 1.132 / 1.214 / 1.298 | 1.176 / 0.090 / 1.075 / 1.182 / 1.283 | 0.253 / 0.195 / 0.072 / 0.205 / 0.640 |
| walker-walk-backwards | 1.296 / 0.073 / 1.254 / 1.306 / 1.333 | 1.281 / 0.070 / 1.207 / 1.293 / 1.332 | 0.233 / 0.182 / 0.070 / 0.186 / 0.573 |
| walker-run-backwards | 1.233 / 0.076 / 1.127 / 1.248 / 1.319 | 1.210 / 0.093 / 1.093 / 1.221 / 1.316 | 0.245 / 0.185 / 0.071 / 0.200 / 0.597 |

### 수직 정도 torso_upright (torso z축과 세계 z축의 cos, −1~1)

평균 / 표준편차 / 5% / 50% / 95% 분위수, 상태 단위(조건당 10,000개).

| source task | A | B | C |
|---|---|---|---|
| walker-stand | 0.963 / 0.156 / 0.946 / 0.983 / 1.000 | 0.961 / 0.141 / 0.924 / 0.984 / 1.000 | -0.054 / 0.374 / -0.616 / -0.060 / 0.667 |
| walker-walk | 0.975 / 0.161 / 0.972 / 0.996 / 1.000 | 0.971 / 0.156 / 0.952 / 0.995 / 1.000 | -0.054 / 0.361 / -0.608 / -0.051 / 0.613 |
| walker-run | 0.969 / 0.145 / 0.924 / 0.993 / 1.000 | 0.952 / 0.189 / 0.884 / 0.989 / 1.000 | -0.042 / 0.386 / -0.609 / -0.048 / 0.706 |
| walker-walk-backwards | 0.960 / 0.174 / 0.944 / 0.986 / 0.999 | 0.959 / 0.148 / 0.886 / 0.988 / 1.000 | -0.057 / 0.364 / -0.616 / -0.048 / 0.609 |
| walker-run-backwards | 0.964 / 0.146 / 0.922 / 0.988 / 1.000 | 0.953 / 0.183 / 0.901 / 0.987 / 1.000 | -0.044 / 0.387 / -0.616 / -0.048 / 0.720 |

### 수평 속도 horizontal_velocity (m/s, +x가 앞)

평균 / 표준편차 / 5% / 50% / 95% 분위수, 상태 단위(조건당 10,000개).

| source task | A | B | C |
|---|---|---|---|
| walker-stand | 0.039 / 0.358 / -0.275 / -0.035 / 0.893 | 0.090 / 0.368 / -0.292 / 0.006 / 0.956 | -0.005 / 0.427 / -0.711 / -0.004 / 0.697 |
| walker-walk | 1.388 / 0.308 / 0.987 / 1.404 / 1.821 | 1.312 / 0.341 / 0.901 / 1.290 / 1.830 | -0.005 / 0.444 / -0.696 / -0.016 / 0.764 |
| walker-run | 4.411 / 1.280 / 1.571 / 4.811 / 5.749 | 3.590 / 1.104 / 1.628 / 3.755 / 5.027 | 0.013 / 0.454 / -0.745 / 0.008 / 0.754 |
| walker-walk-backwards | -1.016 / 0.245 / -1.197 / -1.047 / -0.872 | -1.021 / 0.291 / -1.300 / -1.047 / -0.783 | 0.002 / 0.419 / -0.664 / -0.008 / 0.733 |
| walker-run-backwards | 0.631 / 1.324 / -1.369 / 0.801 / 2.849 | 1.106 / 1.240 / -0.959 / 1.164 / 3.261 | -0.015 / 0.433 / -0.719 / -0.018 / 0.708 |

### Fallen state 비율

fallen = torso_height < 0.6 m (EXPERIMENT1_PREREG.md 2). 보조로 torso_height < 1.2 m(standing 기준 미달) 비율도 적는다.

| source task | A fallen | B fallen | C fallen | A < 1.2 m | B < 1.2 m | C < 1.2 m |
|---|---|---|---|---|---|---|
| walker-stand | 0.003 | 0.002 | 0.956 | 0.014 | 0.034 | 0.994 |
| walker-walk | 0.003 | 0.003 | 0.960 | 0.016 | 0.057 | 0.993 |
| walker-run | 0.002 | 0.005 | 0.939 | 0.374 | 0.612 | 0.993 |
| walker-walk-backwards | 0.004 | 0.003 | 0.956 | 0.022 | 0.044 | 0.993 |
| walker-run-backwards | 0.002 | 0.004 | 0.951 | 0.319 | 0.417 | 0.993 |

- return은 에피소드(500 env step)의 r_logged_t 합이다(`evaluate.py`와 같은 float32 누적, 범위 0~1000). C의 return은 source task 꼬리표의 reward 함수로 계산한 값이다.
- 높이·수직 정도·수평 속도는 저장한 s_t를 복원해 `physics.torso_height()`, `torso_upright()`, `horizontal_velocity()`로 계산했다 (`dm_control/suite/walker.py:84-94`).

## 무결성 검사 (Step 4 절차 1)

- **첫 단계**: 15개 조건의 0번 에피소드를 먼저 수집해 검사했고, 15개 모두 통과한 뒤에만 나머지 1~19번을 수집했다 (`results/exp1/check_ep0.json`).
- **전체 검사**: 수집이 끝난 뒤 300 에피소드 전부를 다시 검사했다 (`results/exp1/check_all.json`, `code/check_rollouts.py`).

| 검사 (에피소드마다, 모든 t) | 기준 | 300 에피소드 중 최대값 |
|---|---|---|
| s_t 복원 → 관측 재계산 vs 저장한 o_t (padded, raw) | ≤ 1e-5 | 0.0 |
| 두 번째 substep 종료 상태 복원 → r_source vs 저장한 두 번째 substep reward | ≤ 1e-6 | 0.0 |
| r_logged_t vs 두 substep reward의 float32 합 | ≤ 1e-5 | 0.0 |
| 두 번째 substep 종료 상태 = s_{t+1}, 다음 관측 = o_{t+1} | 정확히 같음 | 모두 같음 |

- 재현성 관찰: walker-walk C 0번 에피소드(seed 11200)를 수집 전에 로컬 CPU에서 같은 코드로 돌린 return 32.273은 Kaggle 수집값과 같았다.

## 저장 항목 (에피소드 파일 `data/rollouts/<task>/<A|B|C>/epNN.npz` + `.json`)

| 키 | 모양 | 내용 |
|---|---|---|
| `state` | (500, 19) float64 | s_t = `physics.get_state()`: time, qpos 9, qvel 9 |
| `sub_state` | (500, 2, 19) float64 | 두 substep 각각의 종료 상태. `sub_state[t, 1]` = s_{t+1} |
| `obs`, `obs_raw` | (500, 24) float32 | padded 관측 o_t, raw 관측 (walker는 padding이 없어 같은 값) |
| `next_obs` | (500, 24) float32 | o_{t+1} |
| `a_plan` | (500, 6) float32 | planner 출력 (A·B). C는 NaN |
| `noise` | (500, 6) float32 | B의 ε_t ~ N(0, 0.3²). A·C는 NaN |
| `a_exec` | (500, 6) float32 | 실행한 행동 (A: = a_plan, B: clip(a_plan + ε, −1, 1), C: 균등 표본) |
| `r_logged` | (500,) float32 | r_logged_t |
| `sub_reward` | (500, 2) float64 | 두 substep 각각의 reward (source task) |

에피소드 메타데이터(`.json`): task, mt30 task 번호, source task와 그 꼬리표가 의미 있는지, rollout 종류, 에피소드 번호, episode_seed, noise 표준편차, return, 소요 시간, 수집 장치, checkpoint 파일 이름과 sha256, torch·mujoco·dm_control·numpy·python 버전, TD-MPC2 커밋.

**Q 재계산에 필요한 정보**: 매 step의 o_t, noise 전·후 행동(`a_plan`, `a_exec`, `noise`)과 에피소드별 task ID, checkpoint sha256, 버전 정보가 300 에피소드 모두에 있다 (`results/exp1/data_summary.json` `q_recompute_info_present: true`). Q는 수집 중에 계산하지 않았다.

## 실행 기록과 비용

| 항목 | 값 |
|---|---|
| Kaggle 커널 | `tdmpc2-probe-exp1-collect` v1, 2026-10-26 05:38 ~ 07:52 UTC |
| 첫 단계 (0번 에피소드 15개 + 검사) | 575.9초 |
| 둘째 단계 (나머지 285 에피소드) | 6,780.2초 (1.88시간) |
| 커널 전체 | 약 2.2시간. 주간 GPU 할당량 사용량이 1.33 → 3.55시간 |
| 에피소드당 시간 | A 평균 69.3초, B 평균 69.8초 (GPU), C 평균 0.6초 (CPU) |
| 예상 대비 | 예상 A·B 4.2 GPU 시간 → 실제 프로세스 합계 3.87시간 (두 GPU에 나눠 1.9시간 소요) |
| 데이터 크기 | rollout 110 MB (압축 npz) |
