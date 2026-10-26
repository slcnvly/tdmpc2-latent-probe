# EXPERIMENT1_PREREG (Step P0, 데이터 수집 전 고정)

작성: 2026-10-26, Step 4 데이터 수집 전. 이후 바꾸는 항목은 이 파일을 고치지 않고 EXPERIMENT_RESULTS.md의 "계획 대비 변경"에 이유와 함께 적는다.
기준 문서는 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)(V6 수정본)와 [VERIFICATION_RESULTS.md](VERIFICATION_RESULTS.md)다. 용어 s_t, r_logged_t, r_τ(s_t)는 그 정의를 따른다.

## 0. 공통

| 항목 | 값 |
|---|---|
| checkpoint | `multitask/mt30-48M.pt`, sha256 `bacb9ae1891a9b03dc3942a287080beb573220ea9d0263c5efd7a94a5c2ec419` (GATE 2 기록값). 실행 시작 시 다시 계산해 다르면 멈춘다 |
| TD-MPC2 코드 | 커밋 `332f19e`, 수정 없음 |
| 실행 경로 | `code/eval_single_task.py`의 `build_agent`와 같은 순서(`parse_cfg` → `set_seed(1)` → `make_env` → `TDMPC2` → `load`), `compile=true`, `eval_mode=False` |
| task (mt30 번호) | walker-stand 0, walker-walk 1, walker-run 2, walker-walk-backwards 19, walker-run-backwards 20 |
| rollout 종류 (번호) | A policy 0, B noisy policy 1, C random 2 |
| 규모 | 5 task × 3 종류 × 20 에피소드 × 500 env step = 300 에피소드, 150,000 상태 (t = 0, …, 499) |

## 1. 에피소드 seed 규칙

episode_seed = 10000 + 1000 × (mt30 task 번호) + 100 × (종류 번호) + (에피소드 번호 0~19).
예: walker-run-backwards, B, 에피소드 7 → 10000 + 20000 + 100 + 7 = 30107.

에피소드 시작 때 이 seed로 세 난수원을 각각 정한다.

| 난수원 | 설정 | 쓰는 곳 |
|---|---|---|
| planner (torch) | `torch.manual_seed(episode_seed)`: CPU와 모든 CUDA 장치 | MPPI 표본, policy prior 표본, Gumbel 선택, 탐색 noise (`tdmpc2.py`) |
| 환경 초기 상태 | 그 task env의 dm_control task 객체에 `task._random = numpy.random.RandomState(episode_seed)` (`dm_control/suite/base.py:50-56`)를 넣고 `env.reset(task_idx)` | `PlanarWalker.initialize_episode`의 관절 무작위화 |
| 행동 noise (B) / 무작위 행동 (C) | `numpy.random.Generator(PCG64(SeedSequence([episode_seed, 1])))` | B: ε_t ~ N(0, 0.3² I₆). C: a_t ~ U[−1, 1]⁶ |

- 같은 seed를 쓰므로 A·B·C의 같은 task·같은 에피소드 번호는 초기 상태가 같다.
- planner를 한 번도 부르지 않은 프로세스의 첫 `act`는 `torch.compile`과 CUDA graph 준비를 한다. 그래서 프로세스마다 수집 전에 warm-up `act`를 3번 부르고 결과를 버린다. 그 뒤 에피소드마다 위 seed를 다시 건다.
  이렇게 하면 같은 에피소드는 어느 프로세스, 어느 순서로 돌아도 같은 seed에서 시작한다.
- 공식 `evaluate.py`는 `set_seed` 한 번 뒤 난수 흐름이 이어진다. 이 규칙은 그와 다르다 (EXPERIMENT_PLAN.md V6에 적은 대로). GPU 장치나 프로세스가 달라도 비트 단위로 같은지는 보장하지 않는다. 에피소드마다 수집한 장치를 기록한다.

## 2. 넘어진 상태 (fallen state)

- **정의**: torso_height < **0.6 m**.
- **근거**: walker reward의 standing 항 = tolerance(torso_height, bounds=(1.2, ∞), margin=0.6, sigmoid gaussian)이다.
  - 1.2는 `_STAND_HEIGHT` (`dm_control/suite/walker.py:33`), margin 0.6은 `_STAND_HEIGHT/2` (`:141-143`, backwards는 `tdmpc2/tdmpc2/envs/tasks/walker.py:115-117`)다.
  - torso_height = 0.6 m(= bound − margin)에서 standing 값은 value_at_margin 0.1이다 (`dm_control/utils/rewards.py:22`). 그보다 낮으면 0.1 아래다.
- **보조 기록**: standing 기준 미달(torso_height < 1.2 m) 비율도 함께 적는다. 이것은 fallen 정의로 쓰지 않는다.
- 판정에 쓰는 높이는 s_t를 복원한 `physics.torso_height()`다 (= 관측 `height`).

## 3. In-distribution 정의

- **주 정의**: 행 τ(encoder에 넣은 task embedding)에 대해, source task가 τ인 A(policy rollout) 상태.
- **보조 정의**: source task가 τ인 A와 B 상태.
- in-distribution 표는 행마다 test 상태 집합이 다르다. **in-distribution 표 안에서 행끼리는 비교하지 않는다.**
- C는 source task 꼬리표가 의미 없으므로 어느 정의에도 넣지 않는다.

## 4. 데이터 분할

- 에피소드 단위로 train 60% / validation 20% / test 20%.
  - A·B: (source task, 종류) 조합마다 20 에피소드를 12 / 4 / 4로 나눈다.
  - C: source task 꼬리표를 무시하고 C 전체 100 에피소드를 60 / 20 / 20으로 나눈다.
- 분할 난수: `numpy.random.default_rng(20261026)`. 각 층 안의 에피소드 번호를 섞어 앞에서부터 train, validation, test로 자른다.
- 분할 결과를 `data/split.json`에 저장하고 모든 probe가 같이 쓴다.
- 한 상태에서 만든 5개 latent는 그 상태의 에피소드가 속한 split에 함께 들어간다.

## 5. Probe 설정 (모든 입력 조건에 같은 데이터, 같은 설정)

입력 조건:
- 학습된 encoder의 z(s, τ): 5개 행, 768차원
- raw observation o(s): 24차원. τ와 무관하므로 열마다 1개
- random-init encoder의 z(s, τ): 5개 행, 768차원
- probe 개수: linear 5 × 5 + 5 + 5 × 5 = **55개**, MLP 55 × seed 5 = **275개**

전처리:
- 입력 feature: train split의 평균·표준편차로 표준화. 표준편차가 1e-8보다 작은 차원은 1로 나눈다(값이 사실상 상수인 차원).
- target: train split의 평균·표준편차로 표준화해 학습한다. R²는 원래 단위로 되돌린 예측으로 계산한다.

Linear (ridge):
- λ 후보: 10^k, k = −4, −3.5, …, 4 (17개).
- train으로 학습하고 validation R²가 가장 큰 λ를 고른다. 동점이면 큰 λ를 고른다.
- 고른 λ로 train만으로 다시 학습한다. validation은 학습에 넣지 않는다.
- 절편은 표준화한 target에서 0이다.

MLP:
- 구조: 입력 → 256 → ReLU → 256 → ReLU → 1.
- Adam, 학습률 1e-3, weight decay 0, batch 512.
- 최대 200 epoch. validation MSE가 20 epoch 동안 나아지지 않으면 멈추고, validation MSE가 가장 좋았던 epoch의 가중치를 쓴다.
- seed 0~4: 가중치 초기화와 batch 순서에 쓴다.

## 6. 지표와 통계 판정

- **R²**: 1 − Σ(y − ŷ)² / Σ(y − ȳ)². ȳ는 그 평가 집합의 평균이다.
- **신뢰구간**: test 에피소드를 복원추출하는 bootstrap 1,000회, 95% 백분위 구간. bootstrap 난수 seed 20261027.
- **MLP**: bootstrap 1회마다 그 표본에서 seed 5개 각각의 R²를 계산해 평균한다. 점추정은 seed 5개 R²의 평균이다. seed 간 표준편차를 따로 보고한다.
- **주 기준 (같은 열 안 두 조건 비교)**: paired bootstrap difference.
  - bootstrap 1회마다 test 에피소드를 복원추출하고, 같은 표본에서 두 조건의 R² 차이를 계산한다 (1,000회).
  - 95% 구간이 0을 포함하지 않으면 **"차이 있음"**, 포함하면 **"구분되지 않음"**이다.
  - 비교: (a) 학습된 z(τ) 대 raw observation, (b) 학습된 z(τ) 대 random-init z(τ), (c) 학습된 z의 대각선 칸 대 같은 열의 나머지 행 각각.
- **보조 기준**: 각 조건의 95% 구간이 겹치는지도 함께 적는다.
- **행 방향(같은 행, 다른 열) 비교**는 target 분포가 달라 주 비교로 쓰지 않는다.
- **R² 계산 예외**: 평가 집합의 target 분산이 1e-10보다 작거나, 그 집합에 에피소드가 2개 미만이면 R²를 계산하지 않고 이유를 적는다.
  bootstrap 표본에서 이 조건에 걸리면 그 회차를 버리고 버린 횟수를 적는다.

## 7. Heatmap

- matplotlib 정적 PNG.
- 색은 R² −1 ~ 1 고정 범위이고, 범위 밖 값은 끝 색으로 칠한다.
- 칸 안 숫자는 실제 값을 그대로 적는다 (자르지 않음, 소수 셋째 자리 반올림).
- 행: encoder에 넣은 task embedding, 열: 예측하는 reward의 task, 칸: pooled test R².

## 8. 예상 비용과 저장 공간 (실제 가용량 확인: 2026-10-26)

| 항목 | 예상 | 근거 |
|---|---|---|
| A·B rollout | 200 에피소드 × 500 step = 100,000 planning step | GATE 2: 에피소드당 약 76초(0.152초/step) → 약 4.2 GPU 시간. T4 두 장에 프로세스 하나씩 나누면 약 2.1시간 |
| C rollout | 100 에피소드, planning 없음 | CPU, 수 분 |
| latent 추출 | 150,000 상태 × 5 embedding × 768 × float32 = 약 2.3 GB (학습 모델) + 약 2.3 GB (random-init) | 로컬 CPU에서 계산 |
| rollout 데이터 | 상태당 약 100개 수 (s_t, substep 상태 2개, 관측, 행동, reward) → 압축 전 약 120 MB | – |
| probe 학습 | linear 55개, MLP 275개 (train 약 90,000 상태) | linear는 CPU 수 분. MLP는 probe당 최대 200 epoch: GPU에서 probe당 1~2분 추정 → 약 5~9시간 상한, early stopping으로 더 짧을 수 있음 |

실제 가용량:
- Kaggle 주간 GPU 할당량 30시간 중 사용 1.33시간 → **약 28.7시간 남음** (2026-10-31 00:00 UTC 초기화).
- 로컬 디스크 여유 **912 GB**, 로컬 RAM 15 GB, CPU 12개.
- latent(4.6 GB)는 로컬과 Kaggle 출력에만 두고 GitHub에는 올리지 않는다.

---

## 사후 추가 항목 (2026-10-26 08:31 UTC, Step 6 probe를 하나도 학습하기 전에 기록)

이 절은 위 고정 항목을 바꾸지 않는다. 사용자 지시(Step 6 지시서 6-2의 4)에 따라 보조 대조군 하나를 추가한다.

**A1. norm-matched random-init latent (보조 대조군)**

- 이유: random-init 모델의 task embedding norm이 약 0.1(0.085~0.108)이라, 학습 모델(약 0.99)과 encoder에 들어가는 task 입력의 크기가 다르다 (`results/exp1/latent_meta.json`).
- 정의:
  - 모델은 5절의 random-init 대조군과 같다 (`torch.manual_seed(20261028)` 후 `WorldModel(cfg)`).
  - walker 다섯 task(번호 0, 1, 2, 19, 20)의 task embedding 행 e_k만 e_k × (c / ‖e_k‖)로 바꾼다. 방향은 그대로 두고 크기만 곱한다.
  - c는 학습된 mt30-48M의 같은 다섯 행 L2 norm의 평균 = **0.994675**다.
  - c < 1이라 `nn.Embedding(max_norm=1)`의 조회 시 재정규화(`common/world_model.py:21`)는 일어나지 않는다.
  - encoder와 나머지 가중치는 바꾸지 않는다.
- latent 추출: `code/step5_latents_targets.py`의 `encode_all`(공식 `WorldModel.encode`)을 그대로 쓴다.
  추출 경로가 5절 random-init과 같다는 확인으로, 크기를 바꾸지 않은 random-init 모델을 같은 함수로 다시 encode해 저장된 `latent_random.npy`와 일치하는지 비교한다.
- probe 설정, split, target, 통계는 5~6절과 같다.
- 비교 (d): 같은 열 안에서 학습된 z(τ) 대 norm-matched random-init z(τ)의 paired bootstrap difference.
- 보고: 주 대조군이 아니라 **보조 대조군**으로 보고한다.

**A2. 결과 구분 추가 (사용자 지시 6-4)**: 보조 결과로 fallen 상태(torso_height < 0.6 m)를 뺀 pooled test를 계산한다. 정의는 2절과 같다.
