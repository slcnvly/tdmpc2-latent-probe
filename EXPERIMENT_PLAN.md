# EXPERIMENT_PLAN (초안, GATE 2 수정본)

작성: 2026-10-22, 수정: 2026-10-26 (V6: VERIFICATION_RESULTS.md 반영). Step 4 이후는 사용자가 가설을 쓰고 승인한 뒤에만 실행한다.
이 실험은 "task-conditioned encoder가 task에 따라 정보를 어떻게 다르게 보존하는가"의 진단이다.
공유 encoder bottleneck 가설의 직접 검증이 아니다 (BASELINE_AUDIT.md 4절: mt30 encoder는 task embedding을 입력으로 받는다).

## 1. 선택 task와 근거

walker 5개: walker-stand, walker-walk, walker-run, walker-walk-backwards, walker-run-backwards (mt30 task index 0, 1, 2, 19, 20).

코드로 확인한 것 (BASELINE_AUDIT.md):

| 조건 | 결과 |
|---|---|
| 관측 차원 | 다섯 모두 24 (padding 없음) |
| 행동 차원 | 다섯 모두 6 (masking 없음) |
| 관측 함수 | 같은 `PlanarWalker.get_observation` |
| 초기 상태 분포 | 같은 `initialize_episode` |
| 물리 모델 | 동역학 값이 같다. backwards 둘의 XML은 바닥 plane 크기만 다르고, 같은 상태·행동에서 궤적 차이 0.0 |
| reward | 다섯 모두 torso_height, torso_upright, horizontal_velocity의 함수. 목표 속도와 방향만 다르다 |

판단 보류 사항: 바닥 XML이 다르다는 점은 "물리 모델이 동일"의 엄밀한 의미에서는 차이다. 동역학 차이는 관찰되지 않았다.
cheetah family로 바꿀 이유는 찾지 못했다.

task별 reward가 의존하는 양:

| task | torso_height (= 관측 `height`) | torso_upright (= 관측 `orientations[0]`) | horizontal_velocity (관측에 직접 없음) |
|---|---|---|---|
| walker-stand | 사용 | 사용 | 사용 안 함 |
| walker-walk | 사용 | 사용 | ≥ 1 |
| walker-run | 사용 | 사용 | ≥ 8 |
| walker-walk-backwards | 사용 | 사용 | ≤ −1 |
| walker-run-backwards | 사용 | 사용 | ≤ −8 |

## 2. 체크포인트

- mt30, 크기 1개. 아래 규칙으로 정하고, probe 결과를 보기 전에 고정한다. 규칙은 비용을 재기 전에 정했다.
- 규칙: Step 4의 planning 작업량을 5 task × planning rollout 2종 × 20 에피소드 × 500 step = 100,000 step으로 잡고,
  공식 설정(`compile=true`, warm-up 10 step 이후 평균)의 step당 시간 × 100,000 ≤ 5 GPU 시간(즉 0.18초/step 이하)이고
  최대 GPU 메모리 ≤ 12 GB인 크기 중 **가장 큰 것**.
- 큰 것을 고르는 이유: 논문 보고값에서 mt30 normalized score가 크기에 따라 18.9(1M)에서 71.4(317M)까지 오르므로, 정책이 실제로 task를 수행하는 체크포인트에서 상태가 수집되게 하기 위해서다.
- 결과 (`results/gate1/size_choice.json`): 조건을 만족한 크기는 1M, 5M, 19M, 48M. **48M을 골랐다** (0.167초/step → 100,000 step에 4.6시간, 최대 765 MB).
  317M은 1.03초/step(100,000 step에 28.6시간)으로 조건을 넘었다.

## 3. 측정할 양

용어 (VERIFICATION_PLAN.md, VERIFICATION_RESULTS.md V4):
- **s_t**: 관측 o_t를 계산한 물리 상태 (`physics.get_state()`).
- **r_logged_t**: `env.step(a_t)`가 돌려준 reward = 두 action-repeat substep reward의 합 (`envs/dmcontrol.py:55-60`).
- **r_τ(s_t)**: s_t를 복원하고(`set_state` 후 `forward()`, 시간 적분 없음) task τ의 `get_reward(physics)`를 적용한 값. **probe target은 r_τ(s_t)다.**
  V4에서 확인: r_τ(s_{t+1})은 r_logged_t의 두 번째 substep reward와 같고(차이 0.0), reward 함수는 action에 의존하지 않는다.

Step 4 데이터 (task마다 세 종류):
- (a) 공식 평가 방식: 자기 task embedding, `agent.act(obs, t0=t==0, task=τ)`, `eval_mode=False`(공식 `evaluate.py:80`과 같음).
  실행 경로는 V1 수정 후의 `code/eval_single_task.py`와 같다 (V2에서 공식 경로와 50 step 차이 0.0).
- (b) noise rollout:
  - a_plan_t = `agent.act(obs, t0, task)`의 출력. 이미 [−1, 1]로 잘려 있다 (`tdmpc2.py:206`).
  - 실행 행동 a_t = clip(a_plan_t + ε_t, −1, 1), ε_t ~ N(0, 0.3² I_6). 순서는 planner 출력 → noise 더하기 → [−1, 1] clipping → `env.step`.
  - ε_t는 planner와 다른 전용 난수 생성기(`numpy.random.Generator`, 에피소드 seed로 시작)에서 뽑는다. planner의 torch 난수 흐름을 건드리지 않기 위해서다.
  - planner의 warm start(`_prev_mean`)는 noise 전의 계획 평균이다. 공식 코드 동작 그대로다 (`tdmpc2.py:205`).
  - 저장: a_plan_t, ε_t, a_t를 모두 저장한다.
- (c) 균등분포 U[−1, 1]^6 무작위 행동. planner를 부르지 않는다. 전용 numpy 생성기를 쓴다.
- 에피소드 수: (a)(b)(c) 각각 task당 20개 (비용: (a)(b) 합계 100,000 planning step × 0.167초 ≈ 4.6 GPU 시간).
- timestep마다 저장:
  - physics state s_t (`get_state()` 전체), padding 전·후 관측 o_t (walker는 둘이 같다)
  - a_plan_t, ε_t, a_t
  - r_logged_t 전체: 실제 할인 누적 보상을 나중에 계산할 수 있게 매 step 저장한다
  - 두 substep 각각의 reward와 종료 상태: 실제 `env.step` 안에서 dm_control `step`을 감싸 기록한다 (V4와 같은 방법)
  - 다음 관측 o_{t+1}, source task, rollout 종류, episode ID, timestep, 에피소드 seed
- 수집 직후 검증 (assert):
  - s_t를 복원해 다시 계산한 관측이 저장한 o_t와 같다 (V0 기준과 같은 1e-5).
  - 두 번째 substep 종료 상태를 복원한 r_τ(s_{t+1})이 기록한 두 번째 substep reward와 같다 (1e-6).

Q 값 기록 (Step 5에서 저장 데이터로 계산, 수집 중에는 계산하지 않음):
- 매 step, walker 5개 task embedding 각각에 대해 Q(z(s_t, τ), a_t, e_τ).
  z(s_t, τ) = `model.encode(o_t, τ)`이고, a_t는 실제 실행 행동이다 (b에서는 noise 뒤 행동).
- 함수: `WorldModel.Q(z, a, task, return_type=...)` (`common/world_model.py:188-218`).
- 공식 planner의 집계: `return_type='avg'` (`tdmpc2.py:136`). ensemble 5개 중 `torch.randperm`으로 무작위 2개를 골라 two-hot 역변환 후 평균한다 (`world_model.py:214-218`).
- 이 집계는 무작위다. 그래서 저장은 두 가지로 한다:
  - (1) `return_type='all'`로 5개 head 각각의 값 (`math.two_hot_inv` 적용)
  - (2) 공식 방식의 'avg' (고정 seed 생성기로 고른 2개의 평균)
- 수집 중에 계산하지 않는 이유: `randperm`이 torch 난수를 써서 rollout의 planner 난수 흐름이 공식 경로와 달라지기 때문이다.

중단 대비 저장과 seed 규칙 (Kaggle 세션이 끊겨도 이어서 수집):
- 에피소드 하나를 파일 하나로 저장한다: `data/<task>/<rollout종류>/ep<번호 3자리>.npz`. 임시 파일에 쓴 뒤 이름을 바꾸므로, 끝난 에피소드만 파일로 남는다.
- 시작할 때 이미 있는 파일의 에피소드는 건너뛴다.
- 에피소드 seed = 10000 + 1000 × (mt30 task 번호) + 100 × (종류: a 0, b 1, c 2) + 에피소드 번호.
  에피소드 시작 시 이 seed로 다음을 다시 정한다:
  - 그 task env의 초기 상태 난수: `task._random = np.random.RandomState(seed)`
  - torch 난수: `torch.manual_seed(seed)`
  - (b)·(c)의 numpy 생성기
- 그래서 에피소드 결과는 세션 경계나 앞선 에피소드 수에 의존하지 않는다.
- **공식 평가와 다른 점**: 공식 `evaluate.py`는 `set_seed` 한 번 뒤 난수 흐름이 이어진다. 이 차이는 데이터 수집용 seed 규칙이고, 공식 점수 재현에는 쓰지 않는다.

Step 5 latent와 reward target:
- 모든 상태를 합친 pooled 집합 S. 각 s_t에 walker 5개 task embedding으로 z(s_t, τ) = `model.encode(o_t, τ)`.
- 대조군 latent: 같은 구조의 random-init `WorldModel`(seed 고정)의 `encode`.
- reward target: r_τ'(s_t) (5개 τ'). s_t를 복원해 τ'의 `get_reward(physics)`를 적용한다.
  - source task 일치 검증: r_source(s_{t+1})을 그 step의 두 번째 substep reward와 비교한다 (V4와 같은 방법).
  - r_logged_t와는 직접 비교하지 않는다 (두 substep의 합이라 다른 값).

Step 6 probe target: 관측 성분 묶음 `orientations`(14), `height`(1), `velocity`(9) — 코드상 이름 그대로 — 와 r_τ'(s) 5개.
입력: z(s, τ) (τ마다 따로), padding을 뺀 raw 관측(walker는 padding이 없어 24차원 그대로), random-init encoder의 z.

Step 7: ‖z(s, τ) − z(s, τ')‖ 분포와 두 기준 거리 (같은 τ에서 무작위로 짝지은 서로 다른 상태, 같은 에피소드의 인접 timestep).

## 4. 가설

(사용자 작성)

## 5. 대조군

| 대조군 | 무엇을 가르는가 |
|---|---|
| raw 관측 probe | encoder 없이 관측만으로 얻는 R² 기준 |
| random-init encoder (seed 고정, 같은 구조, 같은 task_dim) | 학습 없이 구조만으로 얻는 R² 기준 |
| in-distribution 부분집합 (task τ에서 모은 상태에 embedding τ) | pooled 집합에는 해당 task가 학습 중 본 적 없는 상태가 섞인다 |
| rollout 종류별 결과 | 정책 상태(a), 근방(b), 무작위(c)의 분포 차이 |

## 6. 교란 요인

- **상태 분포**: task마다 정책이 방문하는 상태가 다르다. pooled 집합에서 embedding τ'를 넣은 z는 τ'의 학습 분포 밖 상태를 포함한다.
- **task embedding을 바꾼 입력**: z(s, τ')는 학습 중 나오지 않은 (상태, task) 조합일 수 있다. Step 7 결과마다 명시한다.
- **SimNorm**: latent는 8차원 묶음마다 simplex 위에 있다. 거리와 선형 probe의 해석이 일반 실수 벡터와 다르다.
- **horizontal_velocity는 관측에 직접 없다**: walk·run·backwards reward의 핵심 입력이 관측 성분이 아니므로, reward R²와 관측 성분 R²의 차이가 이 때문일 수 있다.
- **reward 정의**: r_logged_t(두 substep 합)와 probe target r_τ(s_t)(복원한 상태 하나의 reward)는 다른 값이다 (VERIFICATION_RESULTS.md V4).
- **planning noise**: 공식 평가는 `eval_mode=False`라 (a)에도 이미 탐색 noise가 있다.
- **체크포인트 하나**: 크기 하나, 학습 시드 하나다.

## 7. 결과 해석의 한계 (실행 전에 적어 둔 것)

- 선형·MLP probe의 R²는 "그 probe가 꺼낼 수 있는 정보"의 하한이다. R²가 낮다고 정보가 없다고 말할 수 없다.
- task-conditioned encoder 하나의 결과로 task-independent encoder에 대해 말할 수 없다.
- walker 5개는 같은 몸체·같은 관측이다. 다른 몸체로 일반화할 수 없다.
- 공식 mt30 task별 점수가 없어, 체크포인트가 각 walker task를 "공식 수준으로" 수행하는지는 단일 task 학습 곡선과의 비교로만 가늠한다 (INITIAL_RESULTS.md).

## 8. 사전 확정한 통계 기준

- split: 에피소드 단위 train 60 / validation 20 / test 20, source task와 rollout 종류로 층화. probe 학습 전에 고정해 파일로 저장.
- Linear: ridge, λ는 train 에피소드 안에서 교차검증 (에피소드 단위 5-fold, λ ∈ 10^[−4, 4] 로그 간격 17개).
- MLP: 은닉 256-256, ReLU, Adam(lr 1e-3), batch 256, 최대 200 epoch, validation R²가 20 epoch 동안 나아지지 않으면 멈춤.
- 모든 입력 조건에 같은 데이터, 같은 학습 예산. target은 train 평균·표준편차로 표준화.
- 지표: test R², probe seed 5개 평균. 95% 신뢰구간은 test 에피소드 복원추출 bootstrap 1,000회.
- 신뢰구간이 겹치는 차이는 "구분되지 않음"으로 적는다.
- 보고: 전체 pooled test 집합, in-distribution 부분집합.
