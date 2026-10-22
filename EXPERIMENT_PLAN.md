# EXPERIMENT_PLAN (초안, GATE 1 제출용)

작성: 2026-10-22. Step 4 이후는 사용자가 가설을 쓰고 승인한 뒤에만 실행한다.
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

Step 4 데이터 (task마다 세 종류):
- (a) 공식 평가 방식: 자기 task embedding, `agent.act(obs, t0, task)`, `eval_mode=False`(공식 `evaluate.py`와 같음).
- (b) (a)의 행동에 Gaussian noise 추가. 표준편차 **0.3** (행동 범위 [−1, 1]), 더한 뒤 [−1, 1]로 자른다. 실행 전 고정.
- (c) 균등분포 [−1, 1]^6 무작위 행동.
- 에피소드 수: (a)(b)는 task당 20개(2절 비용 규칙과 같은 수), (c)는 task당 20개. 비용 실측 결과에 따라 GATE 1에서 조정을 제안할 수 있다.
- timestep마다 저장: physics state(qpos, qvel, 전체 `get_state()`), padding 전·후 관측, action, 자기 task reward(env step reward와 두 substep reward 각각),
  다음 관측, source task, rollout 종류, episode ID, timestep, seed.

Step 5 latent와 reward target:
- 모든 상태를 합친 pooled 집합 S. 각 s에 walker 5개 task embedding으로 z(s, τ) = `model.encode(obs(s), τ)`.
- 대조군 latent: 같은 구조의 random-init `WorldModel`(seed 고정)의 `encode`.
- reward target r_τ'(s): s를 복원하고 τ'의 `task.get_reward(physics)`를 그대로 적용한 값 (physics 한 substep의 reward).
  env step의 기록 reward는 두 substep의 합이라 이 값과 다르다 (BASELINE_AUDIT.md 6절). 일치 검증은 기록한 두 번째 substep reward와 한다.

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
- **reward 정의**: 기록 reward(두 substep 합)와 r_τ'(s)(한 substep)가 다르다.
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
