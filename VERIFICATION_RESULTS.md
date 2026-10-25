# VERIFICATION_RESULTS (GATE 2)

작성: 2026-10-26. 판정 기준은 [VERIFICATION_PLAN.md](VERIFICATION_PLAN.md)(커밋 `989964a`, 실행 전)를 그대로 썼다.
Step 4 이후(데이터 수집, probe, task swap)는 실행하지 않았다.

## 요약

| 검사 | 판정 | 근거 |
|---|---|---|
| V1-1 task 번호 → embedding 행 | **같음** | 세 경로 모두 0, 1, 2, 19, 20 |
| V1-2 ~ V1-9 코드 대조 | 차이 3개 발견, **자체 경로 수정** | seed 호출 순서, `cudnn.benchmark`, return 합산 방식 |
| V2 trace (walker-walk) | **통과** | 50 step 동안 관측·encoder 입력·행동 최대 차이 0.0 |
| V2 trace (walker-run) | **통과** | 같음, 0.0 |
| V2 trace (walker-run-backwards) | **통과** | 같음, 0.0 (공식 실행의 난수 상태를 복원해 비교) |
| V3 walker-stand | 일치 | 공식 996.88이 자체 5 에피소드 범위 [982.03, 996.88] 안 (끝값과 같음) |
| V3 walker-walk | **불일치** | 공식 990.48이 범위 [963.99, 990.25] 밖 (0.24 위) |
| V3 walker-run | 일치 | 공식 602.38, 범위 [502.82, 623.76] |
| V3 walker-walk-backwards | 일치 | 공식 981.77, 범위 [931.65, 984.99] |
| V3 walker-run-backwards | 일치 | 공식 222.64, 범위 [164.46, 314.12] |
| V4-2 r_τ(s_{t+1}) = 마지막 substep reward | **통과** | 다섯 task 최대 차이 0.0 |
| V4-4 reward의 action 의존 | **통과** (의존하지 않음) | 코드: `get_reward(self, physics)`. 수치: ctrl만 바꾸면 변화 0.0 |
| V5 | 기록만 (판정 없음) | 아래 V5 절 |

V3의 비교 대상은 V1 수정 후 다시 돌린 Experiment 0이다. 수정 전 결과로 판정하면 walker-walk와 walker-run-backwards가 불일치다 (V3 절 표).

## V1. 코드 수준 대조

공식 경로는 `tdmpc2/tdmpc2/evaluate.py`와 그것이 부르는 코드(커밋 `332f19e`), 자체 경로는 `code/eval_single_task.py`(V1 수정 후)와 `code/common_setup.py`다.
판정은 "같음" / "다름(수정함)" / "다름(수정 안 함, 이유)" 중 하나다.

### V1-1. Task 번호 → embedding 행

| task | (a) `TASK_SET['mt30']` 위치 (`common/__init__.py:29, 34`) | (b) wrapper 번호 | (c) embedding 행 | 판정 |
|---|---|---|---|---|
| walker-stand | 0 | 0 | 0 | 같음 |
| walker-walk | 1 | 1 | 1 | 같음 |
| walker-run | 2 | 2 | 2 | 같음 |
| walker-walk-backwards | 19 | 19 | 19 | 같음 |
| walker-run-backwards | 20 | 20 | 20 | 같음 |

- (b) 공식 wrapper: `make_multitask_env`가 `cfg.tasks` 순서대로 env를 만들어 리스트에 넣는다 (`envs/__init__.py:43-50`).
  `MultitaskWrapper.reset(task_idx)`가 `self.envs[task_idx]`를 고른다 (`envs/wrappers/multitask.py:38-39, 49-53`).
  `evaluate.py:71, 76, 80`은 `enumerate(cfg.tasks)`의 번호를 그대로 `reset`과 `act`에 넘긴다.
  확인: 번호 2의 env는 `PlanarWalker(move_speed=8)`, 번호 20은 `BackwardsPlanarWalker(move_speed=8)` (`results/audit_checks.json` `walker_state_restore_checks`).
- (c) checkpoint의 `_task_emb.weight` 크기는 **30 × 64** (mt30-48M, `results/v5/task_embedding_norms_mt30_48M.csv`).
  자체 경로는 `agent.act(obs, t0, task=k)`로 같은 번호 k를 넘기고, `task_emb`가 `self._task_emb(task.long())`로 행 k를 꺼낸다 (`common/world_model.py:96-98`).
  자체 경로 `eval_single_task.py:87`, 공식 `evaluate.py:80`.
- 세 경로가 같은 행을 가리킨다.

### V1-2 ~ V1-7. 대조표

| 항목 | 공식 경로 | 자체 경로 | 판정 |
|---|---|---|---|
| 관측 성분 순서 | dm_control `get_observation`의 OrderedDict 순서(orientations 14, height 1, velocity 9)로 이어 붙임 (`dmcontrol.py:47-49`) | 같은 env 객체 | 같음 |
| 관측 정규화 | 없음 (`dmcontrol.py:47-49`, `tensor.py:19-24`) | 같은 env 객체 | 같음 |
| 관측 dtype | float32 (`dmcontrol.py:49`) | 같음 | 같음 |
| 관측 padding | 0으로 채워 24 (`multitask.py:44-47`). walker는 0개 | 같은 wrapper | 같음 |
| 행동 clipping | planner 출력 `a.clamp(-1, 1)` (`tdmpc2.py:206`) | 같은 `agent.act` | 같음 |
| 행동 자르기 | `action[:env.action_space.shape[0]]` (`multitask.py:56`). walker는 6 → 6 | 같은 wrapper | 같음 |
| 행동 변환 | `.numpy()` (`tensor.py:38`), `astype(action_spec_dtype)` (`dmcontrol.py:56`), `action_scale.Wrapper`로 [−1, 1] → 구동기 범위 (`dmcontrol.py:106`). walker 구동기 범위 [−1, 1] | 같음 | 같음 |
| action repeat | 2 (`dmcontrol.py:57-59`) | 같음 | 같음 |
| `agent.act` 인자 | `act(obs, t0=t==0, task=task_idx)`, `eval_mode` 생략 → False (`evaluate.py:80`) | `act(obs, t0=t == 0, task=task_idx)` (`eval_single_task.py:87`) | 같음 |
| t0 처리 | t0=True면 이전 MPPI 평균을 쓰지 않음 (`tdmpc2.py:166-167`) | 같음 | 같음 |
| planner 설정 | `config.yaml:34-42`: iterations 6, num_samples 512, num_elites 64, num_pi_trajs 24, horizon 3, min_std 0.05, max_std 2, temperature 0.5. iterations는 action_dim ≥ 20일 때만 +2 (`tdmpc2.py:34`) → walker 6 | 같은 `config.yaml`을 compose (`common_setup.py:20-31`) | 같음 |
| discount | task별: `_get_discount(500)` = clip((100−1)/100, 0.95, 0.995) = **0.99** (`tdmpc2.py:35-37, 57-70`, `config.yaml:26-28`). walker 5개 모두 0.99 (커널 로그의 `Discount factor` 출력) | 같음 | 같음 |
| Q 집계 (planner) | `Q(z, a, task, return_type='avg')` (`tdmpc2.py:136`): ensemble 5개 중 무작위 2개의 평균 (`world_model.py:214-218`) | 같음 | 같음 |
| 에피소드 길이 | `Timeout(max_episode_steps=500)` (`dmcontrol.py:110`). env 500 step = physics control step 1000 (25초 / 0.025초) | 같음 | 같음 |
| return 합산 | `ep_reward = 0`; `ep_reward += reward` (reward는 float32 tensor, `evaluate.py:76, 82`; `tensor.py:42`) | **V1 전**: `float(reward)`를 Python float로 합산 → **V1 후**: 공식과 같은 방식 (`eval_single_task.py:79, 94`) | **다름(수정함)** |
| seed 위치 | `set_seed(cfg.seed)` (`evaluate.py:45`: random, numpy, torch, cuda, `common/seed.py:6-10`) → `make_env` (`:54`) → `TDMPC2(cfg)` (`:57`). env 초기 상태는 task별 `task_kwargs={'random': cfg.seed}` (`dmcontrol.py:104`) | **V1 전**: `make_env` → `set_seed` → `TDMPC2` → **V1 후**: 공식 순서 (`eval_single_task.py:38-42`) | **다름(수정함)** |
| `cudnn.benchmark` | `torch.backends.cudnn.benchmark = True` (`evaluate.py:17`) | **V1 전**: 설정 안 함 → **V1 후**: 설정 (`eval_single_task.py:33`) | **다름(수정함)** |
| config 구성 | `@hydra.main` + `parse_cfg` (`evaluate.py:20, 44`) | Hydra compose API + 같은 `parse_cfg`. `get_original_cwd`만 `os.getcwd`로 바꿈 (`work_dir` 계산에만 쓰임, `parser.py:57`) (`common_setup.py:20-31`) | 다름(수정 안 함): 값이 같은 config가 만들어지고, 공식 `@hydra.main`은 스크립트 밖에서 쓸 수 없다 |
| `compile` | 기본 `true` (`config.yaml:74`) | `compile=true` | 같음 |

V1 전 버전은 `code/eval_single_task_pre_v1.py`에 남겨 두었다.
수정이 있었으므로 Experiment 0을 같은 seed(1)·같은 에피소드 수(task당 5)로 다시 실행했다 (V5-1, `results/gate2/exp0_rerun_components.json`).

### V1-8. 소프트웨어 버전

| 패키지 | `docker/environment.yaml` | Kaggle 실행 환경 (`results/gate2/pip_freeze.txt`) | 판정 |
|---|---|---|---|
| python | 3.11 | 3.11 (uv) | 같음 |
| torch | 2.7.1 | 2.7.1+cu128 | 같음 |
| dm-control | 1.0.16 | 1.0.16 | 같음 |
| mujoco | 3.1.2 | 3.1.2 | 같음 |
| gymnasium | 0.29.1 | 0.29.1 | 같음 |
| numpy | 1.24.4 | 1.24.4 | 같음 |
| tensordict | 0.8.3 | 0.8.3 | 같음 |
| hydra-core | 1.3.2 | 1.3.2 | 같음 |
| omegaconf | 2.3.0 | 2.3.0 | 같음 |
| torchrl | 0.8.1 | 설치 안 함 | 다름(수정 안 함): `common/buffer.py`(학습용)만 import하고 평가 경로는 쓰지 않는다 |
| GPU / 드라이버 | (명시 없음) | Tesla T4 | – |

공식 경로와 자체 경로는 같은 Kaggle 세션, 같은 venv에서 돌았다.

### V1-9. checkpoint sha256

`multitask/mt30-48M.pt`: `bacb9ae1891a9b03dc3942a287080beb573220ea9d0263c5efd7a94a5c2ec419`.
로컬 파일과 Kaggle에서 받은 파일(`results/gate2/ckpt_sha256.txt`), HF LFS oid(`bacb9ae1891a9b03...`)가 같다.

## V2. Trace equivalence

### 방법

- **공식 경로**: `evaluate.py`를 수정 없이 `runpy`로 실행했다 (`code/official_eval_hooked.py`).
  인자: `task=mt30 model_size=48 checkpoint=<mt30-48M.pt> eval_episodes=1 seed=1 save_video=false enable_wandb=false`.
  hook은 `TDMPC2.act`와 `MultitaskWrapper.step`을 감싸 값을 복사만 한다.
  encoder 입력은 `model.task_emb(obs, task)`로 계산했다. `encode`가 encoder에 넣는 텐서와 같은 함수다 (`world_model.py:110-111`). 이 계산은 난수를 쓰지 않는다.
- **자체 경로**: `code/eval_single_task.py trace` (V1 수정 후). 같은 Kaggle 세션, 같은 GPU, 같은 venv에서 공식 실행 직후에 돌렸다.
  - walker-stand, walker-walk, walker-run을 공식과 같은 순서로 전 에피소드 돌렸다. 공식 실행도 이 셋이 task 0, 1, 2라 맨 앞에 온다.
  - walker-run-backwards(task 20)는 공식 실행에서 그 task 첫 act 직전에 저장한 torch CPU·CUDA 난수 상태를 복원한 뒤 돌렸다.
- 기록: 각 task 첫 에피소드의 처음 50 step. 관측(24), encoder 입력(24 + 64 = 88), task 번호, 행동(6).
- deterministic 설정은 따로 켜지 않았다. 공식 `evaluate.py`가 켜지 않기 때문이다.

### 결과 ([results/gate2/v2_trace_compare.json](results/gate2/v2_trace_compare.json), [v2_trace_diff.png](results/gate2/v2_trace_diff.png))

| task | 비교 step 수 | 관측 최대 \|차이\| | encoder 입력 최대 \|차이\| | 행동 최대 \|차이\| | task 번호 | 판정 |
|---|---|---|---|---|---|---|
| walker-walk | 50 | 0.0 | 0.0 | 0.0 | 50 step 모두 1 | 통과 |
| walker-run | 50 | 0.0 | 0.0 | 0.0 | 50 step 모두 2 | 통과 |
| walker-run-backwards | 50 | 0.0 | 0.0 | 0.0 | 50 step 모두 20 | 통과 |

- 50 step 전부에서 차이가 정확히 0이다. 그래프에서는 0을 1e-12 위치에 그렸다.
- hook 기록 중 오류는 없었다 (`official_hooked.json` `errors: []`).

### 보조 기록 (판정에 쓰지 않음): 전 에피소드 return

자체 경로는 trace 뒤에도 에피소드를 끝까지 돌렸다. return은 `evaluate.py:76, 82`와 같이 float32로 한 step씩 누적했다.

| task | 공식 1 에피소드 return | 자체 경로 trace 에피소드 return |
|---|---|---|
| walker-stand | 996.876343 | 996.876343 |
| walker-walk | 990.483276 | 990.483276 |
| walker-run | 602.381531 | 602.381531 |
| walker-run-backwards | 222.635315 | 222.635315 |

## V3. 공식 evaluate.py 30-task 평가 (승인된 1회)

### 설정과 비용

- **명령**: `python evaluate.py task=mt30 model_size=48 checkpoint=<mt30-48M.pt> eval_episodes=1 seed=1 save_video=false enable_wandb=false`.
  - `eval_episodes=1`은 config 인자라 코드 수정 없이 설정됐다.
  - `save_video=false`와 `enable_wandb=false`도 config 인자다. 영상 저장과 로그 업로드를 끈 것으로, 평가 계산과는 관계없다.
- **실행 횟수**: 30-task 평가는 이 1회뿐이다.
- **실행 시간**: 2,256.7초(37.6분). 30개 env 생성, 모델 로드, compile을 포함한다.
- **GPU**: Tesla T4. 사용한 쪽 GPU의 `nvidia-smi` 메모리 최대 707 MiB, 사용률 평균 97.1%(5초 간격 451회).
  torch `max_memory_allocated` 562 MB. 다른 한 장은 쓰이지 않았다(최대 3 MiB, 0%).

### 30개 task 점수 ([results/gate2/official_stdout.txt](results/gate2/official_stdout.txt), [v3_scores.json](results/gate2/v3_scores.json))

| task | return | task | return | task | return |
|---|---|---|---|---|---|
| walker-stand | 996.9 | cartpole-swingup | 0.6 | walker-run-backwards | 222.6 |
| walker-walk | 990.5 | cartpole-swingup-sparse | 0.0 | cheetah-run-backwards | 488.4 |
| walker-run | 602.4 | cup-catch | 902.0 | cheetah-run-front | 102.7 |
| cheetah-run | 208.2 | finger-spin | 969.0 | cheetah-run-back | 256.4 |
| reacher-easy | 990.0 | finger-turn-easy | 988.0 | cheetah-jump | 318.1 |
| reacher-hard | 988.0 | finger-turn-hard | 987.0 | hopper-hop-backwards | 354.8 |
| acrobot-swingup | 22.7 | fish-swim | 748.7 | reacher-three-easy | 980.0 |
| pendulum-swingup | 942.0 | hopper-stand | 477.0 | reacher-three-hard | 923.0 |
| cartpole-balance | 996.2 | hopper-hop | 58.0 | cup-spin | 833.5 |
| cartpole-balance-sparse | 1000.0 | walker-walk-backwards | 981.8 | pendulum-spin | 906.7 |

- return은 에피소드(500 env step)의 r_logged_t 합이다. 한 env step의 reward는 0~2(두 substep 각각 0~1)라, 에피소드 return의 범위는 0~1000이다.
- **normalized score**: `evaluate.py:94, 99`의 정의(DMControl task는 return / 10, 30개 평균)로 **64.12**.
  `evaluate.py`가 출력한 값도 64.12이고, 기록한 reward로 다시 계산한 값은 64.117이다.
- **논문 값**(mt30 48M, Figure 7·22)은 **59.4**다. 이번 값은 task당 1 에피소드라 비교가 거칠다. 이 평균으로는 판정하지 않는다.

### Walker 5개 판정 (V0 기준: 공식 1 에피소드 return이 자체 5 에피소드의 최솟값~최댓값 안이면 일치)

| task | 공식 1 에피소드 | 자체 V1 후 5 에피소드 (최소 ~ 최대) | V1 후 판정 | 자체 V1 전 5 에피소드 (최소 ~ 최대) | V1 전 판정 |
|---|---|---|---|---|---|
| walker-stand | 996.876 | 982.03 ~ 996.876 | 일치 (최댓값과 같음) | 981.72 ~ 996.99 | 일치 |
| walker-walk | 990.483 | 963.99 ~ 990.25 | **불일치** | 967.15 ~ 988.76 | 불일치 |
| walker-run | 602.382 | 502.82 ~ 623.76 | 일치 | 565.73 ~ 654.16 | 일치 |
| walker-walk-backwards | 981.770 | 931.65 ~ 984.99 | 일치 | 944.77 ~ 982.85 | 일치 |
| walker-run-backwards | 222.635 | 164.46 ~ 314.12 | 일치 | 163.28 ~ 191.47 | 불일치 |

- 공식 return은 `evaluate.py:76, 82`와 같이 float32로 한 step씩 누적해 계산했다.
- **비교 코드 수정 기록**: 처음 비교 코드는 기록한 reward를 `np.sum`(합산 순서가 다름)으로 더해 walker-stand가 996.8766으로 나왔다. 이 값이면 자체 최댓값 996.8763보다 0.0002 커서 범위 밖이다.
  `evaluate.py`의 누적 방식으로 고친 값은 996.876343이고 자체 최댓값과 같다. 판정 기준은 바꾸지 않았고, 공식 점수를 계산하는 방식만 공식 코드에 맞췄다.
- walker-walk 불일치: 공식 990.483은 자체 최댓값 990.247보다 0.236 크다.
- 1 에피소드 대 5 에피소드라는 한계: 공식 값 하나가 범위 밖이어도 우연일 수 있고, 범위 안이어도 같은 분포라는 뜻은 아니다.
- 관찰 (판정에 쓰지 않음):
  - 자체 V1 후 Experiment 0의 첫 에피소드 return이 공식과 소수점 여섯째 자리까지 같은 task가 둘이다.
    - walker-stand(996.876343): 두 실행 모두 `set_seed` 직후 첫 에피소드다.
    - walker-run-backwards(222.635315): 두 실행 모두 이 에피소드 전에 다른 task 에피소드를 20개(각 500 step) 돌았다.
  - 나머지 셋의 첫 에피소드는 앞서 돈 에피소드 수가 다르고, 값도 다르다.

## V4. Reward 시점 검증 ([results/v4_reward_timing.json](results/v4_reward_timing.json), [code/v4_reward_timing.py](code/v4_reward_timing.py))

### 용어

- **s_t**: 관측 o_t를 계산한 물리 상태.
- **r_logged_t**: `env.step(a_t)`가 돌려준 reward. 두 substep reward의 합이다.
- **r_τ(s_t)**: s_t를 복원하고 시간 적분 없이 τ의 reward 함수를 적용한 값.

### 방법

- walker 5개 task에서 무작위 행동 40 step rollout을 1개씩 돌렸다 (seed 0, CPU, mujoco 3.1.2).
- 실제 `env.step` 안에서 `DMControlWrapper`가 부르는 dm_control `step`을 감싸, 각 substep의 reward와 종료 상태를 기록했다.

### 결과

| task | 4-2: \|r_τ(s_{t+1}) − 마지막 substep reward\| 최대 | 마지막 substep 종료 상태 = s_{t+1} | \|r_logged_t − substep 합\| 최대 | \|r_τ(s_t) − r_logged_t\| 평균 | 같은 값 최대 | 4-4: ctrl만 바꿨을 때 r_τ(s) 변화 최대 |
|---|---|---|---|---|---|---|
| walker-stand | 0.0e+00 | True | 5.4e-08 | 0.339 | 1.005 | 0.0e+00 |
| walker-walk | 0.0e+00 | True | 2.9e-08 | 0.113 | 0.563 | 0.0e+00 |
| walker-run | 0.0e+00 | True | 1.5e-08 | 0.063 | 0.179 | 0.0e+00 |
| walker-walk-backwards | 0.0e+00 | True | 1.5e-08 | 0.109 | 0.252 | 0.0e+00 |
| walker-run-backwards | 0.0e+00 | True | 1.5e-08 | 0.064 | 0.167 | 0.0e+00 |
- **4-2 통과**: 다섯 task 모두 0.0 (기준 ≤ 1e-6). 40 step 모두 마지막 substep 종료 상태가 s_{t+1}과 비트 단위로 같다.
- **r_logged_t는 두 substep reward의 합이다.** float32 변환 오차 수준(최대 5.4e-8)으로 같다.
- **4-4 통과**:
  - 코드: walker 다섯 task의 reward 함수는 `get_reward(self, physics)`로 action을 인자로 받지 않는다 (`dm_control/suite/walker.py:139`, `tdmpc2/tdmpc2/envs/tasks/walker.py:114`).
  - 수치: 같은 상태에서 `physics.data.ctrl`만 무작위로 20번 바꾸고 `forward()`했을 때 r_τ(s) 변화가 다섯 task 모두 0.0이다 (기준 ≤ 1e-9).
  - r_logged_t는 중간 substep 상태를 거치므로 a_t에 따라 달라진다. 이 성질은 reward 함수가 아니라 action repeat에서 온다.

4-3. r_τ(s_t)와 r_logged_t가 다른 값임을 보이는 예시 (walker-walk, 처음 5 step):

| t | r_τ(s_t) | substep 1 reward | substep 2 reward (= r_τ(s_{t+1})) | r_logged_t | r_τ(s_{t+1}) |
|---|---|---|---|---|---|
| 0 | 0.1639 | 0.1634 | 0.1623 | 0.3258 | 0.1623 |
| 1 | 0.1623 | 0.1629 | 0.1644 | 0.3273 | 0.1644 |
| 2 | 0.1644 | 0.1657 | 0.1664 | 0.3322 | 0.1664 |
| 3 | 0.1664 | 0.1664 | 0.1693 | 0.3357 | 0.1693 |
| 4 | 0.1693 | 0.1606 | 0.2354 | 0.3960 | 0.2354 |
- r_logged_t는 r_τ 값 두 개를 더한 것이라 r_τ(s_t)의 약 두 배 크기다. 평균 차이는 위 표의 "\|r_τ(s_t) − r_logged_t\| 평균" 열에 있다.
- **추가 기록 (V0 기준 아님)**: 저장한 s_t를 복원하고 같은 행동으로 두 substep을 다시 돌렸을 때, 실제 s_{t+1}과의 최대 상태 차이는 6.1e-14다.
  0은 아니다. 관측·reward 재계산은 복원만 하면 되므로 이 차이의 영향을 받지 않는다.

## V5. 추가 기록 (판정에 쓰지 않음, 해석 없음)

### V5-1. Reward 성분 분해 ([results/gate2/exp0_rerun_components.json](results/gate2/exp0_rerun_components.json))

V1 수정 후 Experiment 0 재실행이다. 설정은 mt30-48M, seed 1, task당 5 에피소드, `compile=true`, `eval_mode=False`.
매 env step 뒤의 상태 s_{t+1}에서 성분을 계산했고, 성분으로 다시 조립한 reward가 `task.get_reward`와 1e-9 이내로 같음을 매 step 확인했다(`assert`).

성분 정의:
- **standing** = tolerance(torso_height, [1.2 m, ∞), margin 0.6 m, gaussian) (`dm_control/suite/walker.py:141-143`). 0~1.
- **upright** = (1 + torso_upright) / 2 (`:144`). torso_upright는 torso z축과 세계 z축의 cos. 0~1.
- **move**는 tolerance(horizontal_velocity, 목표, margin = speed/2, value_at_margin 0.5, linear)다. 0~1.
  - 앞으로 가는 task (`:149-153`): 목표 [speed, ∞).
  - backwards task (`envs/tasks/walker.py:123-127`): 목표 (−∞, −speed].
  - walker-stand에는 없다.
- **horizontal_velocity**: torso 하위 트리 질량중심의 x 방향 속도, 단위 m/s (`walker.py:92-94`). 양수가 앞(+x)이다.
- **torso 높이 < 1.2 m 비율**: standing 기준 아래인 step의 비율.

| task | 에피소드 return (5개) | standing 평균 | upright 평균 | move 평균 | 수평 속도 평균 ± 표준편차 (m/s) | 높이 < 1.2 m 비율 |
|---|---|---|---|---|---|---|
| walker-stand | 996.9, 996.7, 983.8, 982.0, 983.5 | 0.986 ~ 1.000 | 0.969 ~ 0.988 | – | −0.07 ~ 0.43 (±0.19 ~ 0.77) | 0.000 ~ 0.024 |
| walker-walk | 990.2, 980.1, 975.0, 964.0, 975.5 | 0.983 ~ 1.000 | 0.976 ~ 0.996 | 0.963 ~ 0.990 | 1.34 ~ 1.45 (±0.27 ~ 0.40) | 0.000 ~ 0.034 |
| walker-run | 623.8, 502.8, 615.6, 564.2, 615.6 | 0.982 ~ 0.997 | 0.973 ~ 0.992 | 0.406 ~ 0.553 | 2.96 ~ 4.42 (±1.10 ~ 2.40) | 0.266 ~ 0.418 |
| walker-walk-backwards | 985.0, 980.4, 931.6, 964.1, 942.2 | 0.984 ~ 1.000 | 0.972 ~ 0.992 | 0.924 ~ 0.984 | −1.06 ~ −0.94 (±0.12 ~ 0.50) | 0.000 ~ 0.038 |
| walker-run-backwards | 222.6, 314.1, 188.0, 181.9, 164.5 | 0.981 ~ 0.998 | 0.970 ~ 0.991 | 0.002 ~ 0.187 | −1.23 ~ 1.13 (±0.65 ~ 1.83) | 0.120 ~ 0.430 |

- 범위는 5 에피소드의 에피소드별 평균값 중 최솟값 ~ 최댓값이다. 에피소드별 값은 JSON에 있다.
- 목표 속도: walk 1 m/s 이상, run 8 m/s 이상, walk-backwards −1 m/s 이하, run-backwards −8 m/s 이하.
- walker-run-backwards의 에피소드별 수평 속도 평균: −0.25, −1.23, +0.79, +0.94, +1.13 m/s.

### V5-2. Task embedding 유사도 ([results/v5/task_embedding_cosine_mt30_48M.csv](results/v5/task_embedding_cosine_mt30_48M.csv), [norms CSV](results/v5/task_embedding_norms_mt30_48M.csv))

- mt30-48M `_task_emb.weight` (30 × 64)의 30 × 30 cosine similarity 전체와 행별 L2 norm을 CSV로 저장했다.
- 저장된 가중치 그대로이며, `max_norm=1` 재정규화 전 값이다. 모든 norm이 1 이하라 조회 시 바뀌지 않는다.

walker 5개 (cosine similarity, 소수 넷째 자리):

| | stand | walk | run | walk-backwards | run-backwards |
|---|---|---|---|---|---|
| walker-stand | 1.0000 | 0.7664 | 0.4895 | 0.6718 | 0.4356 |
| walker-walk | 0.7664 | 1.0000 | 0.5650 | 0.6244 | 0.3691 |
| walker-run | 0.4895 | 0.5650 | 1.0000 | 0.3901 | 0.5851 |
| walker-walk-backwards | 0.6718 | 0.6244 | 0.3901 | 1.0000 | 0.5468 |
| walker-run-backwards | 0.4356 | 0.3691 | 0.5851 | 0.5468 | 1.0000 |

벡터 길이 (L2 norm): stand 0.9951, walk 0.9920, run 0.9976, walk-backwards 0.9925, run-backwards 0.9961.

### V5-3. Policy prior 단독 평가 ([results/gate2/policy_prior_eval.json](results/gate2/policy_prior_eval.json))

- 공식 설정: config 인자 `mpc` (`config.yaml:34`). `mpc=false`면 `act`가 planning 없이 encode 후 policy prior에서 행동을 뽑는다 (`tdmpc2.py:114-120`).
- `eval_mode=False`(공식 `evaluate.py`와 같음)라 prior 평균이 아니라 표본 행동이다.
- 실행: mt30-48M, seed 1, task당 5 에피소드. 에피소드당 약 1.4초.

| task | 에피소드 return (5개) | 평균 | move 평균 범위 | 수평 속도 평균 범위 (m/s) |
|---|---|---|---|---|
| walker-stand | 998.0, 956.1, 979.3, 982.3, 979.5 | 979.1 | – | −0.14 ~ 0.03 |
| walker-walk | 201.6, 235.0, 183.1, 185.8, 193.8 | 199.9 | 0.025 ~ 0.089 | −0.07 ~ 0.03 |
| walker-run | 410.4, 168.7, 411.5, 210.5, 383.3 | 316.9 | 0.004 ~ 0.355 | −0.07 ~ 2.83 |
| walker-walk-backwards | 259.4, 249.4, 215.6, 249.6, 220.2 | 238.9 | 0.065 ~ 0.113 | −0.06 ~ 0.78 |
| walker-run-backwards | 173.9, 167.9, 178.8, 171.1, 158.5 | 170.1 | 0.002 ~ 0.019 | 0.15 ~ 2.32 |

## 실패하거나 판정 불가인 항목

| 항목 | 상태 | 이유 |
|---|---|---|
| V3 walker-walk | 불일치 | 공식 1 에피소드 990.48이 자체 5 에피소드 최댓값 990.25보다 0.24 크다. V2에서 같은 난수 흐름의 첫 에피소드는 공식과 같은 값(990.483276)이었다 |
| V3 판정 전반 | 거친 판정 | 공식 값이 1 에피소드다 (VERIFICATION_PLAN.md에 적은 한계) |
| V2 deterministic 설정 | 따로 켜지 않음 | 공식 `evaluate.py`가 켜지 않는다. 결과는 50 step 모두 차이 0이었다 |
| 판정 불가 | 없음 | – |

## 실행 기록

| 실행 | 장소 | 시간 |
|---|---|---|
| V4, V5-2 | 로컬 CPU (venv, torch 2.7.1+cpu) | 수 분 |
| GATE 2 커널 (설치, smoke test, 공식 30-task 1회, 자체 trace, Experiment 0 재실행, policy prior) | Kaggle Tesla T4, 2026-10-25 14:01 ~ 15:21 UTC | 약 80분 |

- 커널 코드: [kaggle/gate2_body.py](kaggle/gate2_body.py) (커밋 `b8f58bb`의 코드로 실행).
- 커널 로그: [results/gate2/kaggle_kernel.log](results/gate2/kaggle_kernel.log).
