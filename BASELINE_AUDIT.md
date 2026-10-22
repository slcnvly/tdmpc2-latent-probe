# BASELINE_AUDIT: TD-MPC2 mt30 체크포인트

작성: 2026-10-22. 모든 경로는 다음 기준이다.

- `tdmpc2/...`: [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2) 커밋 `332f19e` (2026-10-20). 코드는 수정하지 않았다.
- `dm_control/...`: dm-control 1.0.16 (`tdmpc2/docker/environment.yaml`의 고정 버전) 설치본.
- `code/...`, `results/...`: 이 저장소.

"확인"은 코드를 읽고 실행해 본 것, "코드 확인"은 읽기만 한 것, "미확인"은 확인하지 못한 것이다.

## 요약

| 항목 | 결과 |
|---|---|
| 1 관측 / padding | state 관측. 0으로 뒤를 채워 24차원. **walker는 원래 24차원이라 padding 없음.** 행동은 6차원, walker는 masking 없음 (확인) |
| 2 task embedding | `nn.Embedding(30, task_dim, max_norm=1)`. task_dim은 1M·317M이 96, 5M·19M·48M이 64. encoder, dynamics, reward, policy, Q 모두에 concat (확인) |
| 3 encoder | MLP(관측 24 + task_dim → … → latent_dim), 마지막에 LayerNorm 후 SimNorm(group 8) (확인) |
| 4 encode 함수 | `WorldModel.encode` (`tdmpc2/tdmpc2/common/world_model.py:105-114`). 직접 재구현과 출력 차이 0.0 (확인) |
| 5 단일 task 실행 | task index를 직접 넘기면 된다. walker id: stand 0, walk 1, run 2, walk-backwards 19, run-backwards 20 (확인) |
| 6 reward | 다섯 개 모두 physics state만의 함수. 단, env 한 step의 reward는 physics 두 substep reward의 합 (확인) |
| 7 상태 저장·복원 | `physics.get_state()` / `set_state()` + `forward()`로 관측·reward 재계산 오차 0.0 (확인) |
| 8 데이터셋 부분 다운로드 | chunk 단위로만 가능. mt30은 chunk 4개(5.8~12.8 GB), task 단위 분리 없음. 받지 않았다 (코드 확인) |
| 9 실행 비용 | Tesla T4, 공식 planning 설정: step당 1M 0.010초 ~ 317M 1.03초, 최대 GPU 메모리 30 MB ~ 4.6 GB (확인, 9절) |
| 10 Newt | 설치하지 않았다. TD-MPC2가 부적합하다는 근거는 지금까지 없다 |

**사전 정보와 다른 점**

- mt30 19M 체크포인트의 `latent_dim`은 768이 아니라 **512**다. `parse_cfg`가 덮어쓴다 (`tdmpc2/tdmpc2/common/parser.py:67-68`). mt80 19M은 768이다.
- mt30의 `task_dim`은 크기에 따라 다르다: 1M·317M은 96, 나머지는 64 (`parser.py:75`).
- `results/` 폴더에는 **mt30 체크포인트의 task별 점수가 없다.** 단일 task online RL 학습 곡선(`results/tdmpc2/<task>.csv`, 열: step, reward, seed)뿐이다. 논문(arXiv 2310.16828)도 mt30은 30개 task 평균의 normalized score만 보고한다 (Figure 7, Figure 22: 1M 18.9, 5M 28.3, 19M 54.2, 48M 59.4, 317M 71.4). 3단계 비교는 이 제약 아래에서 했다 (INITIAL_RESULTS.md).
- walker-walk-backwards와 walker-run-backwards는 dm_control 원본이 아닌 TD-MPC2의 `envs/tasks/walker.xml`을 쓴다. 바닥 plane의 x 반폭이 250 → 500으로 다르다 (6절).

## 1. 관측 종류, 관측 padding, 행동 padding

- 관측 종류: `config.yaml:6` 기본값 `obs: state`. 체크포인트 encoder의 첫 층 입력이 24 + task_dim(5M은 88)이고 conv 층이 없다 → state 관측 (확인, `results/audit_checks.json` `architecture`).
- 관측 padding: `MultitaskWrapper._pad_obs`가 관측 뒤를 0으로 채워 모든 task 중 최대 차원에 맞춘다 (`tdmpc2/tdmpc2/envs/wrappers/multitask.py:44-47`, 최대 차원 `:20`).
- mt30 최대 관측 길이: **24** (walker 3종, walker backwards 2종, fish-swim). task별 차원은 `results/audit_checks.json` `obs_dims_per_task`. HF dataset card의 "MT30 max obs 24"와 일치한다.
- walker 관측 차원은 24로 최대와 같다 → **padding이 0개**다 (`padding_len: 0`).
- 행동 padding: agent는 최대 행동 차원(6)으로 행동을 내고, env는 앞쪽 `action_dim`개만 쓴다 (`multitask.py:56`).
  planning과 policy는 `_action_masks`로 쓰지 않는 차원을 0으로 만든다 (`tdmpc2/tdmpc2/tdmpc2.py:180-181, 195-197`, `common/world_model.py:22-24, 160-163`).
  walker 행동 차원은 6으로 최대와 같아 mask가 전부 1이다.

## 2. Task embedding

- 정의: `nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)` (`common/world_model.py:21`). 30개 task.
- 정규화: `max_norm=1` — 조회 시 L2 norm이 1을 넘는 행을 1로 줄인다 (PyTorch `nn.Embedding` 동작).
  저장된 가중치의 norm은 0.94~1.00 (`audit_checks.json` `task_emb_norm_min_max_all`).
- 차원: mt30에서 1M·317M은 96, 5M·19M·48M은 64 (`common/parser.py:75`). 확인: 1M 96, 5M·19M·48M 64. 317M은 로컬에 없어 코드로만 확인.
- 결합 위치: `task_emb(x, task)`가 입력 뒤에 embedding을 concat한다 (`world_model.py:90-103`).

| 모듈 | 위치 | task embedding |
|---|---|---|
| encoder | `world_model.py:105-114` (`encode`) | 관측에 concat |
| dynamics | `world_model.py:116-123` (`next`) | z에 concat 후 action concat |
| reward | `world_model.py:125-132` | z에 concat 후 action concat |
| termination | `world_model.py:134-143` | concat. 단 `assert task is None` (`:138`)이라 multi-task에서는 호출되지 않는다. `episodic: false`(`config.yaml:7`)라 쓰이지 않음 |
| policy prior | `world_model.py:146-153` | z에 concat |
| Q-function | `world_model.py:188-200` | z에 concat 후 action concat |

- walker 5개 task embedding 사이 cosine similarity (5M): 0.63 ~ 0.83 (`audit_checks.json` `pairwise_cos_walker_embeddings`).

## 3. Encoder 구조

- 생성: `layers.enc` (`common/layers.py:153-164`):
  `mlp(obs_dim + task_dim, max(num_enc_layers-1, 1) * [enc_dim], latent_dim, act=SimNorm)`.
- `mlp` (`layers.py:121-133`): 은닉층은 `NormedLinear` = Linear → (dropout) → LayerNorm → Mish (`layers.py:94-111`).
  출력층도 `NormedLinear`이며 활성화만 SimNorm이다 → Linear → LayerNorm → SimNorm.
- SimNorm (`layers.py:74-91`): 마지막 차원을 `simnorm_dim`개씩 묶어 묶음마다 softmax. 그룹 크기 8 (`config.yaml:64`).
  출력의 각 8차원 묶음 합이 1임을 확인 (`simnorm_group_sums_close_to_1: true`).
- 입력: `[관측(24, padding 포함), task embedding(task_dim)]`. 출력: `latent_dim`차원, 묶음 수 latent_dim/8.

| 크기 | task_dim | 입력 | 은닉층 | latent_dim | SimNorm 묶음 수 | encoder 파라미터 |
|---|---|---|---|---|---|---|
| 1M | 96 | 120 | 1 × 256 | 128 | 16 | 64,640 |
| 5M | 64 | 88 | 1 × 256 | 512 | 64 | 155,904 |
| 19M | 64 | 88 | 2 × 1024 | 512 | 64 | 1,670,656 |
| 48M | 64 | 88 | 3 × 1792 | 768 | 96 | 7,974,912 |
| 317M | 96 | 120 | 4 × 4096 | 1376 | 172 | 코드로만 확인 (`common/__init__.py:19-23`) |

## 4. 공식 encode 함수

- 경로: `WorldModel.encode` (`common/world_model.py:105-114`). planning도 이 함수를 쓴다 (`tdmpc2.py:153`).
- 이 저장소의 latent 추출은 `model.encode(obs, task)`를 그대로 호출한다.
- 재구현 검증: `encoder(cat[obs, task_emb(task)])`를 직접 계산한 값과 `encode` 출력의 최대 절대 차이 **0.0** (네 크기 모두, `audit_checks.json` `manual_reimplementation_max_abs_diff`).
- 같은 관측에 walker task id만 바꿔 넣으면 출력이 달라진다 (walk 대 run 최대 절대 차이 0.17~0.30). 즉 encoder는 task-conditioned다.
- 체크포인트 로드는 `TDMPC2.load`와 같은 절차(`api_model_conversion` 후 `load_state_dict`, `tdmpc2.py:81-95`)를 CPU에서 따로 실행했다 (`code/common_setup.py: load_world_model`).
  `TDMPC2` 클래스가 `cuda:0`을 고정하기 때문이다 (`tdmpc2.py:20`). GPU 실행(9절, Experiment 0)은 `TDMPC2` 클래스를 그대로 쓴다.

## 5. mt30 체크포인트를 단일 walker task에서 실행하는 방법

- 공식 `evaluate.py`는 mt30이면 30개 task를 차례로 평가한다 (`tdmpc2/tdmpc2/evaluate.py:70-97`).
  README도 "single-task evaluation of multi-task checkpoints is currently not supported"라고 적는다 (`README.md:108`).
- 이 저장소의 `code/eval_single_task.py`는 같은 루프를 원하는 task index에만 돌린다:
  `make_env(cfg)`로 30개 env를 모두 만들고 `env.reset(task_idx=k)`, `agent.act(obs, t0=t==0, task=k)` (`evaluate.py:76-86`과 같은 호출).
- task index (`common/__init__.py:29, 34`의 순서): walker-stand 0, walker-walk 1, walker-run 2, walker-walk-backwards 19, walker-run-backwards 20.
- 행동 차원: walker는 6 = 최대 6이라 잘라낼 것이 없다 (1절).
- 평가 기본값: `evaluate.py`는 `eval_mode`를 넘기지 않으므로 False다. 즉 MPPI 평균에 탐색 noise(`std * randn`)를 더한 행동을 쓴다 (`tdmpc2.py:203-205`). 이 저장소도 그대로 따른다.

## 6. Walker 5개 task의 reward

공통: 모든 reward는 dm_control `Physics`의 값으로만 계산된다. action이나 관측 벡터를 입력으로 받지 않는다.

| task | 정의 | 식 |
|---|---|---|
| walker-stand | `dm_control/suite/walker.py:139-147` (`move_speed=0`) | stand = (3·standing + upright) / 4 |
| walker-walk | 같은 함수, `move_speed=1` (`:60-66`, `_WALK_SPEED=1`) | stand · (5·move + 1) / 6, move = tolerance(v, [1, ∞)) |
| walker-run | 같은 함수, `move_speed=8` (`:71-77`, `_RUN_SPEED=8`) | 같은 식, move = tolerance(v, [8, ∞)) |
| walker-walk-backwards | `tdmpc2/tdmpc2/envs/tasks/walker.py:114-128`, `:21-29` | 같은 식, move = tolerance(v, (−∞, −1]) |
| walker-run-backwards | 같은 함수, `:32-40` | 같은 식, move = tolerance(v, (−∞, −8]) |

- standing = tolerance(torso_height, [1.2, ∞), margin 0.6, gaussian). upright = (1 + torso_upright) / 2.
  move = tolerance(horizontal_velocity, bounds, margin = speed/2, value_at_margin 0.5, linear).
- 입력량: `torso_height` = torso z (`walker.py:88-90`), `torso_upright` = torso xmat zz (`:84-86`),
  `horizontal_velocity` = 센서 `torso_subtreelinvel`의 x (`:92-94`, torso 하위 트리 질량중심 속도).
- **env 한 step의 reward는 physics 두 substep reward의 합이다.** `DMControlWrapper.step`이 같은 행동을 2번 적용하고 reward를 더한다 (`envs/dmcontrol.py:55-60`).
  따라서 기록되는 reward는 r(중간 상태) + r(다음 상태)이고, 중간 상태는 행동에 따라 달라진다. 확인: 합과의 최대 오차 1.3e-8 (`audit_checks.json` `walker_walk_reward_composition`).
  "상태 s의 reward"를 정의할 때 이 점을 처리해야 한다 (EXPERIMENT_PLAN.md).
- 초기 상태 분포: 다섯 task 모두 `PlanarWalker.initialize_episode` (`walker.py:118-129`, 관절 무작위화)를 쓴다. backwards는 이를 상속한다.
- 물리 모델: stand·walk·run은 dm_control의 `walker.xml`, backwards 둘은 `tdmpc2/tdmpc2/envs/tasks/walker.xml`.
  차이는 바닥 plane의 크기 한 줄이다 (`walker.xml:22`, `size="250 .8 .2"` 대 `"500 .8 .2"`).
  같은 상태·같은 무작위 행동 400 step을 두 모델에 적용했을 때 상태 차이 0.0 (rootx 시작 0, −10, −60 세 경우).
  MuJoCo plane은 충돌 계산에서 무한 평면으로 취급된다고 알려져 있다. 이 문서는 위 실험 결과만 근거로 쓴다.
- 나머지 모델 값(nq 9, nv 9, body 8, timestep 0.0025, control timestep 0.025)은 다섯 task가 같다.

reward 입력량과 관측 성분의 관계 (`results/reward_obs_mapping.json`, 무작위 행동 2,000 step):

| reward 입력량 | 관측 성분 | 관계 |
|---|---|---|
| torso_height | `height` (1) | 같다 (최대 차이 0.0) |
| torso_upright | `orientations[0]` (torso xmat xx) | 같다 (최대 차이 0.0). 평면 회전이라 xx = zz |
| horizontal_velocity | 직접 대응하는 성분 없음 | `velocity[1]`(rootx 관절 속도)과 상관 0.39, 최대 차이 1.96. qpos·qvel의 비선형 함수 |

관측 성분 이름 (`walker.py:131-137`): `orientations` 14 (torso, right_thigh, right_leg, right_foot, left_thigh, left_leg, left_foot 각각 xmat의 xx, xz), `height` 1, `velocity` 9 (qvel: rootz, rootx, rooty, 관절 6).

## 7. 상태 저장·복원과 재계산

- 저장: `physics.get_state()` (`dm_control/mujoco/engine.py:235`), 복원: `physics.set_state(s)` (`:243`) 후 `physics.forward()`.
- 재계산: 관측 `task.get_observation(physics)`, reward `task.get_reward(physics)`.
- 검증 (`code/audit_checks.py`): 다섯 task에서 무작위 행동 50 step마다 상태를 저장·복원해 재계산한 관측과 reward의 최대 절대 오차 **0.0**.
  저장 상태에서 두 substep을 다시 실행하면 다음 상태가 정확히 같다 (`all_replayed_states_equal: true`).

## 8. 공식 데이터셋 일부 다운로드 가능 여부

- 위치: HF dataset `nicklashansen/tdmpc2`. mt30은 `mt30/chunk_0.pt` ~ `chunk_3.pt` 네 파일, 크기 12.8, 12.8, 12.8, 5.8 GB (HF API 목록).
- 파일 단위로는 일부만 받을 수 있다(`hf download --include`). 가장 작은 단위가 5.8 GB chunk다.
- task 단위 분리 파일은 없다. 한 chunk 안에 어떤 task가 들어 있는지는 받아 보지 않으면 모른다 (미확인).
- `.pt`(pickle)라 일부 바이트만 받아 읽을 수 없다. 이번 작업에서는 아무것도 받지 않았다.

## 9. 실행 비용

Kaggle Tesla T4 1장, `code/eval_single_task.py cost` (`results/gate1/cost.json`).
walker-walk에서 200 step을 돌렸다. planning 설정은 공식 기본값이다 (iterations 6, num_samples 512, horizon 3, `config.yaml:34-42`).
시간은 `agent.act` 한 번(encode + MPPI planning) + CUDA 동기화이며 env step 시간은 넣지 않았다. 처음 10 step은 평균에서 뺐다.

| 크기 | compile | step당 평균 (초) | 첫 step (초) | 최대 GPU 메모리 (MB) | latent_dim | task_dim |
|---|---|---|---|---|---|---|
| 1M | true | 0.0095 | 57.4 | 30 | 128 | 96 |
| 1M | false | 0.0490 | 0.16 | 32 | 128 | 96 |
| 5M | true | 0.0196 | 48.2 | 90 | 512 | 64 |
| 5M | false | 0.0500 | 0.05 | 100 | 512 | 64 |
| 19M | true | 0.0528 | 47.9 | 272 | 512 | 64 |
| 19M | false | 0.0646 | 0.08 | 296 | 512 | 64 |
| 48M | true | 0.1671 | 49.7 | 765 | 768 | 64 |
| 48M | false | 0.1633 | 0.18 | 843 | 768 | 64 |
| 317M | true | 1.0302 | 0.98 | 4,623 | 1376 | 96 |
| 317M | false | 1.0258 | 0.96 | 6,548 | 1376 | 96 |

- `compile=true`의 첫 step은 `torch.compile`과 CUDA graph 준비 시간이다 (약 50초). 317M은 첫 step이 길지 않았는데, compile이 실제로 적용됐는지는 확인하지 않았다.
- 체크포인트 로드와 30개 env 생성: 4.5~12.8초.
- 30개 env를 모두 만드는 이유: 공식 `make_env`가 mt30이면 30개를 만든다 (`envs/__init__.py:37-55`).
- 메모리는 README의 "8 GB 이상 권장"보다 작게 나왔다 (317M도 4.6~6.5 GB). T4(15 GB)에서 다섯 크기 모두 돈다.

## 10. Newt

설치하지 않았다. 지금까지 확인한 것 중 TD-MPC2 mt30이 이 진단에 부적합하다는 근거는 없다.
