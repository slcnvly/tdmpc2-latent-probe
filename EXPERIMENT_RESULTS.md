# EXPERIMENT_RESULTS (Experiment 1) — GATE 3 시점, 진행 중

작성: 2026-10-26. **Step 6(probe 학습) 이후는 실행하지 않았다.** 이 문서는 지금까지 실행한 Step P0, 4, 5만 담는다.
GATE 3에서 멈췄고, 사용자가 EXPERIMENT_PLAN.md의 가설 칸을 쓰고 승인한 뒤에 Step 6을 실행한다.

## 해석 제한 (실행 전에 받은 지시를 그대로 옮김)

- Cross-task probe 성능 저하는 해당 task 수행 실패를 의미하지 않는다.
- Probe 성능은 information accessibility를 측정하며, 정보의 존재·부재를 증명하지 않는다.
- Reward 예측 능력과 decision sufficiency는 다르다.
- 이 실험은 task-independent shared encoder bottleneck을 직접 검증하지 않는다.
- 같은 열 안(같은 target, 같은 test 상태)의 칸끼리는 직접 비교할 수 있다.
  같은 행 안의 칸끼리는 target 분포가 달라 R² 비교가 target 분산 차이를 반영할 수 있다. 행 방향 비교는 주 비교로 쓰지 않는다.

## 실행한 것

| 단계 | 결과 | 문서·파일 |
|---|---|---|
| P0 | 사전 고정, 수집 전 커밋 `6b9c0f1` | [EXPERIMENT1_PREREG.md](EXPERIMENT1_PREREG.md) |
| Step 4 | 300 에피소드, 150,000 상태 전부 수집, 무결성 검사 전부 통과 | [DATA_SUMMARY.md](DATA_SUMMARY.md) |
| Step 5 | pooled 상태, latent(학습 모델·random-init), reward target, 검사 3개 통과 | 아래 |

- checkpoint: `mt30-48M.pt`, sha256 `bacb9ae1891a9b03dc3942a287080beb573220ea9d0263c5efd7a94a5c2ec419`.
- task ID: walker-stand 0, walker-walk 1, walker-run 2, walker-walk-backwards 19, walker-run-backwards 20.
- seed 규칙: episode_seed = 10000 + 1000 × task 번호 + 100 × 종류(A 0, B 1, C 2) + 에피소드 번호 (PREREG 1).

## Step 5. Pooled state, latent, reward target

### 만든 것 (`data/pooled/`, 코드 `code/step5_latents_targets.py`)

| 파일 | 모양 | 내용 |
|---|---|---|
| `index.npz` | 150,000 | 상태마다 source task, rollout 종류, 에피소드 번호, t |
| `obs.npy` | (150,000, 24) | o_t (encoder 입력 관측) |
| `state.npy` | (150,000, 19) | s_t |
| `targets.npy` | (150,000, 5) | r_τ′(s_t), τ′ 순서 stand, walk, run, walk-backwards, run-backwards |
| `latent_learned.npy` | (5, 150,000, 768) float32, 2.3 GB | 학습된 encoder의 z(s, τ). τ는 위 다섯 task 순서 |
| `latent_random.npy` | (5, 150,000, 768) float32, 2.3 GB | random-init 모델의 z(s, τ) |

- latent는 공식 `WorldModel.encode`(`common/world_model.py:105-114`)로 계산했다.
  계산은 로컬 CPU(float32)에서 했고, rollout의 planner는 GPU에서 돌았다.
- 같은 상태 150,000개에 task embedding만 바꿔 다섯 번 encode했다. 상태를 새로 뽑지 않았다.
- random-init 모델: 같은 config로 `torch.manual_seed(20261028)` 뒤 `WorldModel(cfg)`를 만들고 가중치를 불러오지 않았다.
  task embedding도 이 모델 자체의 무작위 표를 썼다 (task 번호 0, 1, 2, 19, 20의 행).
  - 이 무작위 embedding의 L2 norm은 0.085~0.108이다. 학습된 embedding의 norm은 0.992~0.998이다 (`results/exp1/latent_meta.json`).
- latent 확인 (151개 상태 표본):
  - SimNorm 8차원 묶음 합과 1의 최대 차이: 학습 모델·random-init 모두 2.4e-7.
  - 같은 상태에서 walk embedding과 run embedding의 z 차이(최대 절대값)의 평균: 학습 모델 0.368, random-init 0.0075.

### reward target 검증 (허용 오차는 실행 전에 코드에 고정, 커밋 `ba4f1a3`)

| 검사 | 기준 | 결과 | 판정 |
|---|---|---|---|
| 1 시점 일치 | encoder 입력은 저장한 o_t, reward는 저장한 s_t 복원 후 계산. 같은 인덱스. s_t 복원 관측 vs o_t 최대 차이 ≤ 1e-5 | 0.0 (150,000 상태) | 통과 |
| 2 기록값 대조 | t ≥ 1에서 r_source(s_t) = 직전 step 두 번째 substep reward, 최대 차이 ≤ 1e-6 | 3.3e-16 (149,700 상태, t = 0 300개 제외) | 통과 |
| 3 부작용 없음 | 다섯 reward 계산 뒤 각 env의 `get_state()`가 복원한 s_t와 정확히 같음 | 150,000 상태 × 5 env 모두 같음 | 통과 |

- 검사 1의 코드 근거: `step5_latents_targets.py`에서 encoder 입력은 `obs[n]`, reward는 `set_state(state[n])` → `forward()` → `get_reward`이고, 둘 다 같은 n(같은 에피소드의 같은 t)이다.
- target은 single-substep reward r_τ′(s_t)다. env step 합계 r_logged_t와 다르다.

### 검사 4. 열별 target 분포 (r_τ′(s_t), 단위 없음, 0~1)

| target task τ′ | 집합 | 상태 수 | 평균 | 분산 | 5% | 50% | 95% |
|---|---|---|---|---|---|---|---|
| walker-stand | pooled | 150,000 | 0.7046 | 0.1670 | 0.0715 | 0.9954 | 1.0000 |
| walker-stand | A | 50,000 | 0.9902 | 0.0039 | 0.9817 | 0.9984 | 1.0000 |
| walker-stand | B | 50,000 | 0.9851 | 0.0050 | 0.9523 | 0.9979 | 1.0000 |
| walker-stand | C | 50,000 | 0.1385 | 0.0115 | 0.0526 | 0.1211 | 0.2607 |
| walker-walk | pooled | 150,000 | 0.4228 | 0.1785 | 0.0153 | 0.1665 | 0.9999 |
| walker-walk | A | 50,000 | 0.6028 | 0.1607 | 0.1658 | 0.8240 | 1.0000 |
| walker-walk | B | 50,000 | 0.6245 | 0.1537 | 0.1655 | 0.9213 | 0.9999 |
| walker-walk | C | 50,000 | 0.0409 | 0.0021 | 0.0102 | 0.0262 | 0.1207 |
| walker-run | pooled | 150,000 | 0.2086 | 0.0357 | 0.0130 | 0.1665 | 0.6408 |
| walker-run | A | 50,000 | 0.3068 | 0.0337 | 0.1657 | 0.2484 | 0.7087 |
| walker-run | B | 50,000 | 0.2937 | 0.0225 | 0.1654 | 0.2650 | 0.6130 |
| walker-run | C | 50,000 | 0.0254 | 0.0004 | 0.0094 | 0.0214 | 0.0520 |
| walker-walk-backwards | pooled | 150,000 | 0.2565 | 0.0986 | 0.0148 | 0.1662 | 0.9984 |
| walker-walk-backwards | A | 50,000 | 0.3747 | 0.1167 | 0.1641 | 0.1666 | 0.9990 |
| walker-walk-backwards | B | 50,000 | 0.3529 | 0.1078 | 0.1593 | 0.1666 | 0.9992 |
| walker-walk-backwards | C | 50,000 | 0.0419 | 0.0019 | 0.0100 | 0.0261 | 0.1260 |
| walker-run-backwards | pooled | 150,000 | 0.1365 | 0.0079 | 0.0129 | 0.1662 | 0.2801 |
| walker-run-backwards | A | 50,000 | 0.1940 | 0.0026 | 0.1640 | 0.1666 | 0.2838 |
| walker-run-backwards | B | 50,000 | 0.1899 | 0.0023 | 0.1591 | 0.1666 | 0.2850 |
| walker-run-backwards | C | 50,000 | 0.0255 | 0.0004 | 0.0093 | 0.0211 | 0.0533 |

## 실제 비용과 저장량 (GATE 3 시점)

| 항목 | 값 |
|---|---|
| Kaggle GPU (Step 4) | 커널 약 2.2시간 (T4 × 2). 주간 사용량 3.55 / 30시간 (2026-10-31 초기화) |
| 로컬 CPU (Step 5) | reward target, 검사, latent 두 벌: 3분 40초 |
| rollout 데이터 | 110 MB (압축 npz 300개 + 메타데이터 JSON). 작업 컴퓨터의 `data/rollouts/`와 Kaggle 커널 출력에 있다. GitHub에는 올리지 않았다 (공유 위치 미정) |
| pooled 데이터 (latent 제외) | 약 47 MB, 작업 컴퓨터의 `data/pooled/` |
| latent | 4.6 GB (`data/pooled/latent_*.npy`), 작업 컴퓨터에만 있다. `code/step5_latents_targets.py`로 다시 만들 수 있다 (CPU 약 4분) |

## 계획 대비 변경

| 계획 | 실제 | 이유 |
|---|---|---|
| latent 추출 장소 (명시 없음) | 로컬 CPU | GPU가 필요 없고, 데이터가 로컬에 있다. GPU 계산과 비트 단위로 같지는 않을 수 있다 |
| C rollout (명시 없음) | 같은 Kaggle 세션의 CPU 프로세스 | A·B와 같은 패키지 버전에서 수집하려고 |

## 이 결과로 결론 낼 수 없는 것 (GATE 3 시점)

- probe를 학습하지 않았으므로 information accessibility에 관한 어떤 결론도 아직 없다.
- target 분포 표는 열마다 분산이 크게 다르다. 예: pooled 분산 walker-walk 0.1785, walker-run-backwards 0.0079.
  행 방향 R² 비교가 이 차이를 반영할 수 있다는 해석 제한이 이 데이터에 적용된다.
