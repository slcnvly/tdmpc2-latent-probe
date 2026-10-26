# EXPERIMENT_RESULTS (Experiment 1)

> **가설 미작성 상태에서 Step 6를 실행함.** 사용자 지시의 가설 블록이 비어 있어, EXPERIMENT_PLAN.md의 가설 칸은 "(사용자 작성)"으로 두었다.

작성: 2026-10-26. 실행한 것: Step P0, 4, 5, 6(probe), heatmap·표, 비공개 Kaggle dataset 업로드, 선택 실험(counterfactual task swap).
아래 숫자는 모두 결과 파일에서 읽은 값이다. 해석, 원인 추측, 연구 기여 주장은 쓰지 않는다.

## 해석 제한 (실행 전에 받은 지시를 그대로 옮김)

- Cross-task probe 성능 저하는 해당 task 수행 실패를 의미하지 않는다.
- Probe 성능은 information accessibility를 측정하며, 정보의 존재·부재를 증명하지 않는다.
- Reward 예측 능력과 decision sufficiency는 다르다.
- 이 실험은 task-independent shared encoder bottleneck을 직접 검증하지 않는다.
- 같은 열 안(같은 target, 같은 test 상태)의 칸끼리는 직접 비교할 수 있다.
  같은 행 안의 칸끼리는 target 분포가 달라 R² 비교가 target 분산 차이를 반영할 수 있다. 행 방향 비교는 주 비교로 쓰지 않는다.

## 1. 실행 개요

| 항목 | 값 |
|---|---|
| checkpoint | `mt30-48M.pt`, sha256 `bacb9ae1891a9b03dc3942a287080beb573220ea9d0263c5efd7a94a5c2ec419` (수집과 probe 실행 시작 시 각각 다시 계산해 같음) |
| task ID | walker-stand 0, walker-walk 1, walker-run 2, walker-walk-backwards 19, walker-run-backwards 20 |
| 데이터 | 300 에피소드(5 task × A·B·C × 20), 150,000 상태. [DATA_SUMMARY.md](DATA_SUMMARY.md) |
| seed 규칙 | episode_seed = 10000 + 1000 × task 번호 + 100 × 종류(A 0, B 1, C 2) + 에피소드 번호 ([EXPERIMENT1_PREREG.md](EXPERIMENT1_PREREG.md) 1) |
| probe 설정 | PREREG 5 그대로. ridge λ ∈ 10^[−4:0.5:4], validation으로 선택. MLP 256-256 ReLU, Adam 1e-3, batch 512, 최대 200 epoch, patience 20, seed 0~4 |
| 통계 | PREREG 6 그대로. test 에피소드 bootstrap 1,000회(seed 20261027). 주 기준은 같은 열 안 paired bootstrap difference |

### 실행 위치와 소요 시간 (6-1)

비용 측정 (`results/exp1/probe_cost.json`, 로컬 CPU 12 thread, 칸 하나: 학습된 z 행 walker-walk → 열 walker-walk):

| 측정 | 값 |
|---|---|
| linear | target 하나 2.7초 (λ 17개 포함) |
| MLP | seed 하나 192초 (159 epoch) |
| 전체 예상 (로컬 CPU, 순차) | linear 80개(사전 등록 55 + 보조 25) 수 분. MLP 400개(사전 등록 275 + 보조 125) × 192초 ≈ **21시간** |

- 결정: **probe 전체를 Kaggle(Tesla T4 2장)에서 실행**했다. 로컬 CPU로는 MLP에 약 21시간이 걸린다는 추정이 이유다.
- 시험 실행 결과는 비용 측정용으로만 썼다. 아래 결과는 전체 실행에서 다시 계산한 값이다.

실제 소요 (Kaggle 커널 `tdmpc2-probe-exp1-probe`, 2026-10-26 08:51 ~ 10:33 UTC):

| 단계 | 시간 |
|---|---|
| 데이터·latent·분할 재생성 | 394.4초 |
| linear 80개 (CPU) | 223.2초 |
| MLP 400개 (GPU 프로세스 4개, T4 한 장에 2개씩) | 5,160.1초 (86분). probe당 평균 49.5초 |
| 커널 전체 | 약 1.7시간. 주간 GPU 사용량이 3.55 → 약 5.3시간 |

### 데이터 sha256과 실행 위치에 따른 차이

- Kaggle은 비공개 dataset의 rollout 600개 파일의 sha256을 업로드 전 manifest와 대조했고, 모두 같았다 (`ROLLOUTS_SHA_OK`).
- 같은 스크립트로 pooled 데이터, latent, 분할을 다시 만들었다. 결과 sha256 대조 (`results/exp1/kaggle_vs_local_data.json`, `results/exp1/local_data_sha256.txt`):

| 파일 | 로컬과 sha256 | 차이 |
|---|---|---|
| index, obs, state, torso_height, split.json, split_labels | 같음 | – |
| targets.npy | **다름** | 최대 절대 차이 3.3e-16 (750,000개 중 28,666개가 다름) |
| latent_learned.npy | **다름** | 최대 2.5e-6, 평균 3.7e-8 (상태 3,000개 표본) |
| latent_random.npy | **다름** | 최대 1.0e-6, 평균 5.7e-8 |
| latent_normmatched.npy | **다름** | 최대 1.0e-6, 평균 5.7e-8 |

- 두 실행 환경의 CPU 모델이 다르다 (Kaggle Intel Xeon @ 2.00GHz, 로컬 Intel Core i5-10400). 같은 패키지 버전이다. 차이의 원인은 검증하지 않았다.
- Kaggle에서 다시 만든 데이터로도 Step 5 검사 3개를 다시 통과했다 (`results/exp1/kaggle/step5_checks_kaggle.json`; 검사 2 최대 차이 0.0).
- **probe는 Kaggle에서 만든 latent와 targets로 학습·예측했고, 평가도 그 targets로 했다.** 평가에 쓴 targets와 분할 파일은 Kaggle 출력(`data_kaggle/`)이다.
- 실행 위치 결정 뒤, 첫 probe 커널을 시작 12분 만에 내리고 다시 올렸다. 첫 커널은 Kaggle에서 만든 targets를 출력에 남기지 않아, 평가가 학습과 다른 targets로 계산될 수 있었다. 두 번째 커널은 같은 코드에 출력 저장만 더했다.

## 2. 분할 (6-0, `data/split.json`, `results/exp1/split_check.json`)

| split | 에피소드 | 상태 | 종류별 상태 (A / B / C) | fallen 비율 (torso_height < 0.6 m) |
|---|---|---|---|---|
| train | 180 | 90,000 | 30,000 / 30,000 / 30,000 | 0.3195 |
| validation | 60 | 30,000 | 10,000 / 10,000 / 10,000 | 0.3195 |
| test | 60 | 30,000 | 10,000 / 10,000 / 10,000 | 0.3202 |

- **층**: A·B는 (source task, 종류) 10개 층마다 train 12 / validation 4 / test 4 에피소드다. C는 100 에피소드를 60 / 20 / 20으로 나눴다.
- **C의 source task 꼬리표별 분포**: C는 꼬리표로 층화하지 않았으므로 꼬리표별 test 에피소드 수가 2~7개로 고르지 않다 (`split_check.json` `C_episodes_per_label`).
- **두 split에 동시에 있는 에피소드**: 0개. 300 에피소드가 모두 배정됐다.
- **같은 상태의 latent 5개**: 상태 인덱스 n 하나에 split 꼬리표 하나가 붙으므로, 같은 상태의 latent 5개는 같은 split에 들어간다 (latent 배열이 (5, N, 768)).
- **분할 난수**: seed 20261026. Kaggle에서 다시 만든 분할 파일도 sha256이 같다.

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


## 3. Step 6 결과

모든 칸은 test R²다 (1 − 잔차 제곱합 / 평균 기준 제곱합, 원래 target 단위). 괄호는 test 에피소드 bootstrap 1,000회의 95% 백분위 구간이다. MLP는 seed 5개 R²의 평균이고, bootstrap도 표본마다 seed 5개 평균이다.

필수 그림:
- [heatmap_linear_trained_pooled.png](results/exp1/eval/heatmap_linear_trained_pooled.png)
- [heatmap_mlp_trained_pooled.png](results/exp1/eval/heatmap_mlp_trained_pooled.png)

추가 그림:
- fallen 제외: [linear](results/exp1/eval/heatmap_linear_trained_not_fallen.png), [MLP](results/exp1/eval/heatmap_mlp_trained_not_fallen.png)
- random-init: [linear](results/exp1/eval/heatmap_linear_random_init_pooled.png), [MLP](results/exp1/eval/heatmap_mlp_random_init_pooled.png). fallen 제외도 같은 폴더에 있다.
- norm-matched random-init(보조): [linear](results/exp1/eval/heatmap_linear_normmatched_random_init_pooled.png), [MLP](results/exp1/eval/heatmap_mlp_normmatched_random_init_pooled.png)

그림의 색은 R² −1~1 고정 범위이고, 칸 안 숫자는 실제 값이다(소수 셋째 자리).

모든 칸의 CSV는 [cells.csv](results/exp1/eval/cells.csv)다. MLP seed별 값과 seed 간 표준편차를 포함한다.
모든 paired difference는 [paired_differences.csv](results/exp1/eval/paired_differences.csv)다.

### pooled test (주 결과)

test 상태 30,000개, 에피소드 60개. 열별 target 평균: stand 0.7049, walk 0.4236, run 0.2089, walk-bwd 0.2520, run-bwd 0.1360. 분산: stand 0.1673, walk 0.1781, run 0.0362, walk-bwd 0.0961, run-bwd 0.0079.


**Linear (ridge), 학습된 encoder z(s, τ)** (칸: test R² [95% bootstrap 구간])

| encoder 입력 τ \ target τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | 0.999 [0.999, 0.999] | 0.973 [0.967, 0.977] | 0.998 [0.996, 0.998] | 0.973 [0.954, 0.981] | 0.993 [0.989, 0.995] |
| walker-walk | 0.999 [0.999, 0.999] | 0.976 [0.971, 0.980] | 0.998 [0.997, 0.998] | 0.970 [0.948, 0.979] | 0.993 [0.990, 0.994] |
| walker-run | 0.999 [0.999, 0.999] | 0.971 [0.964, 0.975] | 0.998 [0.997, 0.999] | 0.961 [0.934, 0.973] | 0.991 [0.988, 0.993] |
| walker-walk-backwards | 0.999 [0.999, 0.999] | 0.977 [0.972, 0.981] | 0.997 [0.995, 0.998] | 0.972 [0.953, 0.980] | 0.994 [0.991, 0.996] |
| walker-run-backwards | 0.999 [0.999, 0.999] | 0.966 [0.958, 0.972] | 0.997 [0.995, 0.998] | 0.954 [0.918, 0.972] | 0.993 [0.991, 0.994] |

**Linear (ridge), random-init z(s, τ) (사전 등록 대조군)** (칸: test R² [95% bootstrap 구간])

| encoder 입력 τ \ target τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | 0.753 [0.688, 0.773] | 0.757 [0.714, 0.781] | 0.777 [0.698, 0.804] | 0.651 [0.503, 0.699] | 0.680 [0.596, 0.721] |
| walker-walk | 0.754 [0.688, 0.774] | 0.757 [0.713, 0.781] | 0.776 [0.696, 0.803] | 0.651 [0.503, 0.700] | 0.681 [0.597, 0.722] |
| walker-run | 0.755 [0.690, 0.775] | 0.759 [0.715, 0.782] | 0.779 [0.700, 0.805] | 0.652 [0.504, 0.700] | 0.682 [0.598, 0.722] |
| walker-walk-backwards | 0.754 [0.690, 0.775] | 0.758 [0.714, 0.781] | 0.778 [0.699, 0.804] | 0.652 [0.504, 0.700] | 0.682 [0.599, 0.723] |
| walker-run-backwards | 0.754 [0.689, 0.774] | 0.757 [0.714, 0.781] | 0.777 [0.698, 0.804] | 0.653 [0.505, 0.701] | 0.682 [0.599, 0.722] |

**Linear (ridge), norm-matched random-init z(s, τ) (보조 대조군)** (칸: test R² [95% bootstrap 구간])

| encoder 입력 τ \ target τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | 0.786 [0.738, 0.800] | 0.773 [0.732, 0.796] | 0.800 [0.728, 0.824] | 0.671 [0.522, 0.719] | 0.717 [0.646, 0.752] |
| walker-walk | 0.791 [0.744, 0.803] | 0.772 [0.729, 0.795] | 0.797 [0.725, 0.821] | 0.672 [0.527, 0.720] | 0.721 [0.651, 0.756] |
| walker-run | 0.794 [0.746, 0.808] | 0.780 [0.739, 0.802] | 0.810 [0.742, 0.833] | 0.672 [0.528, 0.717] | 0.722 [0.651, 0.756] |
| walker-walk-backwards | 0.796 [0.751, 0.809] | 0.776 [0.733, 0.799] | 0.805 [0.735, 0.828] | 0.676 [0.529, 0.722] | 0.728 [0.659, 0.761] |
| walker-run-backwards | 0.793 [0.746, 0.805] | 0.774 [0.733, 0.796] | 0.803 [0.732, 0.825] | 0.675 [0.528, 0.722] | 0.724 [0.655, 0.759] |

**MLP (seed 5개 평균), 학습된 encoder z(s, τ)** (칸: test R² [95% bootstrap 구간])

| encoder 입력 τ \ target τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] |
| walker-walk | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| walker-run | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] |
| walker-walk-backwards | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] |
| walker-run-backwards | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] |

**MLP (seed 5개 평균), random-init z(s, τ) (사전 등록 대조군)** (칸: test R² [95% bootstrap 구간])

| encoder 입력 τ \ target τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | 0.993 [0.991, 0.994] | 0.993 [0.991, 0.994] | 0.995 [0.994, 0.996] | 0.991 [0.986, 0.993] | 0.990 [0.986, 0.993] |
| walker-walk | 0.993 [0.991, 0.995] | 0.993 [0.991, 0.994] | 0.995 [0.994, 0.996] | 0.991 [0.987, 0.994] | 0.990 [0.985, 0.993] |
| walker-run | 0.993 [0.991, 0.994] | 0.993 [0.991, 0.994] | 0.995 [0.994, 0.996] | 0.990 [0.984, 0.992] | 0.989 [0.984, 0.992] |
| walker-walk-backwards | 0.994 [0.992, 0.995] | 0.993 [0.992, 0.995] | 0.995 [0.994, 0.996] | 0.991 [0.987, 0.994] | 0.990 [0.986, 0.993] |
| walker-run-backwards | 0.994 [0.992, 0.995] | 0.993 [0.992, 0.995] | 0.995 [0.994, 0.996] | 0.991 [0.986, 0.993] | 0.990 [0.985, 0.993] |

**MLP (seed 5개 평균), norm-matched random-init z(s, τ) (보조 대조군)** (칸: test R² [95% bootstrap 구간])

| encoder 입력 τ \ target τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | 0.994 [0.993, 0.995] | 0.993 [0.992, 0.995] | 0.996 [0.994, 0.996] | 0.991 [0.987, 0.994] | 0.991 [0.987, 0.993] |
| walker-walk | 0.995 [0.993, 0.996] | 0.994 [0.992, 0.995] | 0.996 [0.994, 0.996] | 0.992 [0.987, 0.994] | 0.992 [0.988, 0.994] |
| walker-run | 0.994 [0.992, 0.995] | 0.994 [0.992, 0.995] | 0.996 [0.994, 0.997] | 0.991 [0.987, 0.994] | 0.991 [0.987, 0.994] |
| walker-walk-backwards | 0.994 [0.993, 0.995] | 0.993 [0.992, 0.995] | 0.995 [0.994, 0.996] | 0.992 [0.987, 0.994] | 0.991 [0.987, 0.994] |
| walker-run-backwards | 0.994 [0.993, 0.995] | 0.994 [0.992, 0.995] | 0.996 [0.994, 0.996] | 0.992 [0.988, 0.994] | 0.990 [0.986, 0.993] |

**raw observation o(s)** (열마다 1개, 행 없음)

| probe | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| linear | 0.986 [0.985, 0.987] | 0.791 [0.741, 0.823] | 0.932 [0.895, 0.946] | 0.619 [0.419, 0.688] | 0.880 [0.828, 0.909] |
| mlp | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] | 0.999 [0.999, 0.999] | 1.000 [1.000, 1.000] |

### fallen 상태 제외 pooled test (보조 1)

test 상태 20,395개, 에피소드 60개. 열별 target 평균: stand 0.9806, walk 0.6064, run 0.2970, walk-bwd 0.3532, run-bwd 0.1897. 분산: stand 0.0077, walk 0.1573, run 0.0290, walk-bwd 0.1089, run-bwd 0.0026.


**Linear (ridge), 학습된 encoder z(s, τ)** (칸: test R² [95% bootstrap 구간])

| encoder 입력 τ \ target τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | 0.981 [0.969, 0.987] | 0.959 [0.950, 0.964] | 0.997 [0.994, 0.998] | 0.971 [0.952, 0.979] | 0.975 [0.962, 0.983] |
| walker-walk | 0.981 [0.969, 0.987] | 0.964 [0.957, 0.969] | 0.997 [0.995, 0.998] | 0.967 [0.944, 0.977] | 0.973 [0.960, 0.981] |
| walker-run | 0.981 [0.969, 0.987] | 0.955 [0.946, 0.961] | 0.997 [0.995, 0.998] | 0.956 [0.925, 0.970] | 0.969 [0.951, 0.977] |
| walker-walk-backwards | 0.981 [0.969, 0.987] | 0.966 [0.959, 0.971] | 0.996 [0.993, 0.997] | 0.969 [0.949, 0.978] | 0.980 [0.970, 0.986] |
| walker-run-backwards | 0.981 [0.969, 0.987] | 0.948 [0.936, 0.955] | 0.996 [0.992, 0.997] | 0.947 [0.902, 0.968] | 0.976 [0.960, 0.982] |

**Linear (ridge), random-init z(s, τ) (사전 등록 대조군)** (칸: test R² [95% bootstrap 구간])

| encoder 입력 τ \ target τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | -3.623 [-7.257, -1.991] | 0.707 [0.632, 0.749] | 0.730 [0.642, 0.757] | 0.637 [0.535, 0.682] | 0.016 [-0.224, 0.144] |
| walker-walk | -3.612 [-7.237, -1.987] | 0.706 [0.632, 0.749] | 0.728 [0.639, 0.756] | 0.636 [0.534, 0.682] | 0.017 [-0.225, 0.145] |
| walker-run | -3.582 [-7.199, -1.965] | 0.708 [0.634, 0.750] | 0.732 [0.646, 0.760] | 0.637 [0.536, 0.682] | 0.021 [-0.216, 0.148] |
| walker-walk-backwards | -3.582 [-7.191, -1.966] | 0.707 [0.633, 0.749] | 0.730 [0.643, 0.758] | 0.638 [0.536, 0.683] | 0.023 [-0.215, 0.150] |
| walker-run-backwards | -3.590 [-7.206, -1.967] | 0.707 [0.633, 0.749] | 0.730 [0.643, 0.758] | 0.639 [0.538, 0.684] | 0.024 [-0.212, 0.152] |

**Linear (ridge), norm-matched random-init z(s, τ) (보조 대조군)** (칸: test R² [95% bootstrap 구간])

| encoder 입력 τ \ target τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | -2.685 [-5.484, -1.385] | 0.728 [0.660, 0.768] | 0.766 [0.692, 0.790] | 0.661 [0.561, 0.703] | 0.166 [-0.041, 0.282] |
| walker-walk | -2.612 [-5.415, -1.338] | 0.727 [0.660, 0.767] | 0.761 [0.684, 0.785] | 0.661 [0.558, 0.703] | 0.173 [-0.030, 0.290] |
| walker-run | -2.566 [-5.371, -1.298] | 0.734 [0.668, 0.772] | 0.779 [0.706, 0.800] | 0.661 [0.561, 0.703] | 0.175 [-0.014, 0.291] |
| walker-walk-backwards | -2.503 [-5.237, -1.267] | 0.731 [0.666, 0.770] | 0.771 [0.699, 0.793] | 0.666 [0.567, 0.707] | 0.199 [0.002, 0.308] |
| walker-run-backwards | -2.552 [-5.335, -1.288] | 0.729 [0.662, 0.768] | 0.768 [0.697, 0.790] | 0.667 [0.566, 0.708] | 0.192 [-0.002, 0.309] |

**MLP (seed 5개 평균), 학습된 encoder z(s, τ)** (칸: test R² [95% bootstrap 구간])

| encoder 입력 τ \ target τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | 0.999 [0.998, 0.999] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | 0.999 [0.998, 1.000] |
| walker-walk | 0.999 [0.998, 0.999] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | 0.999 [0.999, 1.000] |
| walker-run | 0.999 [0.998, 0.999] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | 0.999 [0.999, 1.000] |
| walker-walk-backwards | 0.999 [0.998, 0.999] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.999 [0.998, 1.000] |
| walker-run-backwards | 0.999 [0.998, 0.999] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | 0.999 [0.999, 1.000] |

**MLP (seed 5개 평균), random-init z(s, τ) (사전 등록 대조군)** (칸: test R² [95% bootstrap 구간])

| encoder 입력 τ \ target τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | 0.828 [0.732, 0.876] | 0.990 [0.987, 0.992] | 0.992 [0.989, 0.994] | 0.990 [0.985, 0.993] | 0.964 [0.948, 0.976] |
| walker-walk | 0.835 [0.737, 0.883] | 0.990 [0.986, 0.992] | 0.993 [0.989, 0.994] | 0.990 [0.986, 0.993] | 0.963 [0.946, 0.975] |
| walker-run | 0.833 [0.741, 0.880] | 0.990 [0.986, 0.992] | 0.993 [0.989, 0.994] | 0.989 [0.984, 0.992] | 0.962 [0.943, 0.974] |
| walker-walk-backwards | 0.842 [0.752, 0.887] | 0.990 [0.987, 0.992] | 0.993 [0.989, 0.994] | 0.991 [0.987, 0.993] | 0.964 [0.948, 0.975] |
| walker-run-backwards | 0.841 [0.750, 0.887] | 0.990 [0.987, 0.992] | 0.993 [0.989, 0.994] | 0.990 [0.986, 0.993] | 0.964 [0.947, 0.976] |

**MLP (seed 5개 평균), norm-matched random-init z(s, τ) (보조 대조군)** (칸: test R² [95% bootstrap 구간])

| encoder 입력 τ \ target τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | 0.856 [0.776, 0.896] | 0.991 [0.988, 0.993] | 0.993 [0.990, 0.994] | 0.990 [0.986, 0.993] | 0.966 [0.951, 0.976] |
| walker-walk | 0.866 [0.794, 0.903] | 0.991 [0.989, 0.993] | 0.993 [0.990, 0.994] | 0.991 [0.986, 0.993] | 0.970 [0.957, 0.979] |
| walker-run | 0.836 [0.748, 0.883] | 0.991 [0.988, 0.993] | 0.993 [0.990, 0.995] | 0.991 [0.986, 0.993] | 0.966 [0.951, 0.977] |
| walker-walk-backwards | 0.861 [0.783, 0.901] | 0.990 [0.987, 0.992] | 0.993 [0.989, 0.994] | 0.991 [0.987, 0.994] | 0.968 [0.953, 0.978] |
| walker-run-backwards | 0.858 [0.778, 0.898] | 0.991 [0.988, 0.993] | 0.993 [0.990, 0.995] | 0.991 [0.987, 0.994] | 0.964 [0.948, 0.975] |

**raw observation o(s)** (열마다 1개, 행 없음)

| probe | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| linear | 0.816 [0.708, 0.866] | 0.698 [0.622, 0.748] | 0.929 [0.885, 0.942] | 0.555 [0.344, 0.615] | 0.530 [0.385, 0.616] |
| mlp | 1.000 [0.999, 1.000] | 0.999 [0.999, 1.000] | 1.000 [1.000, 1.000] | 0.999 [0.999, 0.999] | 0.999 [0.998, 0.999] |

### in-distribution test (보조 2)

**이 표 안에서는 행끼리 비교하지 않는다. 행마다 test 상태 집합이 다르다.**


**주 정의: source task = τ인 A 상태, linear, 학습된 z(s, τ)** (행 τ의 test 상태 수·에피소드 수는 오른쪽 열)

| 행 τ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards | 상태 / 에피소드 |
|---|---|---|---|---|---|---|
| walker-stand | 0.986 [0.309, 0.993] | 0.802 [0.593, 0.885] | 0.925 [0.852, 0.951] | 0.580 [0.503, 0.624] | 0.807 [0.664, 0.893] | 2,000 / 4 |
| walker-walk | 0.986 [0.940, 0.994] | 0.731 [0.551, 0.813] | 0.972 [0.951, 0.980] | -5.139 [-8.576, -2.906] | 0.815 [0.274, 0.916] | 2,000 / 4 |
| walker-run | 0.781 [-0.332, 0.897] | 0.671 [0.604, 0.725] | 0.993 [0.991, 0.993] | -3.087 [-295.998, -0.497] | -1.640 [-8.721, -0.536] | 2,000 / 4 |
| walker-walk-backwards | 0.980 [0.725, 0.991] | 0.292 [-14.285, 0.613] | 0.715 [0.105, 0.819] | 0.833 [0.738, 0.880] | 0.960 [0.934, 0.974] | 2,000 / 4 |
| walker-run-backwards | 0.953 [-0.030, 0.988] | 0.918 [0.811, 0.925] | 0.961 [0.778, 0.994] | 0.795 [-5.993, 0.797] | 0.983 [-1.080, 0.985] | 2,000 / 4 |

**주 정의: source task = τ인 A 상태, mlp, 학습된 z(s, τ)** (행 τ의 test 상태 수·에피소드 수는 오른쪽 열)

| 행 τ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards | 상태 / 에피소드 |
|---|---|---|---|---|---|---|
| walker-stand | 0.999 [0.962, 1.000] | 0.999 [0.998, 0.999] | 0.999 [0.998, 0.999] | 0.998 [0.998, 0.998] | 0.997 [0.994, 0.998] | 2,000 / 4 |
| walker-walk | 0.999 [0.999, 1.000] | 0.997 [0.993, 0.999] | 0.999 [0.998, 0.999] | 0.976 [0.937, 0.995] | 0.999 [0.993, 1.000] | 2,000 / 4 |
| walker-run | 0.992 [0.954, 0.996] | 0.998 [0.998, 0.999] | 0.999 [0.999, 1.000] | 0.937 [0.482, 0.958] | 0.975 [0.964, 0.981] | 2,000 / 4 |
| walker-walk-backwards | 0.998 [0.989, 0.999] | 0.998 [0.967, 0.999] | 0.997 [0.995, 0.998] | 0.998 [0.998, 0.999] | 0.998 [0.998, 0.999] | 2,000 / 4 |
| walker-run-backwards | 0.997 [0.966, 0.999] | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] | 0.999 [0.988, 0.999] | 0.997 [0.994, 0.999] | 2,000 / 4 |

주 정의: source task = τ인 A 상태: 행별 test 집합의 target 분산

| 행 τ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | 0.0026 | 0.0252 | 0.0006 | 0.0060 | 0.0002 |
| walker-walk | 0.0069 | 0.0199 | 0.0019 | 0.0002 | 0.0002 |
| walker-run | 0.0010 | 0.0143 | 0.0154 | 0.0006 | 0.0000 |
| walker-walk-backwards | 0.0029 | 0.0029 | 0.0002 | 0.0191 | 0.0007 |
| walker-run-backwards | 0.0025 | 0.1252 | 0.0079 | 0.0957 | 0.0047 |

**보조 정의: source task = τ인 A와 B 상태, linear, 학습된 z(s, τ)** (행 τ의 test 상태 수·에피소드 수는 오른쪽 열)

| 행 τ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards | 상태 / 에피소드 |
|---|---|---|---|---|---|---|
| walker-stand | 0.980 [0.779, 0.990] | 0.772 [0.617, 0.850] | 0.923 [0.849, 0.953] | 0.549 [0.388, 0.642] | 0.768 [0.565, 0.855] | 4,000 / 8 |
| walker-walk | 0.975 [0.913, 0.988] | 0.519 [0.215, 0.694] | 0.964 [0.952, 0.973] | -11.124 [-20.640, -6.293] | 0.590 [-0.438, 0.812] | 4,000 / 8 |
| walker-run | 0.916 [0.711, 0.957] | 0.708 [0.625, 0.760] | 0.993 [0.992, 0.994] | -4.200 [-44.500, -1.385] | -0.053 [-2.176, 0.517] | 4,000 / 8 |
| walker-walk-backwards | 0.972 [0.901, 0.985] | -0.157 [-3.139, 0.380] | 0.476 [-0.346, 0.712] | 0.750 [0.544, 0.840] | 0.945 [0.905, 0.963] | 4,000 / 8 |
| walker-run-backwards | 0.970 [0.906, 0.985] | 0.901 [0.846, 0.919] | 0.975 [0.940, 0.991] | 0.818 [0.406, 0.879] | 0.975 [0.753, 0.984] | 4,000 / 8 |

**보조 정의: source task = τ인 A와 B 상태, mlp, 학습된 z(s, τ)** (행 τ의 test 상태 수·에피소드 수는 오른쪽 열)

| 행 τ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards | 상태 / 에피소드 |
|---|---|---|---|---|---|---|
| walker-stand | 0.999 [0.993, 0.999] | 0.999 [0.997, 0.999] | 0.999 [0.997, 0.999] | 0.997 [0.996, 0.998] | 0.996 [0.993, 0.998] | 4,000 / 8 |
| walker-walk | 0.999 [0.998, 0.999] | 0.996 [0.994, 0.998] | 0.998 [0.998, 0.999] | 0.966 [0.932, 0.989] | 0.998 [0.991, 0.999] | 4,000 / 8 |
| walker-run | 0.996 [0.988, 0.998] | 0.998 [0.998, 0.999] | 1.000 [0.999, 1.000] | 0.956 [0.802, 0.986] | 0.992 [0.978, 0.997] | 4,000 / 8 |
| walker-walk-backwards | 0.998 [0.995, 0.999] | 0.996 [0.989, 0.998] | 0.996 [0.992, 0.997] | 0.997 [0.995, 0.999] | 0.998 [0.995, 0.999] | 4,000 / 8 |
| walker-run-backwards | 0.998 [0.995, 0.999] | 0.999 [0.999, 1.000] | 1.000 [1.000, 1.000] | 0.999 [0.998, 0.999] | 0.998 [0.997, 0.999] | 4,000 / 8 |

보조 정의: source task = τ인 A와 B 상태: 행별 test 집합의 target 분산

| 행 τ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | 0.0027 | 0.0274 | 0.0007 | 0.0070 | 0.0002 |
| walker-walk | 0.0042 | 0.0155 | 0.0016 | 0.0002 | 0.0001 |
| walker-run | 0.0025 | 0.0160 | 0.0169 | 0.0005 | 0.0001 |
| walker-walk-backwards | 0.0037 | 0.0024 | 0.0001 | 0.0209 | 0.0008 |
| walker-run-backwards | 0.0040 | 0.1134 | 0.0084 | 0.0648 | 0.0028 |

### paired bootstrap difference 판정 개수 (주 기준)

| 결과 구분 | probe | 비교 | 차이 있음 | 구분되지 않음 | 계산 안 됨 |
|---|---|---|---|---|---|
| pooled | linear | a_learned_vs_raw | 25 | 0 | 0 |
| pooled | linear | b_learned_vs_random | 25 | 0 | 0 |
| pooled | linear | c_diag_vs_row | 14 | 6 | 0 |
| pooled | linear | d_learned_vs_normmatched | 25 | 0 | 0 |
| pooled | mlp | a_learned_vs_raw | 22 | 3 | 0 |
| pooled | mlp | b_learned_vs_random | 25 | 0 | 0 |
| pooled | mlp | c_diag_vs_row | 12 | 8 | 0 |
| pooled | mlp | d_learned_vs_normmatched | 25 | 0 | 0 |
| not_fallen | linear | a_learned_vs_raw | 25 | 0 | 0 |
| not_fallen | linear | b_learned_vs_random | 25 | 0 | 0 |
| not_fallen | linear | c_diag_vs_row | 14 | 6 | 0 |
| not_fallen | linear | d_learned_vs_normmatched | 25 | 0 | 0 |
| not_fallen | mlp | a_learned_vs_raw | 22 | 3 | 0 |
| not_fallen | mlp | b_learned_vs_random | 25 | 0 | 0 |
| not_fallen | mlp | c_diag_vs_row | 12 | 8 | 0 |
| not_fallen | mlp | d_learned_vs_normmatched | 25 | 0 | 0 |
| indist_A | linear | a_learned_vs_raw | 24 | 1 | 0 |
| indist_A | linear | b_learned_vs_random | 25 | 0 | 0 |
| indist_A | linear | d_learned_vs_normmatched | 25 | 0 | 0 |
| indist_A | mlp | a_learned_vs_raw | 22 | 3 | 0 |
| indist_A | mlp | b_learned_vs_random | 25 | 0 | 0 |
| indist_A | mlp | d_learned_vs_normmatched | 25 | 0 | 0 |
| indist_AB | linear | a_learned_vs_raw | 25 | 0 | 0 |
| indist_AB | linear | b_learned_vs_random | 25 | 0 | 0 |
| indist_AB | linear | d_learned_vs_normmatched | 25 | 0 | 0 |
| indist_AB | mlp | a_learned_vs_raw | 21 | 4 | 0 |
| indist_AB | mlp | b_learned_vs_random | 25 | 0 | 0 |
| indist_AB | mlp | d_learned_vs_normmatched | 25 | 0 | 0 |

**pooled, linear, (a) 학습된 z(τ) − raw observation** (R² 차이 [95% 구간], 판정)

| 행 τ \ 열 τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | +0.0127 [+0.0117, +0.0144] | +0.1814 [+0.1532, +0.2262] | +0.0655 [+0.0522, +0.1016] | +0.3538 [+0.2908, +0.5307] | +0.1128 [+0.0852, +0.1625] |
| walker-walk | +0.0127 [+0.0117, +0.0144] | +0.1847 [+0.1559, +0.2300] | +0.0657 [+0.0523, +0.1020] | +0.3509 [+0.2880, +0.5260] | +0.1126 [+0.0848, +0.1621] |
| walker-run | +0.0127 [+0.0117, +0.0144] | +0.1793 [+0.1515, +0.2235] | +0.0660 [+0.0526, +0.1024] | +0.3418 [+0.2800, +0.5099] | +0.1114 [+0.0837, +0.1608] |
| walker-walk-backwards | +0.0127 [+0.0117, +0.0144] | +0.1860 [+0.1575, +0.2320] | +0.0651 [+0.0519, +0.1009] | +0.3525 [+0.2891, +0.5290] | +0.1140 [+0.0861, +0.1640] |
| walker-run-backwards | +0.0127 [+0.0117, +0.0144] | +0.1747 [+0.1472, +0.2180] | +0.0652 [+0.0520, +0.1005] | +0.3352 [+0.2695, +0.5025] | +0.1131 [+0.0845, +0.1645] |

**pooled, linear, (b) 학습된 z(τ) − random-init z(τ)** (R² 차이 [95% 구간], 판정)

| 행 τ \ 열 τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | +0.2466 [+0.2259, +0.3113] | +0.2153 [+0.1951, +0.2546] | +0.2207 [+0.1946, +0.2985] | +0.3220 [+0.2787, +0.4574] | +0.3127 [+0.2731, +0.3937] |
| walker-walk | +0.2456 [+0.2251, +0.3104] | +0.2190 [+0.1980, +0.2581] | +0.2219 [+0.1957, +0.3004] | +0.3189 [+0.2759, +0.4495] | +0.3116 [+0.2722, +0.3929] |
| walker-run | +0.2446 [+0.2241, +0.3092] | +0.2121 [+0.1921, +0.2506] | +0.2193 [+0.1935, +0.2969] | +0.3094 [+0.2683, +0.4341] | +0.3096 [+0.2700, +0.3908] |
| walker-walk-backwards | +0.2448 [+0.2243, +0.3088] | +0.2196 [+0.1987, +0.2593] | +0.2195 [+0.1939, +0.2966] | +0.3196 [+0.2760, +0.4520] | +0.3118 [+0.2725, +0.3929] |
| walker-run-backwards | +0.2454 [+0.2251, +0.3098] | +0.2086 [+0.1895, +0.2464] | +0.2199 [+0.1943, +0.2969] | +0.3018 [+0.2567, +0.4308] | +0.3110 [+0.2711, +0.3923] |

**pooled, linear, (d) 학습된 z(τ) − norm-matched random-init z(τ)** (R² 차이 [95% 구간], 판정)

| 행 τ \ 열 τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | +0.2134 [+0.1997, +0.2613] | +0.1995 [+0.1805, +0.2372] | +0.1976 [+0.1740, +0.2684] | +0.3019 [+0.2593, +0.4337] | +0.2761 [+0.2426, +0.3436] |
| walker-walk | +0.2083 [+0.1958, +0.2553] | +0.2037 [+0.1836, +0.2415] | +0.2008 [+0.1772, +0.2720] | +0.2979 [+0.2559, +0.4287] | +0.2713 [+0.2370, +0.3392] |
| walker-run | +0.2047 [+0.1908, +0.2528] | +0.1906 [+0.1725, +0.2261] | +0.1877 [+0.1654, +0.2549] | +0.2896 [+0.2502, +0.4121] | +0.2693 [+0.2362, +0.3384] |
| walker-walk-backwards | +0.2027 [+0.1898, +0.2483] | +0.2012 [+0.1816, +0.2395] | +0.1919 [+0.1695, +0.2613] | +0.2957 [+0.2531, +0.4255] | +0.2659 [+0.2344, +0.3326] |
| walker-run-backwards | +0.2066 [+0.1944, +0.2534] | +0.1918 [+0.1749, +0.2275] | +0.1946 [+0.1730, +0.2631] | +0.2798 [+0.2356, +0.4036] | +0.2687 [+0.2350, +0.3376] |

**pooled, linear, (c) 대각선 칸 − 같은 열의 다른 행** (R² 차이 [95% 구간], 판정)

| target 열 (= 대각선 행) | 비교 행 | 차이 | 판정 | 보조: 개별 구간 겹침 |
|---|---|---|---|---|
| walker-stand | walker-walk | +0.00001 [+0.00000, +0.00002] | 차이 있음 | 겹침 |
| walker-stand | walker-run | +0.00001 [-0.00001, +0.00004] | 구분되지 않음 | 겹침 |
| walker-stand | walker-walk-backwards | +0.00003 [+0.00002, +0.00005] | 차이 있음 | 겹침 |
| walker-stand | walker-run-backwards | +0.00002 [-0.00001, +0.00006] | 구분되지 않음 | 겹침 |
| walker-walk | walker-stand | +0.00336 [+0.00240, +0.00450] | 차이 있음 | 겹침 |
| walker-walk | walker-run | +0.00539 [+0.00402, +0.00719] | 차이 있음 | 겹침 |
| walker-walk | walker-walk-backwards | -0.00129 [-0.00251, -0.00042] | 차이 있음 | 겹침 |
| walker-walk | walker-run-backwards | +0.01007 [+0.00787, +0.01309] | 차이 있음 | 겹침 |
| walker-run | walker-stand | +0.00049 [+0.00038, +0.00072] | 차이 있음 | 겹침 |
| walker-run | walker-walk | +0.00028 [+0.00021, +0.00037] | 차이 있음 | 겹침 |
| walker-run | walker-walk-backwards | +0.00095 [+0.00068, +0.00158] | 차이 있음 | 겹침 |
| walker-run | walker-run-backwards | +0.00085 [+0.00047, +0.00180] | 차이 있음 | 겹침 |
| walker-walk-backwards | walker-stand | -0.00132 [-0.00335, +0.00060] | 구분되지 않음 | 겹침 |
| walker-walk-backwards | walker-walk | +0.00154 [+0.00021, +0.00444] | 차이 있음 | 겹침 |
| walker-walk-backwards | walker-run | +0.01063 [+0.00663, +0.01936] | 차이 있음 | 겹침 |
| walker-walk-backwards | walker-run-backwards | +0.01729 [+0.00754, +0.03780] | 차이 있음 | 겹침 |
| walker-run-backwards | walker-stand | +0.00029 [-0.00141, +0.00284] | 구분되지 않음 | 겹침 |
| walker-run-backwards | walker-walk | +0.00050 [-0.00083, +0.00252] | 구분되지 않음 | 겹침 |
| walker-run-backwards | walker-run | +0.00171 [+0.00070, +0.00341] | 차이 있음 | 겹침 |
| walker-run-backwards | walker-walk-backwards | -0.00089 [-0.00246, +0.00086] | 구분되지 않음 | 겹침 |

**pooled, mlp, (a) 학습된 z(τ) − raw observation** (R² 차이 [95% 구간], 판정)

| 행 τ \ 열 τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | -0.0000 [-0.0000, -0.0000] | +0.0003 [+0.0002, +0.0004] | +0.0000 [+0.0000, +0.0001] | +0.0005 [+0.0003, +0.0008] | +0.0000 [-0.0001, +0.0001] (구분 안 됨) |
| walker-walk | -0.0000 [-0.0001, -0.0000] | +0.0003 [+0.0002, +0.0004] | +0.0000 [+0.0000, +0.0000] | +0.0005 [+0.0003, +0.0008] | +0.0001 [+0.0000, +0.0001] |
| walker-run | -0.0000 [-0.0001, -0.0000] | +0.0003 [+0.0002, +0.0004] | +0.0000 [+0.0000, +0.0001] | +0.0005 [+0.0003, +0.0008] | +0.0001 [+0.0001, +0.0002] |
| walker-walk-backwards | -0.0000 [-0.0001, -0.0000] | +0.0003 [+0.0002, +0.0004] | +0.0000 [-0.0000, +0.0000] (구분 안 됨) | +0.0005 [+0.0003, +0.0009] | +0.0000 [-0.0002, +0.0001] (구분 안 됨) |
| walker-run-backwards | -0.0000 [-0.0000, -0.0000] | +0.0003 [+0.0002, +0.0004] | +0.0000 [+0.0000, +0.0000] | +0.0005 [+0.0003, +0.0008] | +0.0001 [+0.0000, +0.0002] |

**pooled, mlp, (b) 학습된 z(τ) − random-init z(τ)** (R² 차이 [95% 구간], 판정)

| 행 τ \ 열 τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | +0.0069 [+0.0057, +0.0089] | +0.0067 [+0.0054, +0.0084] | +0.0048 [+0.0040, +0.0063] | +0.0089 [+0.0065, +0.0136] | +0.0098 [+0.0070, +0.0140] |
| walker-walk | +0.0067 [+0.0054, +0.0087] | +0.0069 [+0.0055, +0.0087] | +0.0046 [+0.0038, +0.0061] | +0.0085 [+0.0061, +0.0129] | +0.0103 [+0.0074, +0.0146] |
| walker-run | +0.0067 [+0.0055, +0.0087] | +0.0068 [+0.0054, +0.0086] | +0.0045 [+0.0037, +0.0059] | +0.0101 [+0.0074, +0.0154] | +0.0106 [+0.0075, +0.0155] |
| walker-walk-backwards | +0.0063 [+0.0051, +0.0081] | +0.0066 [+0.0053, +0.0083] | +0.0045 [+0.0038, +0.0059] | +0.0083 [+0.0060, +0.0127] | +0.0098 [+0.0072, +0.0136] |
| walker-run-backwards | +0.0063 [+0.0051, +0.0082] | +0.0065 [+0.0052, +0.0082] | +0.0045 [+0.0037, +0.0059] | +0.0089 [+0.0065, +0.0133] | +0.0099 [+0.0070, +0.0143] |

**pooled, mlp, (d) 학습된 z(τ) − norm-matched random-init z(τ)** (R² 차이 [95% 구간], 판정)

| 행 τ \ 열 τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | +0.0057 [+0.0046, +0.0074] | +0.0063 [+0.0051, +0.0080] | +0.0042 [+0.0035, +0.0055] | +0.0084 [+0.0061, +0.0126] | +0.0091 [+0.0067, +0.0126] |
| walker-walk | +0.0052 [+0.0043, +0.0066] | +0.0058 [+0.0047, +0.0074] | +0.0042 [+0.0035, +0.0055] | +0.0081 [+0.0059, +0.0124] | +0.0081 [+0.0059, +0.0114] |
| walker-run | +0.0064 [+0.0053, +0.0083] | +0.0062 [+0.0050, +0.0078] | +0.0041 [+0.0034, +0.0054] | +0.0083 [+0.0060, +0.0126] | +0.0088 [+0.0062, +0.0127] |
| walker-walk-backwards | +0.0057 [+0.0048, +0.0073] | +0.0065 [+0.0052, +0.0082] | +0.0044 [+0.0037, +0.0057] | +0.0079 [+0.0058, +0.0121] | +0.0086 [+0.0062, +0.0124] |
| walker-run-backwards | +0.0056 [+0.0046, +0.0072] | +0.0060 [+0.0048, +0.0076] | +0.0042 [+0.0035, +0.0055] | +0.0079 [+0.0058, +0.0119] | +0.0096 [+0.0071, +0.0136] |

**pooled, mlp, (c) 대각선 칸 − 같은 열의 다른 행** (R² 차이 [95% 구간], 판정)

| target 열 (= 대각선 행) | 비교 행 | 차이 | 판정 | 보조: 개별 구간 겹침 |
|---|---|---|---|---|
| walker-stand | walker-walk | +0.00001 [+0.00001, +0.00002] | 차이 있음 | 겹침 |
| walker-stand | walker-run | +0.00001 [+0.00001, +0.00002] | 차이 있음 | 겹침 |
| walker-stand | walker-walk-backwards | +0.00001 [+0.00001, +0.00001] | 차이 있음 | 겹침 |
| walker-stand | walker-run-backwards | +0.00001 [+0.00000, +0.00001] | 차이 있음 | 겹침 |
| walker-walk | walker-stand | -0.00002 [-0.00003, -0.00001] | 차이 있음 | 겹침 |
| walker-walk | walker-run | +0.00001 [-0.00001, +0.00004] | 구분되지 않음 | 겹침 |
| walker-walk | walker-walk-backwards | -0.00001 [-0.00003, +0.00001] | 구분되지 않음 | 겹침 |
| walker-walk | walker-run-backwards | +0.00001 [-0.00001, +0.00003] | 구분되지 않음 | 겹침 |
| walker-run | walker-stand | +0.00000 [-0.00000, +0.00000] | 구분되지 않음 | 겹침 |
| walker-run | walker-walk | +0.00002 [+0.00001, +0.00002] | 차이 있음 | 겹침 |
| walker-run | walker-walk-backwards | +0.00002 [+0.00002, +0.00003] | 차이 있음 | 겹침 |
| walker-run | walker-run-backwards | +0.00001 [+0.00000, +0.00001] | 차이 있음 | 겹침 |
| walker-walk-backwards | walker-stand | +0.00003 [+0.00001, +0.00005] | 차이 있음 | 겹침 |
| walker-walk-backwards | walker-walk | +0.00003 [+0.00001, +0.00007] | 차이 있음 | 겹침 |
| walker-walk-backwards | walker-run | +0.00002 [-0.00001, +0.00008] | 구분되지 않음 | 겹침 |
| walker-walk-backwards | walker-run-backwards | +0.00001 [-0.00001, +0.00006] | 구분되지 않음 | 겹침 |
| walker-run-backwards | walker-stand | +0.00007 [+0.00003, +0.00015] | 차이 있음 | 겹침 |
| walker-run-backwards | walker-walk | +0.00000 [-0.00002, +0.00002] | 구분되지 않음 | 겹침 |
| walker-run-backwards | walker-run | -0.00003 [-0.00011, +0.00001] | 구분되지 않음 | 겹침 |
| walker-run-backwards | walker-walk-backwards | +0.00008 [+0.00001, +0.00023] | 차이 있음 | 겹침 |


## 4. 수치로만 적는 사실

### raw observation과 학습된 z가 각 열에서 모두 R² 0.99 이상인가 (pooled)

| probe | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| linear | raw 0.9864 / 학습 z 최소 0.9991 / 아니오 | raw 0.7914 / 학습 z 최소 0.9661 / 아니오 | raw 0.9321 / 학습 z 최소 0.9971 / 아니오 | raw 0.6193 / 학습 z 최소 0.9545 / 아니오 | raw 0.8800 / 학습 z 최소 0.9914 / 아니오 |
| mlp | raw 1.0000 / 학습 z 최소 0.9999 / 예 | raw 0.9995 / 학습 z 최소 0.9998 / 예 | raw 0.9999 / 학습 z 최소 0.9999 / 예 | raw 0.9992 / 학습 z 최소 0.9997 / 예 | raw 0.9997 / 학습 z 최소 0.9997 / 예 |

- **MLP**: 다섯 열 모두에서 raw observation과 학습된 z 다섯 행이 모두 R² 0.99 이상이다 (최솟값: raw 0.9992, 학습된 z 0.9997).
- **linear**: 다섯 열 모두에서 둘 다 0.99 이상인 열은 없다 (raw 0.619~0.986, 학습된 z 0.955~0.999).

### random-init 5×5에서 같은 열 안 행끼리 R² 최대 차이 (pooled, 최댓값 − 최솟값)

| probe | 입력 | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|---|
| linear | 학습된 z | 0.00003 | 0.01136 | 0.00095 | 0.01861 | 0.00260 |
| linear | random-init z | 0.00200 | 0.00152 | 0.00284 | 0.00164 | 0.00210 |
| linear | norm-matched random-init z | 0.01069 | 0.00769 | 0.01336 | 0.00482 | 0.01135 |
| mlp | 학습된 z | 0.00001 | 0.00003 | 0.00002 | 0.00003 | 0.00012 |
| mlp | random-init z | 0.00065 | 0.00040 | 0.00029 | 0.00181 | 0.00070 |
| mlp | norm-matched random-init z | 0.00117 | 0.00070 | 0.00034 | 0.00051 | 0.00158 |

- random-init 모델 자체의 task embedding L2 norm은 0.085~0.108이다. 학습된 embedding은 0.992~0.998, norm-matched는 0.995다 (`results/exp1/latent_meta.json`, `normmatched_meta.json`).

### pooled 결과와 fallen 제외 결과의 칸별 차이 (pooled R² − fallen 제외 R²)

**linear, learned: pooled R² − fallen 제외 R²**

| 행 τ \ 열 τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | +0.0179 | +0.0142 | +0.0010 | +0.0019 | +0.0179 |
| walker-walk | +0.0181 | +0.0117 | +0.0009 | +0.0031 | +0.0192 |
| walker-run | +0.0180 | +0.0153 | +0.0009 | +0.0053 | +0.0226 |
| walker-walk-backwards | +0.0184 | +0.0110 | +0.0012 | +0.0028 | +0.0139 |
| walker-run-backwards | +0.0181 | +0.0181 | +0.0014 | +0.0077 | +0.0168 |

**linear, random: pooled R² − fallen 제외 R²**

| 행 τ \ 열 τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | +4.3753 | +0.0507 | +0.0474 | +0.0146 | +0.6643 |
| walker-walk | +4.3653 | +0.0510 | +0.0482 | +0.0149 | +0.6644 |
| walker-run | +4.3362 | +0.0506 | +0.0466 | +0.0147 | +0.6610 |
| walker-walk-backwards | +4.3369 | +0.0506 | +0.0472 | +0.0144 | +0.6589 |
| walker-run-backwards | +4.3435 | +0.0506 | +0.0469 | +0.0139 | +0.6577 |

**linear, normmatched: pooled R² − fallen 제외 R²**

| 행 τ \ 열 τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | +3.4709 | +0.0452 | +0.0336 | +0.0098 | +0.5504 |
| walker-walk | +3.4027 | +0.0452 | +0.0361 | +0.0113 | +0.5486 |
| walker-run | +3.3605 | +0.0465 | +0.0318 | +0.0110 | +0.5470 |
| walker-walk-backwards | +3.2998 | +0.0450 | +0.0344 | +0.0100 | +0.5296 |
| walker-run-backwards | +3.3447 | +0.0449 | +0.0342 | +0.0080 | +0.5319 |

**linear, raw observation: pooled R² − fallen 제외 R²**: walker-stand +0.1709, walker-walk +0.0933, walker-run +0.0031, walker-walk-backwards +0.0643, walker-run-backwards +0.3497

**mlp, learned: pooled R² − fallen 제외 R²**

| 행 τ \ 열 τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | +0.0011 | +0.0001 | +0.0000 | +0.0000 | +0.0008 |
| walker-walk | +0.0013 | +0.0001 | +0.0001 | +0.0000 | +0.0006 |
| walker-run | +0.0013 | +0.0001 | +0.0000 | +0.0000 | +0.0005 |
| walker-walk-backwards | +0.0013 | +0.0001 | +0.0001 | +0.0000 | +0.0009 |
| walker-run-backwards | +0.0012 | +0.0001 | +0.0000 | +0.0000 | +0.0006 |

**mlp, random: pooled R² − fallen 제외 R²**

| 행 τ \ 열 τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | +0.1651 | +0.0030 | +0.0028 | +0.0007 | +0.0255 |
| walker-walk | +0.1586 | +0.0031 | +0.0027 | +0.0007 | +0.0266 |
| walker-run | +0.1604 | +0.0031 | +0.0026 | +0.0007 | +0.0276 |
| walker-walk-backwards | +0.1512 | +0.0029 | +0.0027 | +0.0006 | +0.0255 |
| walker-run-backwards | +0.1529 | +0.0030 | +0.0026 | +0.0007 | +0.0257 |

**mlp, normmatched: pooled R² − fallen 제외 R²**

| 행 τ \ 열 τ′ | walker-stand | walker-walk | walker-run | walker-walk-backwards | walker-run-backwards |
|---|---|---|---|---|---|
| walker-stand | +0.1386 | +0.0028 | +0.0027 | +0.0009 | +0.0244 |
| walker-walk | +0.1283 | +0.0027 | +0.0027 | +0.0007 | +0.0217 |
| walker-run | +0.1571 | +0.0028 | +0.0026 | +0.0009 | +0.0246 |
| walker-walk-backwards | +0.1329 | +0.0029 | +0.0028 | +0.0005 | +0.0235 |
| walker-run-backwards | +0.1363 | +0.0028 | +0.0026 | +0.0006 | +0.0259 |

**mlp, raw observation: pooled R² − fallen 제외 R²**: walker-stand +0.0003, walker-walk +0.0002, walker-run +0.0001, walker-walk-backwards +0.0001, walker-run-backwards +0.0008

- fallen 제외 test 집합: 상태 20,395개, 에피소드 60개.
- walker-stand 열의 target 분산은 pooled 0.1673, fallen 제외 0.0077이다. walker-run-backwards 열은 pooled 0.0079, fallen 제외 0.0026이다.

### 기타 실행 기록

- **MLP seed 간 표준편차** (pooled, 칸별): 최대 0.0014, 중앙값 0.00026.
- **MLP epoch**: 400개 중 34개가 early stopping 없이 최대 200 epoch까지 돌았다. 400개 전체의 epoch 수는 최소 48, 중앙값 126, 최대 200이다. 200 epoch에 닿은 probe 목록은 `results/exp1/eval/facts.json` `mlp_epochs`에 있다.
  - 입력별로 보면 학습된 z 1개, random-init 14개, norm-matched 12개, raw 7개다.
  - PREREG의 최대 epoch를 바꾸지 않았다.
- **ridge λ**: 80개 중 19개가 후보의 가장 작은 값(1e-4)에서 골라졌다. 가장 큰 값(1e4)은 0개다.
  - 고른 값 전체는 `results/exp1/probes/linear_meta.json`에 있다.
  - PREREG의 λ 후보를 바꾸지 않았다.
- **R² 계산 불가**: 없었다. bootstrap에서 버린 회차도 없었다 (모든 칸 `boot_dropped` 0).
- **표준편차가 1e-8보다 작은 입력 차원**: 없었다.

## 5. 선택 실험: counterfactual task swap (`results/exp1/task_swap/task_swap.json`, `code/task_swap.py`)

### 정의

- latent 768차원은 8차원 묶음 96개이고, 묶음마다 softmax라 categorical 분포다 (SimNorm, `common/layers.py:74-91`).
- **거리**: d_TV(z, z′) = 묶음별 total variation 0.5‖z_g − z′_g‖₁의 96개 평균. 범위 0~1. L2 거리도 함께 계산했다.
- **비교 대상**: test 상태 30,000개.
  - **swap**: 같은 s_t에 embedding τ 대 τ′ (walker 10쌍)
  - **다른 상태**: 같은 τ에서 무작위로 짝지은 다른 test 상태 (seed 20261029)
  - **인접 step**: 같은 τ에서 같은 에피소드의 s_t 대 s_{t+1}
- latent는 로컬에서 만든 파일이다. Kaggle 파일과 최대 2.5e-6 차이가 있다.
- **task embedding을 바꾼 입력 z(s, τ′)는, s가 τ′ 아래에서 방문된 상태가 아니면 학습 분포 밖일 수 있다.**

### 결과 (d_TV 중앙값의 범위: swap은 10쌍, 나머지는 task 5개)

| encoder | swap | 다른 상태 | 인접 step |
|---|---|---|---|
| 학습된 모델 | 0.0247 ~ 0.1806 | 0.4590 ~ 0.4788 | 0.1128 ~ 0.1228 |
| random-init | 0.0022 ~ 0.0031 | 0.3892 ~ 0.3893 | 0.3707 ~ 0.3709 |
| norm-matched random-init | 0.0231 ~ 0.0298 | 0.3883 ~ 0.3903 | 0.3697 ~ 0.3715 |

학습된 모델의 swap 쌍별 d_TV 중앙값:

| 쌍 | 값 | 쌍 | 값 |
|---|---|---|---|
| stand–walk | 0.0247 | walk–walk-backwards | 0.0710 |
| stand–walk-backwards | 0.0596 | run–run-backwards | 0.0840 |
| walk-backwards–run-backwards | 0.1524 | walk–run | 0.1698 |
| run–walk-backwards | 0.1745 | stand–run | 0.1748 |
| stand–run-backwards | 0.1785 | walk–run-backwards | 0.1806 |

L2 거리 중앙값 범위 (학습된 모델): swap 0.318~2.042, 다른 상태 5.017~5.317, 인접 step 1.379~1.532.

## 6. 비공개 Kaggle dataset 업로드

| 항목 | 값 |
|---|---|
| dataset | `seungwonryoo/tdmpc2-probe-exp1-rollouts`, version 1 |
| 공개 설정 | **private**. 업로드 전 CLI 기본값이 private임을 확인했다 (`--public`을 주지 않음). 업로드 뒤 API `is_private: True`, 인증 없는 조회는 HTTP 403 |
| 올린 것 | `exp1_rollouts.tar` 113,387,520 B (sha256 `a79cc7e75c8ab663b8099260bd3ed8f7aa11fd5f2130898a307042cc05ba860a`), `MANIFEST_sha256.txt` |
| Kaggle 쪽 형태 | tar가 자동으로 풀려 `exp1_rollouts/rollouts/<task>/<종류>/epNN.{npz,json}` 600개 파일로 저장됐다 |
| 파일별 sha256 | [results/exp1/rollouts_manifest_sha256.txt](results/exp1/rollouts_manifest_sha256.txt) (600개). probe 커널이 받은 파일이 이 목록과 모두 같았다 |
| 올리지 않은 것 | latent(4.6 GB), 그 밖의 파생 데이터 |

로컬에도 `data/rollouts/`에 그대로 있다. probe 예측값(약 115 MB)은 로컬 `results/exp1/probes/`와 Kaggle 커널 출력에만 두고 GitHub에는 올리지 않았다. 공개 저장소에 데이터를 올리는 일이 되기 때문이다.

## 7. PREREG 대비 추가·변경

| 항목 | 내용 | 기록 시점 |
|---|---|---|
| 보조 대조군 A1 | norm-matched random-init latent | PREREG 사후 추가 항목, 2026-10-26 08:31 UTC, probe 학습 전 (커밋 `9d67b7a`) |
| 결과 구분 A2 | fallen 제외 pooled test | 같은 사후 추가 항목 |
| 실행 위치 | 로컬 CPU 계획(명시 없음) → Kaggle T4 | 비용 측정 후. 이유는 1절 |
| 데이터 파일 | Kaggle에서 다시 만든 latent·targets가 로컬과 sha256이 다름 (최대 2.5e-6 / 3.3e-16) | 실행 후 발견, 1절. 평가는 Kaggle 파일로 했다 |
| probe 커널 재시작 | 첫 커널을 시작 12분 만에 내림 | 1절 |
| norm-matched 경로 확인 | 크기를 바꾸지 않은 random-init을 같은 함수로 다시 encode한 값과 저장된 latent_random의 차이: 로컬 2.1e-7, Kaggle 0.0 | 실행 기록 |

PREREG의 설정(λ 후보, MLP 하이퍼파라미터, 표준화, 분할, 통계 기준, seed)은 바꾸지 않았다.

## 8. 실패 및 누락

- 누락된 probe, 계산되지 않은 R²는 없다.
- in-distribution 결과에서 비교 (c)(대각선 대 다른 행)는 계산하지 않았다. 행마다 test 집합이 달라 같은 표본에서 짝지을 수 없다.

## 9. 이 결과로 결론 낼 수 없는 것

- **decision sufficiency**: reward를 예측할 수 있다는 것은 그 latent로 좋은 결정을 내릴 수 있다는 것과 다르다.
- **task-independent shared encoder bottleneck**: 이 encoder는 task-conditioned다. task-independent encoder에 대해서는 아무것도 말할 수 없다.
- **정보의 존재·부재**: probe R²는 그 probe가 꺼낼 수 있는 정도(information accessibility)다. R²가 낮다고 정보가 없다고, 높다고 encoder가 그 정보를 쓴다고 말할 수 없다.
- **task 수행 실패 여부**: cross-task 칸의 R²는 그 task를 수행하는 능력과 관계가 없다.
- **run 계열 저성능의 원인**: walker-run, walker-run-backwards의 낮은 return(DATA_SUMMARY.md)의 원인은 이 실험으로 알 수 없다.
- **ceiling**: MLP 결과는 raw observation과 학습된 z가 모든 열에서 R² 0.99 이상이다.
  - 이 범위에서 paired difference가 "차이 있음"으로 나온 칸이 있어도, 그 차이의 크기는 R² 1e-5~1e-3 수준이다 (paired_differences.csv).
  - 그 크기가 의미 있는지는 이 결과로 정할 수 없다.
- **일반화**: 체크포인트 하나(mt30-48M, 학습 시드 하나), walker 몸체 하나, 이 상태 분포(A·B·C 혼합, pooled의 약 1/3이 fallen)에 대한 결과다.
- **행 방향 비교**: 같은 행 안 다른 열의 R² 차이는 target 분산 차이를 반영할 수 있다 (열별 분산 0.0079~0.1781).
- **counterfactual swap**: 바꾼 embedding 입력이 학습 분포 밖일 수 있어, 거리가 학습된 계산을 반영하는지 알 수 없다.

## 10. 재현

```bash
# 데이터 (Step 4): Kaggle GPU, kaggle/exp1_collect_body.py
# Step 5 (로컬 또는 Kaggle CPU)
cd code
python step5_latents_targets.py --data ../data/rollouts --out ../data/pooled
python extract_normmatched.py --pooled ../data/pooled
python torso_height.py --pooled ../data/pooled
python make_split.py --pooled ../data/pooled --out ../data/split.json --report ../results/exp1/split_check.json
# Step 6 (실제 실행: Kaggle T4, kaggle/exp1_probe_body.py)
python probe.py linear --data ../data --out ../results/exp1/probes
python probe.py mlp --data ../data --out ../results/exp1/probes --device cuda --worker K --num-workers 4 --tag wK
# 평가 (로컬 CPU, Kaggle의 targets와 분할 파일 사용)
python evaluate_probes.py --data ../data_eval_kaggle --probes ../results/exp1/probes --out ../results/exp1/eval
python make_tables.py --eval ../results/exp1/eval --probes ../results/exp1/probes
python task_swap.py --data ../data --out ../results/exp1/task_swap
```
