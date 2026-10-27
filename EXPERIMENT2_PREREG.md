# EXPERIMENT2_PREREG — 극단적 task swap

작성: 2026-10-27, Experiment 2 실행 전.
이 문서의 설정과 판정 기준은 결과를 본 뒤 바꾸지 않는다. 바꿔야 하면 실행을 멈추고 보고한다.

## 0. 지시문 해석 (실행 전 기록)

- **MLP 실행 여부**: 지시문 A0.3에는 "linear(ridge)만. MLP는 실행하지 않음"이 남아 있다. 하지만 3절 사용자 가설(주 지표 MLP), A0.5(MLP 판정 기준과 random-init 기준선), A2.1(linear·MLP 표와 칸 분류 표)은 MLP 결과를 요구한다.
  - 따라서 MLP probe를 실행한다.
  - λ 확장 보조 결과와 서 있는 상태 전용 probe는 지시문대로 linear만 실행한다.
- **"pooled 주 결과" (서 있는 상태 전용 probe의 행 선택)**: 그 probe가 linear이므로 linear pooled 주 결과를 기준으로 한다 (4절).
- **MLP 방식**: Experiment 2의 MLP는 모든 행(대각선 포함)을 같은 multi-output 방식(출력 5개)으로 학습한다. 대각선 기준도 이번 실행의 multi-output 값이다. Experiment 1의 single-output 값은 판정에 쓰지 않는다.

## 1. 입력

| 항목 | 값 |
|---|---|
| checkpoint | `mt30-48M.pt` (sha256 `bacb9ae1891a9b03dc3942a287080beb573220ea9d0263c5efd7a94a5c2ec419`) |
| 상태 | Experiment 1 pooled 150,000 상태. 관측 `obs.npy` sha256 `64b9e182d5be7e585c7a3ca967d1e9a5f924f577b3a0c6216b6e2fd217e1477b` |
| 분할 | Experiment 1 `split_labels.npy` (sha256 `5c2121764a9d5d80259e55463623cd089efd50f48a3057b38e5ac2d5eb24dcb2`): train 90,000 / validation 30,000 / test 30,000 |
| target | r_τ′(s_t), τ′ = walker 5개. Experiment 1 평가에 쓴 Kaggle 생성 `targets.npy` (sha256 `10fcd0ebfb2b45164458062f011aebf2e631176d626cd9060dfcd78cf6cf2f64`) |
| fallen | torso_height < 0.6 m (`torso_height.npy` sha256 `1610695ec26a0b7579a36723f8c833e1759948145629ae552a38724b470258ca`) |

실행 위치: Kaggle(Tesla T4 2장)에서 비공개 rollout dataset으로 위 파일을 Experiment 1과 같은 스크립트로 다시 만든다. obs, split, targets, torso_height의 sha256이 위 값과 다르면 probe를 실행하지 않고 멈춘다.

## 2. 행 (embedding)

| 행 | 개수 | 인코딩 |
|---|---|---|
| mt30 task embedding | 30 (walker 5 + 나머지 25), task id 0~29 | 공식 `WorldModel.encode(obs, task_id)` |
| 영벡터 | 1 | 주입 함수 |
| 무작위 방향 벡터 | 5 | 주입 함수. `numpy.random.default_rng(20261102).standard_normal((5, 64))`의 각 행을 L2 norm 1로 나눈 뒤 c30을 곱한다. c30 = 학습된 30개 embedding L2 norm의 평균 |
| random-init 기준선 (표 밖 참고 행) | 1 | Experiment 1 random-init 모델(`torch.manual_seed(20261028)`, `WorldModel(cfg)`)의 task id 0(walker-stand) 행을 공식 encode로 인코딩 |

- 학습된 30개 embedding의 L2 norm은 모두 1 미만이다 (실행 전 확인: 0.9548~0.9998). 따라서 `nn.Embedding(max_norm=1)`의 재정규화는 일어나지 않는다. 주입 벡터에는 재정규화를 적용하지 않는다.
- **주입 함수**: `inject(model, obs, v) = model._encoder['state'](cat([obs, v 반복], -1))`. 공식 `encode`(`world_model.py:105-114`)에서 embedding lookup만 뺀 경로이고, embedding 표는 수정하지 않는다.
- **주입 함수 확인 (실행 전, 로컬 CPU)**: walker 5개 embedding 벡터를 주입 함수에 넣은 결과와 Experiment 1 `latent_learned.npy`의 차이. 상태 3,000개(인덱스 0부터 50 간격)에서 최대 절대 차이 ≤ 1e-5가 아니면 멈춘다.
  - Kaggle 실행에서도 같은 상태에서 주입 함수와 공식 encode를 비교해 기록한다.
- **기록**: 각 행의 L2 norm, walker 5개 embedding 각각과의 cosine similarity.
  - 영벡터의 cosine은 정의되지 않으므로 "정의 안 됨"으로 적는다.
- **정렬**: walker 5개와의 cosine 최댓값 내림차순. 영벡터는 맨 뒤에 둔다. 같은 값이면 task id 순서, 그다음 대조 벡터 순서다.

## 3. Probe

모든 probe는 같은 행의 z(s, τ)(768차원)를 입력으로 받는다. encoder는 고정이다.

### Linear (ridge)

Experiment 1(`code/probe.py`)과 같다.
- feature와 target은 train 통계로 standardization한다.
- 출력 5개를 각각 따로 fit한다.
- λ는 validation R²로 고르고, 같으면 큰 λ를 고른다.
- **주 결과**: λ ∈ 10^[−4:0.5:4] (17개)
- **보조 결과 (λ 확장)**: λ ∈ 10^[−8:0.5:4] (25개). 표의 모든 행(walker 5 포함)과 random-init 기준선에 적용한다.

### MLP (multi-output)

| 항목 | 값 |
|---|---|
| 구조 | 768 → 256 → ReLU → 256 → ReLU → 5 |
| 학습 | Adam lr 1e-3, weight decay 0, batch 512, 최대 200 epoch |
| early stopping | validation의 5개 target 평균 MSE (standardized 단위), patience 20, 최고 상태 복원 |
| loss | 5개 standardized target MSE 평균 |
| seed | 0~4 (초기화와 batch 순서). 칸 값은 seed 5개 R²의 평균 |
| 정밀도 | float32 |

### 서 있는 상태 전용 probe (보조, linear만)

- train, validation, test 모두에서 torso_height ≥ 0.6 m인 상태만 쓴다. standardization도 그 train 부분집합 통계로 한다.
- λ 후보는 주 결과와 같다 (10^[−4:0.5:4]).
- **행**: walker 5개 embedding, raw observation(24차원), Experiment 1 random-init 모델의 walker 5개 행, 그리고 4절 규칙으로 고른 비-walker embedding 3개.

## 4. 서 있는 상태 전용 probe의 비-walker 행 선택 규칙

- 후보: walker가 아닌 mt30 task embedding 25개. 영벡터와 무작위 벡터는 후보가 아니다.
- 기준: linear pooled 주 결과(λ 10^[−4:0.5:4])에서 5열 평균 test R²(점추정).
- 가장 낮은 3개를 고른다. 같으면 task id가 작은 쪽을 고른다.
- 선택은 Kaggle 실행 안에서 이 규칙대로 자동으로 한다.

## 5. 지표와 통계

- **test R²**: 1 − SSE/SST. reward 원래 단위다.
- **RMSE**: sqrt(SSE/n), reward 단위. MLP는 seed 5개 RMSE의 평균이다.
- **bootstrap**: test 에피소드 단위 1,000회, seed 20261027 (Experiment 1과 같다).
  - 한 결과 구분 안의 모든 행과 열이 같은 resampling을 공유한다(paired).
  - 95% 구간은 2.5/97.5 백분위다.
  - 표본 SST < 1e-10이면 그 표본을 버리고 개수를 기록한다.
- **paired 차이**: Δ = R²(대각선) − R²(해당 행). 대각선 = 그 열 τ′와 같은 walker embedding 행이다. bootstrap 표본마다 계산한다.

## 6. 판정 기준 (linear, MLP 각각, 열마다)

- **"실질적으로 낮음"**: Δ의 95% 구간 하한 > 0 이고 Δ 점추정 ≥ 0.02.
- **"구분되지 않음 또는 실질적 차이 없음"**: 그 외.
- **0.02의 근거**: Experiment 1 학습된 z linear probe의 같은 열 안 walker 행 간 최대 차이 0.019 (`results/exp1/eval/facts.json`). MLP에도 같은 값을 쓴다 (사용자 지시).
- **보조 기준 (판정과 별도 표시)**: RMSE 비율(해당 행 / 대각선, 점추정) ≥ 1.5.
- **MLP 추가 기준선**: random-init 기준선 행의 multi-output MLP R²(같은 결과 구분, 같은 열, 점추정)보다 어떤 행의 MLP R² 점추정이 낮으면 "무작위 encoder 수준 미만"으로 따로 표시한다.
  - linear에도 같은 표시를 참고로 계산한다.
- **칸 분류**: (linear "실질적으로 낮음" 여부, MLP "실질적으로 낮음" 여부) → 둘 다 아님 / linear만 / MLP만 / 둘 다.
- **판정 개수 요약**: 25개 비-walker 행에서 열별로 "실질적으로 낮음" 칸 수를 센다. 영벡터와 무작위 5개는 따로 센다.

## 7. 결과 구분

| 구분 | 내용 |
|---|---|
| pooled test (주) | test 30,000 상태 |
| fallen 제외 test (보조) | pooled로 학습한 같은 probe를 torso_height ≥ 0.6 m인 test 상태에서 평가. 대각선 기준도 같은 부분집합에서 구한다 |
| λ 확장 (보조) | pooled test에서 같은 판정 |
| 서 있는 상태 전용 probe (보조) | 그 probe의 walker 대각선을 기준으로 같은 판정 |

"판정이 바뀐 칸"은 같은 (행, 열)에서 linear 주 결과 판정과 다른 칸으로 센다.

## 8. 저장 정책과 거리

- latent는 저장하지 않는다. 행 하나마다 인코딩 → probe → 예측 저장 → latent 삭제 순서로 처리한다.
- 저장하는 예측값(test, validation)은 Kaggle 출력과 로컬에만 두고 GitHub에는 올리지 않는다.
- **거리**: 각 행의 z와 walker k(5개) embedding z 사이의 d_TV. Experiment 1 정의(SimNorm 8차원 묶음 96개의 total variation 0.5‖z_g − z′_g‖₁ 평균)를 같은 상태에서 계산한다.
  - test 30,000 상태의 중앙값, 5%, 95% 분위수를 저장한다.
- **산점도**: x = 행의 5열에 대한 "그 열 대각선 z와의 d_TV 중앙값"의 평균. y = 그 행의 5열 평균 R²(pooled 주 결과). linear, MLP 각각 그린다.
- **관계 수치 (서술용, 판정 없음)**: Spearman 순위 상관과 Pearson 상관.
  - (a) 36행 전체
  - (b) 비-walker task 25행

## 9. 학습 분포 주의 문구 (결과 문서에 그대로 옮김)

walker 관측(24차원)은 mt30 최대 관측 길이와 같아 padding이 없다. 비-walker task embedding은 학습 때 뒤쪽이 0으로 채워진 관측과만 짝지어졌으므로, walker 관측과의 조합은 학습 분포 밖이다.

(근거: `tdmpc2/envs/wrappers/multitask.py:20` 최대 관측 길이, `:44-47` 뒤쪽 0 채움.)

## 10. 실행 제한

- 학습은 probe만 한다. encoder와 world model은 고정이다.
- Experiment 1 데이터는 다시 수집하지 않는다.
- 공개 위치에 아무것도 올리지 않는다.
