# EXPERIMENT2_RESULTS (극단적 task swap)

작성: 2026-10-28. 설정과 판정 기준은 [EXPERIMENT2_PREREG.md](EXPERIMENT2_PREREG.md)(커밋 `093215a`, 실행 전)를 그대로 따랐다.
아래 숫자는 모두 결과 파일에서 읽은 값이다. 해석, 원인 추측, 연구 기여 주장은 쓰지 않는다.

## 학습 분포 주의 (PREREG 9절을 그대로 옮김)

walker 관측(24차원)은 mt30 최대 관측 길이와 같아 padding이 없다. 비-walker task embedding은 학습 때 뒤쪽이 0으로 채워진 관측과만 짝지어졌으므로, walker 관측과의 조합은 학습 분포 밖이다.

(근거: `tdmpc2/envs/wrappers/multitask.py:20` 최대 관측 길이, `:44-47` 뒤쪽 0 채움.)

영벡터와 무작위 방향 벡터 행도 학습 때 encoder에 들어간 적이 없는 입력이다.

## 1. 실행 개요

| 항목 | 값 |
|---|---|
| Kaggle 커널 | `seungwonryoo/tdmpc2-probe-exp2` (비공개), Tesla T4 2장. 저장소 커밋 `4f02317`. 2026-10-27 07:03 UTC 시작, 3,842초 |
| checkpoint | `mt30-48M.pt`, sha256 `bacb9ae1…c419` (커널에서 다시 계산, PREREG와 같음) |
| 데이터 재생성 | 비공개 dataset의 rollout 600개 파일 sha256 대조 `ROLLOUTS_SHA_OK`. 다시 만든 `obs`, `targets`, `torso_height`, `split_labels`의 sha256이 PREREG 1절 값과 모두 같음 (`results/exp2/kaggle/data_sha_check.txt`) |
| 주입 함수 확인 | walker 5개 embedding, 상태 3,000개. 주입 함수와 공식 `encode`의 최대 절대 차이 0.0, Kaggle에서 다시 만든 Experiment 1 latent와도 0.0 (`results/exp2/kaggle/inject_check_kaggle.json`). 로컬 CPU 사전 확인은 최대 2.4e-7 (`results/exp2/inject_check_local.json`). 기준 ≤ 1e-5 통과 |
| 행 | 37개: mt30 task embedding 30, 영벡터 1, 무작위 방향 벡터 5, random-init 기준선 1. 서 있는 상태 전용 probe 행 13개 |
| 소요 시간 | 데이터 재생성 292.8초, 37행 인코딩·probe(GPU 프로세스 4개) 3,120.0초, 서 있는 상태 전용 probe 250.7초 |
| 평가 | 로컬 CPU에서 `code/exp2_evaluate.py`, 표는 `code/exp2_tables.py`. 평가 입력의 split·targets·torso_height·index는 Experiment 1 Kaggle 커널 출력이며 sha256이 PREREG·`results/exp1/kaggle/sha256_kaggle.txt`와 같다 |

- 표 전체: [results/exp2/eval/tables.md](results/exp2/eval/tables.md). 모든 칸: [cells.csv](results/exp2/eval/cells.csv). 요약 수치: [summary.json](results/exp2/eval/summary.json).
- 그림: [linear pooled](results/exp2/eval/heatmap_pooled_linear.png), [MLP pooled](results/exp2/eval/heatmap_pooled_mlp.png), [linear fallen 제외](results/exp2/eval/heatmap_not_fallen_linear.png), [MLP fallen 제외](results/exp2/eval/heatmap_not_fallen_mlp.png), [λ 확장](results/exp2/eval/heatmap_pooled_linear_lambda_ext.png), [서 있는 상태 전용](results/exp2/eval/heatmap_standing_only_linear.png), [d_TV 대 R² 산점도](results/exp2/eval/scatter_tv_vs_r2.png).
- 예측값(`rows/*.npz`, 약 300 MB)은 PREREG 8절대로 Kaggle 출력과 로컬 `kaggle_out/exp2/`에만 두고 GitHub에는 올리지 않았다. 행별 메타데이터(λ, epoch, d_TV 분위수)는 `results/exp2/kaggle/rows/*.json`에 있다.

## 2. 결과 구분별 상태 수

| 구분 | 상태 | 에피소드 | 열별 target 분산 (stand, walk, run, walk-bwd, run-bwd) |
|---|---|---|---|
| pooled test | 30,000 | 60 | 0.1673, 0.1781, 0.0362, 0.0961, 0.0079 |
| fallen 제외 test | 20,395 | 60 | 0.0077, 0.1573, 0.0290, 0.1089, 0.0026 |

## 3. 판정 결과: "실질적으로 낮음" 칸 수 (열별)

판정: Δ = R²(대각선) − R²(행)의 95% paired bootstrap 구간 하한 > 0 이고 Δ ≥ 0.02 (PREREG 6절).

| 결과 구분 | 행 묶음 (행 수) | stand | walk | run | walk-bwd | run-bwd |
|---|---|---|---|---|---|---|
| pooled, linear (주) | 비-walker task (25) | 0 | 25 | 1 | 25 | 11 |
| | 영벡터 (1) | 0 | 1 | 0 | 1 | 0 |
| | 무작위 벡터 (5) | 0 | 5 | 0 | 5 | 5 |
| pooled, MLP (주) | 비-walker task (25) | 0 | 0 | 0 | 0 | 0 |
| | 영벡터 (1) | 0 | 0 | 0 | 0 | 0 |
| | 무작위 벡터 (5) | 0 | 0 | 0 | 0 | 0 |
| fallen 제외, linear | 비-walker task (25) | 25 | 25 | 5 | 25 | 25 |
| | 영벡터 (1) | 1 | 1 | 0 | 1 | 1 |
| | 무작위 벡터 (5) | 5 | 5 | 0 | 5 | 5 |
| fallen 제외, MLP | 비-walker task (25) | 16 | 0 | 0 | 0 | 4 |
| | 영벡터 (1) | 1 | 0 | 0 | 0 | 0 |
| | 무작위 벡터 (5) | 5 | 0 | 0 | 0 | 0 |
| pooled, linear λ 확장 | 비-walker task (25) | 0 | 25 | 1 | 25 | 11 |
| | 영벡터 (1) | 0 | 1 | 0 | 1 | 0 |
| | 무작위 벡터 (5) | 0 | 5 | 0 | 5 | 5 |
| 서 있는 상태 전용, linear | 비-walker task (3) | 3 | 3 | 0 | 3 | 3 |

칸 분류 (linear, MLP)와 보조 기준(RMSE 비율, random-init 미만) 개수는 [tables.md](results/exp2/eval/tables.md)에 있다.

## 4. 수치로만 적는 사실

### 대각선과 비-walker 25행의 R² 범위 (점추정)

| 결과 구분 | 열 | 대각선 | 비-walker 최소 | 비-walker 최대 |
|---|---|---|---|---|
| pooled, linear | stand | 0.9992 | 0.9804 | 0.9940 |
| | walk | 0.9761 | 0.9094 | 0.9558 |
| | run | 0.9981 | 0.9781 | 0.9928 |
| | walk-bwd | 0.9718 | 0.8418 | 0.9362 |
| | run-bwd | 0.9931 | 0.9458 | 0.9778 |
| pooled, MLP | stand | 0.9999 | 0.9857 | 0.9995 |
| | walk | 0.9998 | 0.9970 | 0.9993 |
| | run | 0.9999 | 0.9975 | 0.9997 |
| | walk-bwd | 0.9998 | 0.9962 | 0.9993 |
| | run-bwd | 0.9998 | 0.9907 | 0.9993 |
| fallen 제외, linear | stand | 0.9813 | 0.5360 | 0.8885 |
| | walk | 0.9644 | 0.8689 | 0.9336 |
| | run | 0.9972 | 0.9723 | 0.9910 |
| | walk-bwd | 0.9690 | 0.8242 | 0.9284 |
| | run-bwd | 0.9764 | 0.7993 | 0.9229 |
| fallen 제외, MLP | stand | 0.9982 | 0.6729 | 0.9872 |
| | walk | 0.9997 | 0.9961 | 0.9991 |
| | run | 0.9999 | 0.9964 | 0.9995 |
| | walk-bwd | 0.9998 | 0.9964 | 0.9992 |
| | run-bwd | 0.9993 | 0.9694 | 0.9974 |

- 네 구분 모두에서 비-walker 행이 대각선보다 높은 칸(Δ 구간 상한 < 0)은 없다. (사후 집계, PREREG 판정 항목 아님)
- pooled MLP에서 비-walker 25행의 모든 칸(125칸)은 RMSE 비율 ≥ 1.5다. 같은 칸들의 "실질적으로 낮음" 판정은 0칸이다. Δ 점추정이 0.02 미만이기 때문이다.

### random-init 기준선 미만 (MLP, 점추정)

- pooled MLP: walker-stand 열의 cheetah 계열 5행(cheetah-run, cheetah-run-backwards, cheetah-run-front, cheetah-run-back, cheetah-jump), walker-run-backwards 열의 cheetah-run-backwards. 6칸.
- fallen 제외 MLP: walker-stand 열의 같은 cheetah 계열 5행. 5칸.
- linear는 모든 구분에서 0칸이다 (참고용).
- random-init 기준선의 pooled linear R²(stand 열 0.7526)는 Experiment 1 random-init linear 값(0.753)과 같다. 기준선의 pooled MLP R²는 0.9913~0.9953이다 (multi-output).

### 판정이 바뀐 칸 (linear pooled 주 결과 대비)

- λ 확장: 0칸.
- 서 있는 상태 전용 probe: 4칸. cheetah-run → stand(낮음 아님 → 낮음), cheetah-run → run(낮음 → 낮음 아님), cheetah-run-back → stand(낮음 아님 → 낮음), cheetah-run-backwards → stand(낮음 아님 → 낮음).
- 서 있는 상태 전용 probe의 비-walker 행: PREREG 4절 규칙에 따라 cheetah-run-backwards(0.9313), cheetah-run-back(0.9330), cheetah-run(0.9341)이 골라졌다 (linear pooled 5열 평균 R²). 상태 수 train 61,244 / validation 20,414 / test 20,395.

### d_TV와 R² (서술용, 판정 없음)

x = 행의 5열 "대각선 z와의 d_TV 중앙값" 평균, y = 그 행의 5열 평균 pooled R².

| probe | 행 집합 | n | Pearson | Spearman |
|---|---|---|---|---|
| linear | 36행 전체 | 36 | −0.725 | −0.559 |
| linear | 비-walker 25행 | 25 | −0.397 | −0.384 |
| MLP | 36행 전체 | 36 | −0.438 | −0.586 |
| MLP | 비-walker 25행 | 25 | −0.433 | −0.455 |

- walker 5행의 x: 0.0875~0.1206. 비-walker·영벡터·무작위 행의 x는 [tables.md](results/exp2/eval/tables.md) 마지막 표에 있다.

### probe 학습 기록

| 항목 | 값 |
|---|---|
| ridge λ (주 결과, 185칸) | 후보 최솟값 1e-4에서 선택된 칸 39개, 최댓값 1e4는 0개 |
| ridge λ (확장, 185칸) | 확장 최솟값 1e-8에서 선택된 칸 11개 |
| ridge λ (서 있는 상태 전용) | 최솟값 1e-4에서 선택된 칸 9개 |
| MLP epoch (185 probe) | 최소 53, 중앙값 114, 최대 200. 200 epoch에 도달한 probe 3개(random-init 기준선, acrobot-swingup, cheetah-run-front 각 1개) |
| 상수 차원 | 37행 모두 0개 |

## 5. PREREG 대비 추가·변경

| 항목 | 내용 | 기록 시점 |
|---|---|---|
| 결과 회수 시점 | Kaggle 커널은 2026-10-27에 끝났지만, 작업 세션이 끊겨 결과를 2026-10-28에 받아 평가했다 | 이 문서 |
| 평가 환경 | 로컬 Windows Python 3.12.2, numpy 2.5.2 (`requirements.txt` 고정값 numpy 1.24.4와 다름). 평가 코드는 numpy만 쓰며, bootstrap 표본은 `numpy.random.default_rng(20261027)`로 만든다. 이전 Experiment 1 평가 환경과 같은 버전인지는 확인하지 않았다 | 평가 실행 시 |
| 평가 코드 수정 | `exp2_evaluate.py`의 λ 경계 개수 집계가 `l == 1e-4`로 비교해, 격자 값 `9.999999999999999e-05`를 세지 못했다(0개로 출력). 상대 허용 오차 비교로 고쳤다. R²·구간·판정(`cells.csv`)은 수정 전후 바이트 단위로 같다 | 평가 실행 후, 결과 문서 작성 전 |
| 표 생성 코드 | `code/exp2_tables.py` 추가 (표만 만들며 수치를 새로 계산하지 않는다) | 평가 실행 후 |

PREREG의 설정(행, 주입 함수, λ 후보, MLP 하이퍼파라미터, 표준화, 분할, 통계 기준, seed, 행 선택 규칙)은 바꾸지 않았다.

## 6. 실패 및 누락

- 누락된 행, probe, 칸은 없다 (37행 × 2 파일, 서 있는 상태 전용 1개).
- bootstrap에서 SST < 1e-10으로 버린 표본은 모든 칸에서 0개다 (`cells.csv`의 `boot_dropped`, `diff_boot_dropped`).

## 7. 이 결과로 결론 낼 수 없는 것

- **학습 분포 밖 입력**: 비-walker embedding, 영벡터, 무작위 벡터와 walker 관측의 조합은 학습 때 없던 입력이다. 이 칸의 R²가 학습된 계산을 반영하는지 알 수 없다.
- **정보의 존재·부재**: probe R²는 그 probe가 꺼낼 수 있는 정도(information accessibility)다. "실질적으로 낮음"이 정보가 없다는 뜻이 아니고, 높은 R²가 world model이 그 정보를 쓴다는 뜻도 아니다.
- **decision sufficiency, task 수행**: reward 예측 R²는 그 embedding으로 walker task를 수행하는 능력과 다르다. 이 실험은 planning을 하지 않았다.
- **linear와 MLP 판정 차이의 원인**: pooled에서 linear는 낮음 판정이 많고 MLP는 0칸이다. 이 차이가 무엇 때문인지 이 실험으로 정할 수 없다. MLP 칸은 R² 0.99 근처라 Δ 기준(0.02)에 닿기 어려운 범위다.
- **d_TV와 R²의 관계**: 상관 수치는 행 36개(또는 25개)의 기술 통계다. 판정 기준이 없고 인과 관계를 말할 수 없다.
- **행 방향 비교**: 같은 행 안 다른 열의 R² 차이는 target 분산 차이를 반영할 수 있다 (pooled 열별 분산 0.0079~0.1781).
- **task-independent shared encoder bottleneck**: 이 encoder는 task-conditioned다. task-independent encoder에 대해서는 아무것도 말할 수 없다.
- **일반화**: checkpoint 하나(mt30-48M), walker 몸체 하나, Experiment 1 상태 분포(A·B·C 혼합)에 대한 결과다.

## 8. 재현

```bash
# Kaggle (T4 x2): kaggle/exp2_body.py, 저장소 커밋 4f02317
python kaggle/build.py 4f02317 exp2 && kaggle kernels push -p kaggle/kernel_exp2
kaggle kernels output seungwonryoo/tdmpc2-probe-exp2 -p kaggle_out/exp2
kaggle kernels output seungwonryoo/tdmpc2-probe-exp1-probe -p kaggle_out/exp1_probe \
    --file-pattern '(index\.npz|split_labels\.npy|targets\.npy|torso_height\.npy|split\.json)$'
# data_eval_kaggle/{split_labels.npy, split.json, pooled/{index.npz, targets.npy, torso_height.npy}}에 복사
cd code
python exp2_evaluate.py --data ../data_eval_kaggle --rows ../kaggle_out/exp2/out/rows \
    --emb ../results/exp2/kaggle/inject_check_kaggle.json --out ../results/exp2/eval
python exp2_tables.py --eval ../results/exp2/eval --emb ../results/exp2/kaggle/inject_check_kaggle.json
```
