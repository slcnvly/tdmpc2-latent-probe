# VERIFICATION_PLAN (V0: 판정 기준, 실행 전 기록)

작성: 2026-10-25, V1~V5 실행 전. 이 파일의 기준은 결과를 본 뒤 바꾸지 않는다.
바꿔야 하면 VERIFICATION_RESULTS.md에 이유와 함께 적고 사용자에게 알린다.

## 용어 (V4에서 고정, 모든 문서·코드에서 사용)

- **s_t**: 관측 o_t를 계산한 물리 상태 (`physics.get_state()`: time, qpos, qvel, act).
- **r_logged_t**: `env.step(a_t)`가 돌려준 reward. action repeat 2회 동안의 substep reward 합.
- **r_τ(s_t)**: s_t를 복원하고 시간 적분 없이(`set_state` 후 kinematics만 갱신하는 `forward()`) task τ의 `get_reward(physics)`를 적용한 값. probe target이다.

## 실행 구성 (기준 적용 대상)

- **공식 경로 (official run)**: `tdmpc2/tdmpc2/evaluate.py`를 수정 없이 실행한다.
  `task=mt30 model_size=48 checkpoint=<mt30-48M.pt> eval_episodes=1 seed=1 save_video=false`.
  - `save_video=false`는 config 인자다. 영상 렌더링을 끄는 것뿐이고 계산 경로는 같다.
  - 실행은 `runpy.run_path`로 한다. 코드는 바꾸지 않고, 기록용 hook만 붙인다:
    - `TDMPC2.act`의 입력·출력
    - encoder 입력 (`_encoder['state']`의 forward pre-hook)
    - 각 task 첫 act 직전의 torch CPU·CUDA 난수 상태
  - hook은 값을 복사만 하고 계산에 개입하지 않는다.
- **자체 경로 (own run)**: `code/eval_single_task.py` (V1에서 공식과 다른 점을 고친 뒤의 버전).
- 이 공식 실행 1회가 V2(trace)와 V3(30-task 점수)에 함께 쓰인다. 30-task 평가는 이 1회뿐이다.

## V1 (코드 대조) 기준

- 항목마다 공식 경로와 자체 경로의 파일·줄 번호를 나란히 적는다.
- "같음" / "다름(수정함)" / "다름(수정 안 함, 이유)" 셋 중 하나로 판정한다.
- **task 번호 → embedding 행** 대응이 세 경로(TASK_SET 목록, wrapper 번호, checkpoint 표의 행) 중 하나라도 다르면 즉시 실패로 보고하고 V2 이후를 멈춘다.

## V2 (trace equivalence) 기준

- 비교 대상: walker-walk, walker-run, walker-run-backwards의 첫 에피소드 처음 50 step.
  step마다 환경 관측, encoder 입력 벡터 전체(관측 + task embedding), task 번호, 출력 행동을 비교한다.
- 난수 맞추기:
  - 공식 실행은 task 0, 1, 2(walker-stand, walker-walk, walker-run)를 순서대로 돈다. 자체 경로도 seed 1에서 같은 세 task를 같은 순서로 전 에피소드 돌려 난수 흐름을 맞춘다.
  - walker-run-backwards(task 20)는 공식 실행에서 그 task 첫 act 직전에 저장한 torch CPU·CUDA 난수 상태를 자체 경로에서 복원해 맞춘다.
  - 환경 초기 상태는 task마다 따로 있는 env 난수(`task_kwargs={'random': seed}`)가 정하므로, 같은 seed면 각 task 첫 에피소드에서 같다.
- 허용 오차와 판정:

| 판정 | 조건 |
|---|---|
| **통과** | 50 step 모두에서 관측과 encoder 입력의 최대 절대 차이 ≤ 1e-5, 행동의 최대 절대 차이 ≤ 1e-4, task 번호 일치 |
| **실패** | step 0(첫 act, 같은 초기 상태와 같은 난수 상태에서 한 번 계산)에서 관측·encoder 입력 차이 > 1e-5 또는 행동 차이 > 1e-4, 또는 task 번호 불일치 |
| **판정 불가 (비결정성)** | step 0은 통과 기준 안인데, 이후 step에서 허용 오차를 넘는 경우. 처음 넘는 step 번호를 보고한다 |

- 허용 오차의 근거: 관측과 행동은 float32다. 같은 GPU, 같은 버전, 같은 compile 설정의 같은 연산이면 결과가 비트 단위로 같거나 float32 반올림 수준(~1e-7)으로 같을 것으로 본다.
  여유를 두어 관측 1e-5, 행동 1e-4로 정했다.
  closed-loop rollout에서는 작은 수치 차이가 step을 거치며 커질 수 있다(MPPI의 top-k 선택이 바뀌면 행동이 크게 달라진다). 그래서 step 0의 일치를 경로 동일성의 직접 증거로 두고, 이후의 벌어짐은 "판정 불가"로 따로 둔다.
- walker-walk는 Experiment 0에서 참고값과 가까웠던 대조군이다. 세 task 판정은 따로 한다.

## V3 (공식 30-task 평가) 기준

- 공식 실행의 walker task 점수(1 에피소드 return)가 Experiment 0 자체 경로 5 에피소드의 최솟값~최댓값 범위 안이면 **일치**, 밖이면 **불일치**.
  - 비교 대상은 V1 수정 후 다시 돌린 Experiment 0이다. 수정이 없으면 기존 결과를 쓴다. 둘 다 있으면 둘 다 적는다.
  - 범위 끝값과 같으면 일치로 본다.
- 한계: 공식 값은 1 에피소드뿐이고 자체 값은 5 에피소드다. 범위 밖이 우연일 수 있고, 범위 안이 같은 분포를 뜻하지도 않는다.
- 30개 평균은 `evaluate.py:94, 99`의 정의(DMControl task는 return / 10, Meta-World는 success × 100의 task 평균)로 계산해 논문 값 59.4와 나란히 적는다.
  1 에피소드라 비교가 거칠다. 이 평균으로는 판정하지 않는다.
- 공식 경로와 자체 경로가 같은 seed·같은 난수 흐름이면 walker-stand, walker-walk, walker-run의 1 에피소드 return은 같은 값이 나와야 한다 (V2와 같은 근거).
  자체 경로 첫 세 에피소드와의 차이도 적는다. 이것은 V2의 보조 기록이고 V3 판정에는 쓰지 않는다.

## V4 (reward 시점) 기준

- **4-2 통과**: 각 walker task에서 마지막 substep 종료 상태(= s_{t+1})를 복원해 계산한 r_τ(s_{t+1})와, r_logged_t의 마지막 substep 성분의 최대 절대 차이 ≤ 1e-6.
- **4-3**: r_τ(s_t)와 r_logged_t의 값 예시를 표로 적는다 (판정 없음).
- **4-4 통과**:
  - 코드: `get_reward(physics)`가 action 인자를 받지 않는다.
  - 수치: 같은 상태에서 `physics.data.ctrl`만 바꾸고 `forward()`했을 때 r_τ(s)의 최대 절대 차이 ≤ 1e-9.

## V5

판정에 쓰지 않는 기술 기록이다. 기준 없음.

## 실패 시 진행 규칙

| 상황 | 처리 |
|---|---|
| V1 task 번호 대응 실패 | 멈추고 보고 |
| V1의 그 밖의 차이 | 자체 경로를 공식 쪽에 맞춰 고치고 계속 |
| V2 실패 | V3~V5는 계속 실행하고 기록한다. 둘은 같은 실행에서 나온다. GATE 2 보고에 "자체 경로가 공식 경로와 다름"으로 적는다 |
| V2 판정 불가 | 계속 |
| V3 불일치 | 계속 |
| V4 실패 | 계속하되, EXPERIMENT_PLAN.md target 정의 수정(V6)에 그 사실을 적는다 |
| 커널 실행 자체가 실패 | 고칠 수 있는 오류(설치, 경로)는 고쳐 다시 실행한다. 공식 30-task 실행이 끝까지 돌지 못한 경우는 1회로 세지 않는다 |
