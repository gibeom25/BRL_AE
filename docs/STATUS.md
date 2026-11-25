# STATUS (자동 진행, 2026-11-25)

마지막 갱신: 21:52 KST — 자동 체인 전체 종료 (M1~M4 학습·오프라인·rollout 1~4단계 완료). 남은 것: verifier(M5), 보조 perturbation(6단계), 집계·그래프

## 단계별 상태

| # | 단계 | 상태 | 비고 / 로그 |
|---|---|---|---|
| 1 | 설치 (env `brl_ae`, torch, LIBERO-plus, transformers, VLM) | 완료 | torch 2.11.0+cu128, cap (12,0), 매 설치 후 확인 OK. 추가 해결: libero import(.pth), termcolor, **mujoco 3.17 비호환 → 2.3.7**, wand/libjpeg import 순서 |
| 2 | 다운로드 (assets 6.0GB, libero_spatial 5.9GB) | 완료 (20:00) | assets는 zip 내부 경로 때문에 `libero/libero/assets` 심볼릭 링크 |
| 3 | 00_env_check.py (1~4) | 완료 | `outputs/experiment1/env_check/env_check.json` |
| 3b | perturbation 정량표 + 그리드 | 완료 | 아래 표, `outputs/experiment1/env_check/perturb_matrix.{json,png}` |
| 4 | P2 데이터 확인 | 완료 | 62,250 frames, split 450/50 demo, norm_stats(train만). `outputs/experiment1/data_check/` |
| 5 | P3 h 캐싱 | 완료 (20:28, 7.9분) | 62,250×16×960 fp16, NaN 없음. h_bar 코사인 off-diag 0.937~0.985 (평균 0.967) → 모든 쌍 0.99 이상 아님, 통과. 평균 중심화 후 −0.50~0.37 |
| 6 | P4 AE 학습 코드 | 완료 (smoke) | 4개 모델 smoke 통과 |
| 7 | P7 오프라인 평가 | **완료** (M1~M4) | 아래 표. `outputs/experiment1/offline/{M1_h,M2_O,M3_hO,M4_moe}.json` |
| 8 | `06_rollout.py` | 완료 | perturbation 채움, M5_verify만 TODO |
| 9 | P5 스모크 (M1_h 500 step, 실제 캐시) | **통과** | val MSE 0.302→0.265. rollout clean 1 ep·camera 1 ep 모두 220 step 끝까지 실행, JSON 생성, 실패 영상 저장 (`outputs/experiment1/rollouts_p5/`). episode 본체 ~4.3s, 병렬 5 workers로 10 ep 39s (VLM 로딩 포함) |
| 10 | P6 학습 M1_h + M3_hO 30k | **완료** | M1_h 20:47 (val MSE 0.0900, 18k 이후 정체), M3_hO 20:56 (val MSE 0.0630). |
| 10b | M2_O(필수) + M4_moe 30k 동시 학습 | **완료** (~21:25) | M2_O val MSE 0.0626 (28.9분), M4_moe val MSE 0.0624 (29.6분). M4 g: 0.5 → 0.94(500 step) → 0.44(30k), rollout 중 평균 ~0.30 |
| 10c | 자동 체인 `experiment1/run/phase4.sh` | **완료** (21:50) | 로그 `outputs/setup_logs/phase4_chain.log` |
| 11 | phase5 rollout 1~4단계 (n=200씩) | **완료** (21:50) | 아래 성공률 표. 5단계(M5)·6단계(보조 perturbation) 미실행. 로그 `outputs/experiment1/rollouts/run_logs/phase5.log` |
| 12 | `09_probe_h.py` (CPU, nice 19) | **완료** (20:59) | meanpool R² 0.955, concat16 R² 0.985, 셔플 대조군 ≈0. 로그 `outputs/setup_logs/probe_h.log` → `outputs/experiment1/offline/probe_h.json` |

## P5 스모크 결과 요약
- M1_h 500 step (실제 캐시): val MSE 0.302 → 0.265.
- rollout task 0: **clean 1 ep 실패 (220 step 완주)**, **camera 1 ep 실패 (220 step 완주, variant `_view_13_15_100_0_0`)**. 500 step 모델이라 실패는 예상 범위. JSON 생성 OK, 실패 영상 저장 OK (`outputs/experiment1/rollouts_p5/`). episode 본체 ~4.3s.
- 캐시 h vs 온라인 h (같은 함수 `vlm_features.py`): 같은 hdf5 프레임 cos **0.9999**, 같은 sim state의 env 렌더 cos **0.984** (아래 표).

## 오프라인 의존도 (P7, val n=6163, normalized action MSE, 같은 ε 공유)
| 모델 | normal | h_zero | h_swap | O_zero | 의존도 h_zero / h_swap / O_zero |
|---|---|---|---|---|---|
| M1_h | 0.0770 | 0.3444 | 0.4684 | – | +3.47 / +5.08 / – |
| M3_hO | 0.0601 | 0.0740 | 0.0652 | 0.3641 | +0.23 / **+0.08** / **+5.06** |
| M2_O | 0.0616 | – | – | 0.3797 | – / – / +5.16 |
| M4_moe | 0.0608 | 0.0588 | 0.0615 | 0.3657 | −0.03 / +0.01 / +5.02 |
→ 오프라인 MSE 기준으로 M3·M4는 h를 거의 쓰지 않음(h_swap +8%, +1%), O 제거 시 +500% 이상.

## Rollout 성공률 (LIBERO-Spatial, 10 tasks × 20 ep = n=200, 30k step ckpt)
| 모델 | clean | Camera | clean + h_swap(온라인) |
|---|---|---|---|
| M1_h | 106/200 = **0.530** | 1/200 = **0.005** | N/A (미실행, 계획에 없음) |
| M2_O | 127/200 = **0.635** | 127/200 = **0.635** | – |
| M3_hO | 184/200 = **0.920** | 183/200 = **0.915** | 79/200 = **0.395** |
| M4_moe | 139/200 = **0.695** | 123/200 = **0.615** | N/A (미실행) |
| M5_verify | N/A (미실행) | N/A (미실행) | – |
- M2_O clean과 Camera는 200개 episode가 성공·step 수까지 완전히 같음 → Camera가 손목 이미지를 바꾸지 않는다는 것과 일치(정상성 확인).
- **해석 메모 (예상과 다른 점)**: 오프라인에서는 M3가 h를 거의 무시(h_swap +8%)했지만, rollout에서 h를 다른 태스크로 바꾸면 성공률 0.920 → 0.395로 크게 떨어짐. 즉 M3는 h를 "어느 bowl을 집을지" 같은 태스크 식별에 쓰고, 프레임별 action 오차(MSE)는 대부분 O로 결정되는 것으로 보임 → 오프라인 MSE 의존도는 shortcut을 과대평가할 수 있음. 또 M3(0.920) > M2(0.635)이므로 h는 실제로 기여함.
- Camera: M1은 붕괴(0.530→0.005), M3는 유지(0.915). h+O가 Camera 교란에 강인하다는 H2와 일치.
- M4_moe는 M3보다 낮음(0.695/0.615). router g는 학습 후반 ~0.44, rollout 중 ~0.30으로 O 쪽에 기울었으나 극단적이지 않음.

## h 선형 probe (09_probe_h.py) — 정답 EE 위치
- 물체 위치 **미포함**. 이유: hdf5에 물체 위치 키가 없고, 92차원 MuJoCo `states`에서 꺼내려면 시뮬레이터 모델(joint qpos 주소)로 state 복원이 필요.
- 랜덤 초기화 VLM 대조군: GPU가 학습 2개 + rollout으로 사용 중이라 **미실행** (선택 항목).
- ridge, alpha는 val로 선택(요청대로; 약간 낙관적이며 셔플 대조군도 같은 절차).

| 입력 | 차원 | val R² (x,y,z 평균) | 정답 셔플 대조군 | 태스크 내 R² |
|---|---|---|---|---|
| mean-pool | 960 | 0.955 (0.937/0.980/0.948) | −0.003 | 0.919 |
| concat16 | 15360 | 0.985 (0.976/0.995/0.983) | −0.007 | 0.972 |

## 캐시 h vs 온라인 h (함정 4 확인)
온라인 h도 같은 `common/models/vlm_features.py`의 `VLMFeatureExtractor.encode`로 계산. 9 프레임(3 tasks) 평균:

| 비교 | 상대 L2 | 코사인 |
|---|---|---|
| 같은 hdf5 프레임: 캐시(batch 64) vs 온라인(batch 1) | 1.4% | 0.9999 |
| 같은 sim state: 캐시 vs **env 렌더** 온라인 (실제 rollout 경로) | 18.0% | 0.984 |
| (참고) 같은 demo t vs t+8 | 18.6% | 0.984 |
| (참고) 다른 태스크 같은 t | 35.8% | 0.936 |

## 발표용 표: perturbation별로 바뀌는 카메라 (실측)
같은 task·같은 초기 상태(pruned_init[0])·같은 1 step 후, clean 대비 평균 절대 픽셀 차이(0–255). 3 tasks (0,4,8) × 5 variants = n=15.

| perturbation | 정면 (→ h) MAD | 손목 (→ O) MAD | 손목 완전 동일 비율 | n |
|---|---|---|---|---|
| Camera | 33.1 ± 6.6 | 0.0 ± 0.0 | 100% | 15 |
| Light | 30.7 ± 15.6 | 36.2 ± 14.9 | 0% | 15 |
| Noise | 22.3 ± 18.5 | 0.0 ± 0.0 | 100% | 15 |

→ Camera·Noise는 h 경로만 교란, Light는 h와 O 모두 교란. 그리드: `outputs/experiment1/env_check/perturb_matrix.png`

## 모델 크기 차이 (발표 질문 대비)
M1_h 11.33M vs M3_hO 22.85M의 차이 11.52M = 손목 ResNet18 backbone 11.18M + wrist proj 0.20M + proprio MLP 0.15M. transformer(10.65M)·action 입출력(0.31M)은 동일. M4_moe 23.90M(ae_h 6.00 + ae_O 17.16 + router 0.74)로 M3와 비슷. (고정 VLM 507M은 모두 미포함)

## 직접 해야 할 일
- CLAUDE.md는 보호 파일이라 수정하지 않음 → §3에 "conda 환경: `brl_ae`", "mujoco==2.3.7 고정(3.x는 robosuite 1.4.0과 비호환)"을 직접 추가 권장.
- (sudo, 선택) `sudo apt install libmagickwand-dev` — 지금은 conda-forge imagemagick으로 동작 중이라 필요 없음.

## 내려야 할 결정
1. **[중요] 학습 h vs rollout h 분포 차이**: 같은 sim state에서 env 렌더와 hdf5 이미지가 픽셀 MAD ~2–3.6 차이 → h 상대오차 18%. 8 step 시간 변화와 같은 크기, 다른 태스크 차이의 절반. 함수 불일치는 아님(같은 프레임이면 1.4%).
   - A: 그대로 진행 (현재). B: 학습 데이터 정면 이미지를 현재 env로 재렌더링해 h 재캐싱(렌더 수 분 + 캐싱 8분 + 재학습). 손목 이미지도 비슷한 차이(MAD ~1–2) 있음.
   - 근거: `outputs/experiment1/cache/h_online_check.json`
2. obs–action 시점: 데이터의 obs[j]는 action j **이후** 관측(P2에서 확인). 현재 LIBERO 관례대로 그대로 사용(`obs_action_shift: 0`). 1이면 의미상 정확한 쌍, 재학습 필요.
3. 보조 perturbation: Light(h+O 모두 교란, 기본값) vs Noise(h만 교란, Camera와 같은 성격). `AUX=noise`로 변경 가능.
4. Camera variant 수가 태스크 0은 6개뿐(light는 태스크 4가 4개) → 20 ep/task면 variant 반복 사용(초기 상태는 매번 다름).

## 다음 프롬프트
- 다음: (1) verifier 구현(03_gen_negatives → 04_train_verifier → M5 rollout), (2) phase5 6단계(보조 perturbation, `ONLY=6`), (3) 07_aggregate·08_plots로 표·그래프 생성, (4) 결정 1(렌더 차이) — M1 clean 0.53이 낮은 원인 후보.
