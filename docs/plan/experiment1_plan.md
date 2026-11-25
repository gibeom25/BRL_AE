# Experiment 1 — AE의 $h$/$O$ 활용과 Perturbation 강인성 (LIBERO-plus)

## 1. 목적과 가설

| 번호 | 가설 | 검증 방법 |
|---|---|---|
| H1 | BC에서 AE는 관측 $O$ 대신 VLM 출력 $h$를 조건으로 써도 학습된다 | `M1_h` clean 성공률 |
| H2 | $h$와 $O$를 같이 쓰면 perturbation(특히 Camera)에 더 강인하다 | `M3_hO` vs `M1_h`, Camera 성공률 |
| H3 | $h$+$O$를 단순 결합하면 AE가 $h$를 무시하는 shortcut이 생긴다 | 오프라인 의존도, $h$ 교체 rollout |
| H4 | MoE는 라우터가 같은 손실로 학습돼 $O$ 쪽으로 쏠린다 | `M4_moe`의 $g$ 추이 |
| H5 | generator($h$)–verifier($O$) 분리는 shortcut 없이 $O$를 활용한다 | `M5_verify` vs `M1_h` |

Camera를 메인 교란으로 쓰는 이유: 카메라 시점 변화는 정면 카메라(→ $h$)만 바꾸고
손목 카메라($O$)는 바꾸지 않는다. 따라서 $O$ 경로의 기여를 가장 깨끗하게 볼 수 있다.

## 2. 설정

### 데이터
- LIBERO-Spatial: 태스크 10개 × 시연 50개 (약 6만 프레임)
- 분할: 태스크별 시연 45개 학습 / 5개 검증 (검증 50개는 오프라인 평가용)
- 입력: 정면 이미지(→ VLM), 손목 이미지, proprio (EEF pos/ori + gripper, 실제 키는 데이터에서 확인)
- 출력: 7차원 action (delta EEF 6 + gripper 1)

### 고정 VLM과 $h$
- 모델: SmolVLM2-500M 계열 (HF 모델 이름은 Phase 0에서 확인)
- 레이어: 마지막 hidden state (설정 `h_layer`로 변경 가능)
- `h_tokens` = 이미지 토큰 adaptive avg pool 12개 + 지시문 이후 텍스트 토큰 pool 4개 = (16, d_vlm), fp16
- 추가 저장: 태스크별 평균 $\bar h$ (h 교체 실험용)

### AE 공통 구조
- prefix 토큰: [`h_tokens` → Linear] (16) + [손목 ResNet18 특징 → Linear] (1) + [proprio → MLP] (1)
- 뒤쪽 토큰: noisy action $H$개 + $\tau$ 임베딩(각 action 토큰에 더함)
- 6층 transformer, d=384, head 6, 약 15~20M 파라미터
- 출력: action 토큰 위치의 velocity $v$

| 하이퍼파라미터 | 값 |
|---|---|
| chunk $H$ / 실행 step | 10 / 8 |
| batch / lr / optimizer | 256 / 1e-4 / AdamW (wd 1e-4), cosine, warmup 1k |
| train step | 30k (smoke: 50) |
| 추론 | Euler 10 step |
| 손목 이미지 증강 | random crop(90%) + color jitter 약하게 |

### 모델별 차이
| ID | prefix에 들어가는 것 | 비고 |
|---|---|---|
| `M1_h` | $h$ | |
| `M2_O` | $O$ | 선택 |
| `M3_hO` | $h$ + $O$ | |
| `M4_moe` | AE$_h$(prefix $h$), AE$_O$(prefix $O$) 두 개 | 각 AE는 3층으로 줄여 총 크기 유사하게. 라우터 MLP 입력 $[\bar h_{pool}, f(O), \tau]$, 출력 $g\in(0,1)$. 500 step마다 $g$ 평균·분포 기록 |
| `M5_verify` | generator = `M1_h` 체크포인트 | verifier는 별도 학습 |

### Verifier (`M5_verify`)
- 입력: 손목 특징(ResNet18, 학습) + proprio + action chunk(flatten) → MLP → logit
- 데이터 (1:1:1:1):
  - 양성: 시연 $(O, A)$
  - 음성 1: `M1_h`가 같은 프레임에서 생성한 chunk
  - 음성 2: 시연 chunk + 가우시안 노이즈 (정규화 공간 σ=0.1~0.3 랜덤)
  - 음성 3: 같은 궤적에서 ±10 step 떨어진 chunk
- BCE, 5k step. 검증 세트에서 양성/음성 분류 정확도 기록
- 추론: `M1_h`에서 후보 $N=8$ → 최고 점수 실행

## 3. 평가

### 3.1 오프라인 (검증 세트, action MSE)
| 조건 | 설명 | 대상 |
|---|---|---|
| normal | 그대로 | 전체 |
| h_zero | `h_tokens`=0 | M1, M3, M4 |
| h_swap | 다른 태스크의 $\bar h$로 교체 | M1, M3, M4 |
| O_zero | 손목·proprio 토큰=0 | M3, M4 |

의존도 $=\dfrac{\text{MSE(조건)} - \text{MSE(normal)}}{\text{MSE(normal)}}$.
`M3_hO`에서 h_swap 의존도 ≪ O_zero 의존도면 H3을 지지한다.
샘플링 노이즈 영향을 줄이기 위해 같은 seed의 $\epsilon$을 조건 간에 공유한다.

### 3.2 Rollout (LIBERO-plus)
- 규모: 태스크 10개 × 10 episode = n=100 (모델·조건당), 환경 프로세스 4~6개 병렬
- 실행 순서 (위가 우선):

| # | 모델 | 조건 |
|---|---|---|
| 1 | M1, M3 | clean |
| 2 | M1, M3 | Camera |
| 3 | M3 | clean + h_swap (온라인) |
| 4 | M4 | clean, Camera |
| 5 | M5 | clean, Camera |
| 6 | 전체 | Light 또는 Noise |

### 3.3 결과 JSON 스키마
```json
{
  "model": "M3_hO", "perturb": "camera", "h_mode": "normal",
  "n": 100, "success": 47, "success_rate": 0.47,
  "per_task": {"0": [5, 10], "1": [3, 10]},
  "ckpt": "outputs/experiment1/ckpt/M3_hO/step30000.pt",
  "seed": 0, "timestamp": "..."
}
```
경로: `outputs/experiment1/rollouts/{model}__{perturb}__{h_mode}.json`
오프라인: `outputs/experiment1/offline/{model}.json`

## 4. 진행 순서와 시간표 (20:00 시작, 08:00 마감)

| 시간 | Phase | GPU 작업 | 완료 기준 |
|---|---|---|---|
| 20:00–20:40 | 0. 환경 점검 | torch cu128, LIBERO/LIBERO-plus, VLM 로딩 | env reset + 랜덤 10 step + 이미지 저장 |
| 20:40–21:30 | 1. $h$ 캐싱 / AE 코드 | `01_cache_h.py` | shape·NaN·태스크 간 코사인 유사도 출력 |
| 21:30–22:00 | 2. 스모크 테스트 | M1 500 step + rollout 1 ep | **rollout 1 episode가 끝까지 실행**, 1 ep 소요 시간 측정 |
| 22:00–23:30 | 3. 학습 1차 | M1 + M3 동시 | 검증 MSE 기록, 체크포인트 |
| 23:30–01:00 | 4. 학습 2차 | M4 학습 + verifier 데이터 생성·학습 | $g$ 로그, verifier 정확도 |
| 23:30–01:00 | 4'. 오프라인 평가 | M1, M3 의존도 | `offline/*.json` |
| 01:00–04:30 | 5. Rollout | 표 3.2 순서 | `rollouts/*.json` |
| 04:30–06:30 | 6. 정리 | (선택) M2 학습 | 표 2개, 그래프 3개 |
| 06:30–08:00 | 7. 발표 자료 | — | |

**중단 규칙**
- 22:30까지 rollout이 안 되면: 학습은 계속하고 rollout 디버깅 병행. 최악의 경우 오프라인 결과로 발표.
- 시간이 부족하면 살리는 순서: (1) M1·M3 + 오프라인 의존도 → (2) M4 $g$ 그래프 → (3) M5 → (4) 보조 perturbation.

## 5. 산출물 (발표용)

- 표 1: 성공률 (M1/M3/M4/M5 × clean/Camera/보조), n 표기
- 표 2: 오프라인 의존도 (M1/M3/M4 × h_zero/h_swap/O_zero)
- 그림 1: 조건별 성공률 막대 그래프
- 그림 2: M4 라우터 $g$ 학습 곡선 (평균 + 10/90 분위)
- 그림 3: $h$/$O$ 의존도 막대 그래프

## 6. 결과 해석 시나리오

| 관찰 | 해석 |
|---|---|
| M3 > M1 (Camera), M3의 $h$ 의존도 낮음 | 강인성은 오르나 $h$를 무시 → shortcut 문제 제기 (계획대로) |
| M3의 $h$ 의존도도 높음 | 이 설정에서는 shortcut이 약함 → 태스크·장면 다양성이 커질 때의 문제로 가설 조정 |
| M4의 $g$가 쏠리지 않음 | 쏠림 미관찰, 성능 이득 여부만 보고 |
| M5 < M1 | verifier 음성 샘플 설계 문제로 분석, 개선 방향 제시 |

## 7. 관련 연구 (발표 슬라이드용)

- $h$+$O$ 구조: HiRT (CoRL 2024), FiS-VLA (NeurIPS 2025) — 동기는 주로 추론 속도. 강인성·shortcut 분석은 공백.
- generate-then-verify: V-GPS (CoRL 2024), RoboMonkey (CoRL 2025) — verifier는 정책과 같은 관측을 보고 성능 향상이 목적.
  우리는 verifier가 generator가 보지 않는 $O$를 전담하고, 목적이 shortcut 방지·모달리티 역할 분리.
- Action expert MoE: DiTEA (AAAI) — 태스크 간 충돌 해결 목적. 모달리티 분리 목적의 MoE는 확인되지 않음.
- shortcut 배경: Causal Confusion in Imitation Learning (NeurIPS 2019), Fighting Copycat Agents (NeurIPS 2020) — 서지 정보 재확인 필요.
