# CLAUDE.md — Action Expert Shortcut & Robustness 연구

이 파일은 Claude Code가 이 저장소에서 작업할 때 항상 따르는 규칙과 맥락이다.
상세 실험 계획은 `docs/plan/experiment1_plan.md`를 먼저 읽는다.

---

## 1. 연구 개요 (한 문단)

VLA의 Action Expert(AE)는 BC로 학습되며, Diffusion Policy의 관측 조건 $O$ 자리를
VLM 출력 $h$가 대체하는 구조다. 우리는 AE가 $h$와 직접 관측 $O$를 **함께** 쓰면
LIBERO-plus perturbation에 더 강인해질 것이라고 본다. 단, 이때 AE가 $h$를 무시하고
$O$에만 의존하는 **shortcut**이 생길 수 있다. 이를 확인하고 막는 방법
(MoE → generator–verifier)을 비교한다. 향후(experiment2)에는 SSM 기반 물리 상태
인코더 $z_t$를 추가해 동역학 교란으로 확장한다.

## 2. 용어와 기호 (코드·문서에서 반드시 이 이름을 쓴다)

| 기호 | 의미 | 코드 이름 |
|---|---|---|
| $h$ | 고정 VLM이 (정면 이미지 + 지시문)에서 만든 토큰 | `h_tokens` |
| $O$ | 손목 카메라 특징 + 로봇 proprio | `obs_wrist`, `obs_proprio` |
| $A$ | action chunk, 길이 $H$ | `actions` (B, H, 7) |
| $\tau$ | FM 시간, 0=노이즈 → 1=정답 | `tau` |
| $g$ | MoE 라우터 가중치 ($h$-expert 비중) | `router_g` |
| $D(O,A)$ | verifier 점수 | `verifier_logit` |

모델 ID (결과 파일, 그래프, 표에서 동일하게 사용):

| ID | 구성 |
|---|---|
| `M1_h` | AE($h$) |
| `M2_O` | AE($O$) — 선택 |
| `M3_hO` | AE($h$+$O$), prefix concat |
| `M4_moe` | $g\,v_h + (1-g)\,v_O$, 라우터 입력 $[\bar h, f(O), \tau]$ |
| `M5_verify` | `M1_h` 후보 $N$개 → $D(O,A)$ 최고 점수 선택 |

FM 학습식:
$A^\tau = \tau A + (1-\tau)\epsilon$, 손실 $\|(A-\epsilon) - v_\theta(A^\tau,\tau,\text{cond})\|^2$.
추론은 Euler 10 step, $\tau: 0 \to 1$.

## 3. 하드웨어·환경 제약

- GPU: **RTX 5090 1장 (32GB, Blackwell sm_120)**.
  - PyTorch 2.7+ **cu128** 빌드 필수. 다른 빌드면 커널 오류가 난다. 작업 시작 시 확인한다.
- 동시에 돌리는 GPU 학습 작업은 **최대 2개**. rollout(시뮬레이터, CPU 위주)은 학습과 겹쳐도 된다.
- 렌더링: `export MUJOCO_GL=egl`.
- 마감: 발표 전날 밤 작업. 시간이 부족하면 `docs/plan/experiment1_plan.md`의 우선순위를 따른다.
- conda 환경: brl_ae (Python 3.10, torch 2.11.0+cu128). 모든 작업은 이 환경에서 실행한다.

## 4. 저장소 규칙

- `common/`: 여러 실험이 공유하는 코드 (데이터, AE 백본, FM, 평가). 수정하면 `docs/documents/logs/`에 이유를 남긴다.
- `experimentN/`: 해당 실험 전용 코드, 설정, 실행 스크립트.
- `outputs/` (git 제외): 캐시, 체크포인트, rollout 결과, 그래프. 경로 규칙:
  `outputs/experimentN/{cache,ckpt,rollouts,offline,tables,figures}/`
- 모든 스크립트:
  - `argparse` + YAML 설정 (`experimentN/configs/*.yaml`), CLI 인자로 덮어쓰기 가능
  - `--smoke` 플래그: 아주 작은 규모로 끝까지 실행 (데이터 몇 배치, step 수십 개, episode 1개)
  - `--seed` (기본 0), 시작 시 설정 전체를 출력하고 출력 폴더에 `config_used.yaml` 저장
  - 진행 로그는 stdout + `outputs/.../log.txt`
- 결과는 **JSON으로만** 저장하고, 표·그래프는 JSON에서 생성한다 (`07_aggregate.py`, `08_plots.py`).

## 5. 결과 정직성 규칙 (중요)

- 표와 그래프에 넣는 수치는 실제로 실행된 결과 JSON에서만 가져온다. 추정값·예상값을 채워 넣지 않는다.
- 모든 성공률에는 `n`(episode 수)을 함께 기록한다.
- 실행이 실패했거나 끝나지 않은 항목은 표에 `N/A (미실행)`으로 둔다.
- 예상과 다른 결과도 그대로 기록하고, `docs/documents/logs/`에 해석 메모를 남긴다.

## 6. 알려진 함정 (구현 시 반드시 확인)

1. **디코더 VLM의 causal attention**: 이미지 토큰이 텍스트보다 앞에 오면, 이미지 토큰의
   hidden state에는 지시문 정보가 없다. `h_tokens`는
   (이미지 토큰 pooling 12개) + (지시문 이후 텍스트 토큰 pooling 4개) = 16개로 만든다.
   텍스트 쪽 토큰이 이미지와 지시문을 모두 본 표현이다.
2. **LIBERO 이미지 방향**: 원본 hdf5의 정면/손목 이미지는 180도 뒤집혀 저장된 경우가 많다.
   학습 데이터와 rollout 관측에 **같은** 변환이 적용되는지 이미지를 저장해 눈으로 확인한다.
3. **정규화 통계**: action과 proprio의 min-max 통계는 학습 세트에서만 계산하고 체크포인트에 같이 저장한다.
   rollout에서 같은 통계로 역정규화한다.
4. **h 캐시와 온라인 h 일치**: rollout에서는 VLM을 온라인으로 돌린다. 캐싱 때와 같은 전처리,
   같은 레이어, 같은 pooling 함수(`common/models/vlm_features.py` 하나만 사용)를 쓴다.
5. **LIBERO-plus 인터페이스**: perturbation 적용 방법은 설치된 LIBERO-plus 코드를 직접 읽고 확인한다.
   추측해서 구현하지 않는다. Robot 교란은 평가 코드상 일부만 적용되는 문제가 보고되었으므로
   이번 실험에서는 Camera를 메인, Light 또는 Noise를 보조로 쓴다.

## 7. 작업 방식

- 큰 작업 전에 계획을 짧게 말하고 시작한다. 코드를 쓰면 바로 `--smoke`로 실행해 확인한다.
- 각 Phase가 끝나면 `docs/documents/logs/YYYYMMDD.md`에 한 줄 요약(무엇을, 결과, 문제)을 추가한다.
- 수 GB 이상 다운로드, `outputs/` 삭제, 환경 재설치는 실행 전에 사용자에게 확인한다.
- 장시간 작업은 `nohup` 또는 `tmux`로 띄우고, 로그 경로와 확인 명령을 알려준다.

## 8. 자주 쓰는 명령

```bash
export MUJOCO_GL=egl
python experiment1/scripts/00_env_check.py
python experiment1/scripts/01_cache_h.py --config experiment1/configs/cache_h.yaml
python experiment1/scripts/02_train_ae.py --config experiment1/configs/m1_h.yaml
python experiment1/scripts/06_rollout.py --model M1_h --perturb none --episodes_per_task 10
python experiment1/scripts/07_aggregate.py && python experiment1/scripts/08_plots.py
```

## 9. 전체 파일 구조
ae-shortcut/                      # 저장소 루트
├── CLAUDE.md
├── README.md
├── .gitignore                    # outputs/, 캐시, 체크포인트 제외
│
├── docs/
│   ├── plan/                     # 실험 계획서만
│   │   ├── experiment1_plan.md   # 이번 실험 계획
│   │   └── experiment2_plan.md   # SSM 물리 상태 인코더 (추후)
│   └── documents/                # 계획 외 나머지 자료
│       ├── overview.md           # 전체 연구 개요·가설 (h+O, shortcut, SSM 로드맵)
│       ├── experiment1_results.md  # P12에서 생성
│       ├── related_work.md       # HiRT, FiS-VLA, V-GPS, RoboMonkey, DiTEA 등
│       └── logs/
│           └── YYYYMMDD.md       # Phase별 진행 기록
│
├── common/                       # 실험 간 공유 코드
│   ├── data/
│   │   └── libero_dataset.py     # 로딩, 분할, 정규화, chunk 샘플링
│   ├── models/
│   │   ├── vlm_features.py       # h 추출 (캐싱·rollout 공용, 하나만 사용)
│   │   ├── flow_matching.py      # FM 손실, Euler 샘플러
│   │   ├── ae_transformer.py     # prefix 조건 AE 백본
│   │   └── encoders.py           # 손목 ResNet18, proprio MLP
│   ├── eval/
│   │   ├── rollout_core.py       # 환경 루프, 재계획, 성공 판정
│   │   └── perturbations.py      # LIBERO-plus 교란 래퍼
│   └── utils/                    # config, seed, logging
│
├── experiment1/                  # 이번 실험: h/O shortcut, LIBERO-plus
│   ├── configs/                  # cache_h, m1_h, m2_O, m3_hO, m4_moe, verifier.yaml
│   ├── models/
│   │   ├── moe.py                # M4
│   │   └── verifier.py           # M5
│   ├── scripts/
│   │   ├── 00_env_check.py
│   │   ├── 01_cache_h.py
│   │   ├── 02_train_ae.py
│   │   ├── 03_gen_negatives.py
│   │   ├── 04_train_verifier.py
│   │   ├── 05_offline_eval.py
│   │   ├── 06_rollout.py
│   │   ├── 07_aggregate.py
│   │   └── 08_plots.py
│   └── run/                      # phase4.sh, phase5_rollouts.sh
│
├── experiment2/                  # 다음 실험: SSM 물리 상태 인코더 z_t
│   ├── README.md                 # 계획만 (동역학 교란, GRU/transformer 비교군)
│   ├── configs/
│   ├── models/                   # ssm_encoder.py, dynamics_head.py
│   └── scripts/
│
└── outputs/                      # git 제외
    ├── experiment1/
    │   ├── cache/                # h_tokens, h_bar, split.json, 정규화 통계
    │   ├── ckpt/                 # M1_h/, M3_hO/, M4_moe/, verifier/
    │   ├── offline/              # {model}.json
    │   ├── rollouts/             # {model}__{perturb}__{h_mode}.json, 영상
    │   ├── tables/
    │   └── figures/
    └── experiment2/
