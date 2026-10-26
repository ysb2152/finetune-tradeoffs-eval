# EVALUATION — 평가 방법·지표

측정은 재현 가능하게 한다: 테스트셋 크기·하드웨어·시간을 함께 기록하고, 결과는 before/after로 적는다. 미측정 수치는 넣지 않는다.

## 평가셋

Qasper(NLP 논문 QA). 질문마다 gold evidence 문단과 reference answer가 있어 **생성 품질(Answer-F1)**과 **충실성(NLI)**을 둘 다 채점할 수 있다. 단일 주석 질문만 사용(다중 주석은 보류 처리). 학습/검증/held-out은 논문 단위로 분리한다.

## 측정 지표 (품질 × 비용)

**품질**

| 지표 | 정의 | 출처 |
|---|---|---|
| Answer-F1 | Qasper 표준 token-level F1 (reference 답 대비) | paper-rag `src/eval/answer.py` |
| 충실율(NLI) | 근거=전제·답=가설 entailment 비율 | paper-rag `src/eval/nli.py` |
| 비함의율(NLI) | 1 − 충실율 (neutral+contradiction, 환각의 근사) | 〃 |

**비용**

| 지표 | 정의 |
|---|---|
| peak VRAM | 학습 중 최대 GPU 메모리 |
| wall-clock | 학습 전체 소요 시간 |
| trainable params | 학습되는 파라미터 수(및 전체 대비 비율) |

## 측정 설계

- **베이스라인**: 튜닝 전 Qwen2.5-1.5B-Instruct(same prompt)로 품질 측정. 모든 학습 결과는 이 대비 Δ로 읽는다.
- **방법 축(P0-4)**: full FT / LoRA / QLoRA — 데이터·프롬프트·시드·에폭·eval 고정, 레짐만 변경.
- **근거 축(P0-5)**: oracle 근거 / 검색 근거(dense→리랭커 top-5). Phase 1 oracle → Phase 2 검색.
- **held-out**: 학습에 쓰지 않은 논문에서 같은 척도로 재측정해 과적합 확인.
- **환경**: Windows, RTX 3080(10GB). 버전·시드 고정.

## 결과 — Phase 1 (oracle 근거)

**측정 조건**: 학습셋 train 42편(단일주석 114샘플), 평가 held-out 10편/33문. RTX 3080(10GB), bf16 + gradient checkpointing + `adamw_bnb_8bit`, batch 1 × grad-accum 8(유효 8), 3 epochs, `max_seq_len` 1280, lr full 2e-5·LoRA/QLoRA 2e-4(방법별 관례). 입력 = oracle 근거 + 질문 → reference 답(생성만 격리). 코드 [scripts/train_sft.py](scripts/train_sft.py)·[scripts/baseline_oracle.py](scripts/baseline_oracle.py)·[scripts/baseline_nli.py](scripts/baseline_nli.py). (B-3·B-4)

### 비용 (학습, 42 step / 3 epoch)

| 레짐 | wall-clock | peak VRAM | trainable params | train_loss |
|---|---|---|---|---|
| full | 523.6s | 10.85 GB | 1,543,714,304 (100%) | 0.634 |
| LoRA | 109.3s | 5.12 GB | 18,464,768 (~1.2%) | 0.620 |
| QLoRA | 142.8s | 4.34 GB | 18,464,768 (~1.2%) | 0.629 |

full은 peak 10.85GB로 3080의 물리 10GB를 초과해 Windows 공유 시스템 메모리로 스필(OOM은 없으나 7~12s/step으로 느림). LoRA·QLoRA는 물리 VRAM 안에 적재되어 wall-clock ~5배↓. QLoRA는 메모리 최소(4.34GB)지만 4-bit 역양자화 연산으로 LoRA보다 느림 = 메모리↔속도 트레이드오프.

### 품질 (held-out 33문, 튜닝 전 대비 Δ)

| 모델 | Answer-F1 | Δ(F1) | 충실율(NLI) | 환각율 |
|---|---|---|---|---|
| base (튜닝 전 1.5B) | 0.543 | — | 0.769 | 0.231 |
| full | 0.472 | −0.071 | 0.630 | 0.370 |
| LoRA | 0.471 | −0.072 | 0.667 | 0.333 |
| QLoRA | 0.517 | −0.026 | 0.704 | 0.296 |

NLI는 보류(Unanswerable)·근거없음을 제외하고 채점(scored: base 26, full·LoRA 27, QLoRA 27).

### 답 유형별 Answer-F1 (회귀 원인 분해)

| 유형 (n) | base | full | LoRA | QLoRA |
|---|---|---|---|---|
| extractive (22) | 0.47 | 0.45 | 0.45 | 0.47 |
| abstractive (4) | 0.42 | 0.42 | 0.42 | 0.42 |
| unanswerable (5) | 1.00 | 0.80 | 0.80 | 1.00 |
| boolean (2) | 0.50 | 0.00 | 0.00 | 0.00 |

### 해석 (정직)

- **비용**: LoRA·QLoRA는 trainable을 full의 ~1.2%로 줄여 옵티마이저·활성 메모리를 함께 줄이고, 그 결과 물리 VRAM 안에 적재되어 full 대비 wall-clock ~5배↓. "최소 메모리(QLoRA) ≠ 최고 속도(LoRA)".
- **품질**: 전체 F1은 세 레짐 모두 튜닝 전보다 소폭 낮다. 그러나 유형별로 분해하면 **지배적 유형인 extractive(n22)·abstractive(n4)는 사실상 불변**이고, 회귀는 **boolean(n2: 0.50→0.00)·unanswerable(n5: 1.00→0.80) 소수 유형에 집중**된다. 육안 확인 결과, 파인튜닝 모델은 Yes/No·Unanswerable로 짧게 답해야 할 문항에서 근거의 내용을 추출해버린다(예: "Yes"→"NYT corpus, 24 relations", "Unanswerable"→"No"). SFT 학습셋이 extractive로 편중돼 system 프롬프트의 지시 따르기가 약화된 것이다. BIBREF 인용표지 누출("TransE BIBREF8…")도 있으나 1~2문으로 부차적.
- **결론**: 이 설정의 트레이드오프 = **지배 유형에서 품질 중립 + 비용 대폭↓**. 측정된 회귀는 "파인튜닝이 품질을 해쳤다"가 아니라 소수 유형의 지시 따르기 약화 + n=33의 fragile aggregate로 설명된다. 레짐 간 순위(full/LoRA/QLoRA)는 n이 작아 노이즈 범위이므로 과신하지 않는다.
- **한계·다음**: eval n 확대, SFT 데이터에 boolean/unanswerable 타깃 비중 보강, 타깃·근거에서 BIBREF 제거. train_loss(0.62~0.63)는 held-out 품질과 괴리하므로 일반화 지표로 쓸 수 없다. Phase 2(검색 근거)에서 동일 3 레짐 재측정 예정.
