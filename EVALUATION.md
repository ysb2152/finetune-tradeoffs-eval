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

## 결과 — 개선 반복 (REF 마커 정리 + eval 확대)

**동기(B-5)**: Phase 1 회귀 원인을 데이터로 검증하니, train 타깃은 boolean+unanswerable이 이미 31%(유형 편중 가설 기각)인 반면 **근거(모델 입력)의 39%(45/114)가 Qasper placeholder 마커로 오염**(BIBREF 121·TABREF 29·FIGREF 26·SECREF 8·FLOAT 8·EQREF 1)돼 있었다.

**개입**: (1) REF 패밀리(`[A-Z]{2,}REF\d*`)와 FLOAT/FORMULA를 제거하는 정규화를 학습 타깃·근거와 추론 근거에 동일 적용([src/data/sample.py](src/data/sample.py) `clean_refs`), 데이터·3 레짐 재생성·재학습. (2) 미학습 평가셋을 held-out 10편(33문) → val 8 + held-out 18편(53문)으로 확대(val은 튜닝 미사용이라 재활용). 학습 비용은 파라미터 수가 불변이라 B-3과 동일.

**교란 분리**: 정리와 eval셋 변화가 동시에 들어가므로, 같은 held-out 33문에 정리만 적용한 통제 실행을 추가해 정리 단독 효과를 분리했다.

### Answer-F1 — 3조건 분해

| 모델 | uncleaned 33 (Phase 1) | cleaned 33 (통제) | cleaned 53 |
|---|---|---|---|
| base | 0.543 | 0.517 | 0.529 |
| full | 0.472 | 0.477 | 0.514 |
| LoRA | 0.471 | 0.478 | 0.515 |
| QLoRA | 0.517 | 0.528 | 0.556 |

**base 대비 Δ:**

| | uncleaned 33 | cleaned 33 | cleaned 53 |
|---|---|---|---|
| full | −0.071 | −0.040 | −0.015 |
| LoRA | −0.072 | −0.039 | −0.014 |
| QLoRA | −0.026 | +0.011 | +0.027 |

정리 단독(1→2열)으로 gap이 약 절반 좁혀지며, 메커니즘은 base 하락(0.543→0.517)과 FT 소폭 상승이다. eval셋 확대(2→3열)가 QLoRA 우위를 더 벌린다. 즉 Phase 1 대비 큰 반전처럼 보이던 것은 정리 + 셋 변화의 합이고, **정리 단독 효과는 modest**하다.

### 답 유형별 Answer-F1 (extractive 중심)

| 조건 | base | full | LoRA | QLoRA |
|---|---|---|---|---|
| extractive, uncleaned 33 (n22) | 0.47 | 0.45 | 0.45 | 0.47 |
| extractive, cleaned 33 (n22) | 0.43 | 0.46 | 0.41 | 0.49 |
| extractive, cleaned 53 (n33) | 0.43 | 0.49 | 0.46 | 0.50 |

uncleaned에선 FT ≤ base였으나 정리하면 FT > base로 뒤집히고 두 eval셋에서 일관된다 — BIBREF 누출이 FT의 extractive 이득을 가리고 있었음을 보인다.

### 전체 유형별 (cleaned 53)

| 유형 (n) | base | full | LoRA | QLoRA |
|---|---|---|---|---|
| extractive (33) | 0.43 | 0.49 | 0.46 | 0.50 |
| abstractive (6) | 0.33 | 0.33 | 0.34 | 0.32 |
| boolean (5) | 0.60 | 0.20 | 0.20 | 0.40 |
| unanswerable (9) | 1.00 | 0.89 | 1.00 | 1.00 |

unanswerable 회귀는 회복(LoRA/QLoRA 1.00)됐고, boolean은 여전히 낮으나 n=5(통제셋 n=2)로 극소라 수치 결론에 쓰지 않는다.

### 충실성 (NLI)

| 조건 | 모델 | scored | abstained | 충실율 | 환각율 |
|---|---|---|---|---|---|
| cleaned 53 | base | 39 | 14 | 0.692 | 0.308 |
| | full | 43 | 9 | 0.465 | 0.535 |
| | LoRA | 43 | 10 | 0.512 | 0.488 |
| | QLoRA | 43 | 10 | 0.628 | 0.372 |
| cleaned 33 | base | 25 | 8 | 0.720 | 0.280 |
| | full | 27 | 5 | 0.556 | 0.444 |
| | LoRA | 27 | 6 | 0.593 | 0.407 |
| | QLoRA | 27 | 6 | 0.704 | 0.296 |

NLI는 FT가 base보다 낮지만 **교란이 있다**: base가 더 자주 보류(abstained↑)해 쉬운 문항만 채점되는 반면 FT는 더 많이 답해 분모가 커진다(사과-대-사과 아님). 또 정리로 전제(근거) 텍스트가 바뀌어 수치가 기계적으로 이동한다. QLoRA가 base에 최근접.

### 해석 (정직)

- **정리는 실재하나 modest한 이득**이다. base 대비 gap을 절반쯤 좁히지만, full/LoRA는 cleaned에서도 base보다 −0.04(33문) 아래이고 **QLoRA만 base를 근소 상회**(+0.01~0.03). "FT가 base를 크게 못 넘는다"는 큰 결론은 뒤집히지 않았다.
- **가장 방어 가능한 발견: 정리가 지배 유형 extractive에서 FT 이득을 드러냈다**(FT > base, 두 eval셋 일관). B-4의 "BIBREF는 부차적"은 출력 쪽만 본 것이고, 입력 쪽 39% 오염이 FT 이득을 가리고 있었다.
- **한계**: 정리·셋 확대로도 n은 여전히 작다(boolean n5/통제 n2). 레짐 순위는 노이즈 범위. train_loss(0.59~0.60)는 held-out과 여전히 괴리.
