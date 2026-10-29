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

## 결과 — Phase 2 (검색 근거)

**측정 조건**: Phase 1·B-5와 동일한 train 42편(114샘플)·평가 cleaned 53문(val 8 + held-out 10, 18편)에서 **근거만 oracle→검색으로 교체**한다. 검색 = within-paper(질문 소속 논문 문단만) dense(bge-m3, CLS+L2, top-20) → cross-encoder 리랭커(bge-reranker-v2-m3) top-5. `clean_refs`는 검색 근거에도 동일 적용(train/inference 분포 일치). 학습 비용은 파라미터 수가 불변이라 B-3과 같고, 검색은 데이터 생성·추론에만 비용이 붙는다. 추가 지표 **evidence recall** = gold 근거 문단이 top-5에 포함된 비율(gold 있는 문항 평균). 코드 [src/retrieve/](src/retrieve/)(embed·rerank·pipeline), [scripts/baseline_search.py](scripts/baseline_search.py).

### Answer-F1 — oracle vs 검색 (cleaned 53문)

| 모델 | oracle | 검색 | Δ(검색−oracle) |
|---|---|---|---|
| base | 0.529 | 0.287 | −0.242 |
| full | 0.514 | 0.322 | −0.192 |
| LoRA | 0.515 | 0.279 | −0.236 |
| QLoRA | 0.556 | 0.304 | −0.252 |

**evidence recall(top-5) = 0.553** (4 레짐 동일 — 검색은 결정적이라 모델과 무관; oracle은 정의상 1.0). 검색 top-5가 gold 근거 문단의 ~55%만 회수한다. 분모는 gold 문단 id가 매핑된 40문항(나머지 13문항은 abstractive·boolean·unanswerable로 gold paragraph id 없음). 분포는 bimodal — 40 중 recall 1.0이 19·0.0이 13으로, "다 맞히거나 다 놓치거나"에 가깝고 완전 실패 13건이 무관 근거로 답을 짓게 해 F1 드롭의 직접 원인이 된다.

### 답 유형별 Answer-F1 (검색, cleaned 53)

| 유형 (n) | base | full | LoRA | QLoRA |
|---|---|---|---|---|
| extractive (33) | 0.29 | 0.37 | 0.34 | 0.35 |
| abstractive (6) | 0.12 | 0.15 | 0.11 | 0.11 |
| boolean (5) | 0.60 | 0.40 | 0.40 | 0.40 |
| unanswerable (9) | 0.22 | 0.22 | 0.11 | 0.22 |

oracle(위 cleaned 53 표) 대비 가장 큰 이동은 **unanswerable 붕괴**(base 1.00→0.22)와 **extractive 하락**(base 0.43→0.29)이다. extractive에서 FT>base 관계는 검색에서도 유지된다(full 0.37·LoRA 0.34·QLoRA 0.35 > base 0.29).

### 충실성 (NLI, cleaned 53) — oracle vs 검색

| 모델 | oracle 충실율 | 검색 충실율 | 검색 scored |
|---|---|---|---|
| base | 0.692 | 0.735 | 49 |
| full | 0.465 | 0.608 | 51 |
| LoRA | 0.512 | 0.596 | 52 |
| QLoRA | 0.628 | 0.510 | 51 |

### 해석 (정직)

- **검색은 base·세 레짐 모두 F1을 ~0.19–0.25 떨어뜨린다**(검색 축의 지배적 효과). 원인은 evidence recall 0.553 — top-5가 gold 근거의 45%를 놓친다.
- **드롭의 핵심 동인은 unanswerable 붕괴**다. oracle에선 1.00(근거 없음→"Unanswerable")이었는데, 검색은 **항상 top-5 문단을 반환**해 모델이 무관한 근거를 보고 답을 지어낸다 → 0.11~0.22. "근거 없음"을 못 가리는 것이 노이즈 근거의 핵심 실패 모드다. 지배 유형 extractive도 base 0.43→0.29로 내리지만 **FT>base는 유지**(FT 0.34~0.37) — B-5의 extractive 이득이 노이즈 근거에서도 살아남는다.
- **레짐 순위가 근거 조건에 따라 바뀐다**: oracle 1위 QLoRA(0.556)가 검색에선 드롭이 가장 커(−0.252) full(0.322)에 밀린다. full이 oracle→검색 gap을 가장 적게 잃는다(−0.192). 다만 검색 레짐 간 F1 폭(0.279~0.322)은 n=53에서 노이즈 범위이므로, 단단한 결론은 "FT가 노이즈 근거에서 붕괴하지 않고 지배 유형 extractive에서 base를 상회한다"까지다.
- **충실성**: 검색에서 base가 가장 충실(0.735)하나 F1은 최저 — 더 자주 보류한다. full·LoRA는 oracle보다 충실율이 오히려 오르는데(0.47→0.61, 0.51→0.60), 전제(근거)가 항상 존재해 entailment 분모·내용이 바뀐 기계적 이동이라 oracle과 직접 비교는 주의. QLoRA만 검색에서 base 아래로 내려간다(0.51).
- **종합**: oracle×검색 매트릭스가 보여주는 트레이드오프 = **근거 품질(recall 0.55)이 레짐 선택보다 F1을 더 가르고**, 적응 방법의 상대 우위는 근거 조건에 의존한다(oracle 최고 QLoRA, 검색 최강건 full). 생성 병목 위에 **검색 병목**이 겹친다.
- **한계**: n=53, 유형별 n은 더 작다(unanswerable 9·boolean 5). within-paper 검색이라 논문 라우팅 오류는 배제(그건 paper-rag가 측정) — 실제 end-to-end는 더 낮을 수 있다.

## 결과 — Phase 2 확대 평가 (eval N: 53→202)

**측정 조건(B-7)**: 53문의 "n 작음" 단서를 떼려고 **재학습 없이 평가셋만** 확대했다. Qasper dev split(현 학습·평가와 완전 분리)에서 단일주석 질문이 많은 순으로 60편을 뽑아 기존 eval 18편(val 8 + held-out 10)에 더해 **78편/202문**으로 늘렸다(기존 53문은 상위집합으로 그대로 포함). 모델은 B-5(oracle `_clean`)·B-6(검색 `_search`) 가중치를 재사용하고 근거·채점 파이프라인도 동일하다. dev raw는 공식 HF 형식(qas·full_text가 리스트)이라 `load_paper`의 평탄화 스키마로 변환하고 top-level id를 주입, 적재 검증을 통과한 논문만 채택했다.

### Answer-F1 — oracle vs 검색 (202문)

| 모델 | oracle | base 대비 | 검색 | base 대비 | Δ(검색−oracle) |
|---|---|---|---|---|---|
| base | 0.430 | — | 0.249 | — | −0.181 |
| full | 0.467 | +0.037 | 0.278 | +0.029 | −0.189 |
| LoRA | 0.478 | +0.048 | 0.265 | +0.016 | −0.213 |
| QLoRA | 0.482 | +0.052 | 0.275 | +0.027 | −0.207 |

53문에서 full·LoRA가 base에 못 미치던 집계가 **202문에선 전 레짐이 oracle·검색 양쪽에서 base를 상회**한다(oracle +0.04~0.05, 검색 +0.02~0.03). oracle 레짐 순위는 base < full < LoRA < QLoRA 단조 증가(QLoRA 최고), 검색은 full이 최고다. **evidence recall(top-5) = 0.603**(gold 문단 id가 매핑된 156문 평균; 4 레짐 동일, oracle은 정의상 1.0).

### 답 유형별 Answer-F1 (202문, base/full/LoRA/QLoRA)

| 유형 (n) | oracle | 검색 |
|---|---|---|
| extractive (106) | 0.39 / 0.48 / 0.48 / 0.48 | 0.26 / 0.33 / 0.32 / 0.32 |
| abstractive (49) | 0.17 / 0.18 / 0.20 / 0.19 | 0.16 / 0.16 / 0.15 / 0.14 |
| boolean (18) | 0.44 / 0.44 / 0.39 / 0.44 | 0.44 / 0.44 / 0.56 / 0.39 |
| unanswerable (29) | 1.00 / 0.93 / 1.00 / 1.00 | 0.24 / 0.17 / 0.07 / 0.28 |

지배 유형 extractive(52%)에서 **FT가 base를 oracle·검색 양쪽에서 확실히 상회**(oracle 0.39→0.48, 검색 0.26→0.32~0.33)해 B-5·B-6의 핵심 발견이 큰 셋에서 재확인된다. 검색의 **unanswerable 붕괴**(oracle ~1.0 → 검색 0.07~0.28)도 유지 — F1 드롭의 주동인.

### 충실성 (NLI, 202문) — oracle vs 검색

| 모델 | oracle 충실율 | 검색 충실율 | oracle scored | 검색 scored |
|---|---|---|---|---|
| base | 0.699 | 0.692 | 146 | 182 |
| full | 0.542 | 0.542 | 168 | 190 |
| LoRA | 0.557 | 0.528 | 167 | 197 |
| QLoRA | 0.575 | 0.528 | 167 | 180 |

base가 가장 충실(0.70)하나 F1은 최저 — B-5와 같은 교란(base가 더 자주 보류해 scored 분모↓, 쉬운 문항만 채점). FT는 **F1↑·충실율↓** 트레이드오프를 202문에서도 보인다.

### evidence recall 분포 (검색, 202문)

top-5 recall 0.603 (gold 문단 id가 매핑된 156문 기준; 나머지 46문은 abstractive·boolean·unanswerable로 gold paragraph id 없음). 분포는 53문과 같은 bimodal — 156 중 1.0이 87·0.0이 51·중간 18로 "다 맞히거나 다 놓치거나"다. 완전 실패(51건)가 무관 근거로 답을 짓게 해 F1 드롭의 직접 원인이 된다.

### 해석 (정직)

- **"n=53 노이즈" 단서를 뗄 근거가 생겼다.** 202문에서 전 FT가 oracle·검색 양쪽에서 base를 상회하고, oracle은 base<full<LoRA<QLoRA 단조 증가다. 53문에서 full/LoRA가 base에 −0.01 못 미치던 집계는 작은 n의 artifact였다.
- **레짐 간 폭은 여전히 modest**다(oracle 0.467~0.482, 검색 0.265~0.278). 단단한 서열은 "FT > base, oracle 최고 QLoRA, 검색 최고 full"까지고, full↔LoRA↔QLoRA 미세 차는 과장하지 않는다.
- B-6의 두 기둥 — **지배 유형 extractive에서 FT 이득**과 **검색의 unanswerable 붕괴** — 이 큰 셋에서 모두 재확인됐다. 근거 품질(recall 0.60)이 레짐 선택보다 F1을 더 가른다는 결론은 유지된다.
- **절대 F1이 53문보다 낮다**(base oracle 0.529→0.430): dev 60편이 abstractive 비중↑(24%)·난이도↑라 평균이 내려간 것이고, 레짐 간 **상대** 비교에는 영향이 없다.

## 결과 — abstention 실험 (검색 unanswerable 붕괴 완화 시도)

**측정 조건(B-8)**: B-6·B-7에서 검색 F1 드롭의 주동인으로 지목한 **unanswerable 붕괴**(검색이 항상 근거를 반환해 "근거 없음"을 못 가림)를 완화하려 abstention을 넣었다. 검색 하네스(`baseline_search.py`)에 `--abstain-threshold`를 추가해, 리랭커 top-1 로짓이 임계값 τ 미만이면 근거를 비워 생성한다(oracle의 unanswerable 경로를 재사용 — 강제 "Unanswerable" 문자열이 아니라 빈 근거). 임계값은 test 튜닝을 피하려 τ=None으로 top_score 분포를 먼저 수집해 유형 분리를 보고 정했다. 평가셋·가중치는 B-7과 동일(202문, `_search`).

### 임계값 보정 — top_score 분포 (base, τ=None)

| 유형 (n) | p25 | median | p75 |
|---|---|---|---|
| unanswerable (29) | −3.37 | −1.65 | 0.84 |
| answerable (173) | −2.46 | −0.38 | 1.74 |

방향은 맞으나(unanswerable이 더 낮음) 겹침이 커, τ를 올리면 unanswerable을 잡는 만큼 answerable도 버린다:

| τ | 기권재현(unans) | 오기권(ans) |
|---|---|---|
| −6 | 10.3% | 2.9% |
| −5 | 20.7% | 6.9% |
| −4 | 20.7% | 13.9% |
| −3 | 34.5% | 20.2% |
| 0 | 69.0% | 56.1% |

유형별 F1(기존 search)로 추정한 순효과가 최대가 되는 **τ=−5**를 택했다(더 올리면 answerable 피해가 unanswerable 회복을 추월). abstention은 리랭커 점수만 보므로 4 레짐이 동일하게 **18/202문**(unanswerable 6/29·answerable 12/173)을 기권한다.

### Answer-F1 — τ=None vs τ=−5 (202문)

| 모델 | τ=None | τ=−5 | Δ |
|---|---|---|---|
| base | 0.249 | 0.260 | +0.011 |
| full | 0.278 | 0.277 | −0.001 |
| LoRA | 0.265 | 0.275 | +0.009 |
| QLoRA | 0.275 | 0.285 | +0.009 |

base·LoRA·QLoRA는 **+0.01** 소폭 개선, full은 사실상 상쇄(−0.001). 유형별(base) None→τ−5로 보면 unanswerable 0.241→0.379(+0.138)가 이득원이고, boolean 0.444→0.389·extractive 0.257→0.253·abstractive 0.164→0.157이 오기권 손실이다. full은 answerable가 제일 강해 오기권 피해가 unanswerable 회복을 정확히 깎았다.

### 해석 (정직)

- **리랭커 top-1 로짓은 abstention 신호로 약하다.** unanswerable·answerable 점수가 심하게 겹쳐(median −1.65 vs −0.38), unanswerable을 의미 있게 잡는 어떤 τ도 answerable을 그만큼 버린다. τ=−5에서 unanswerable은 29문 중 6문만 잡히고 전체 이득은 +0.01에 그친다.
- **실패 모드는 실재하나 임계값만으로는 닫히지 않는다.** 제대로 닫으려면 검색 점수가 아닌 별도 answerability 신호(전용 분류기 등)가 필요하다 — 이것이 측정으로 드러난 한계다.
- 그래도 abstention 메커니즘(빈 근거 → oracle unanswerable 경로)은 작동했다: 기권한 unanswerable은 F1이 올랐고(0.24→0.38), 부작용은 오기권 손실로 정량화됐다.

## 결과 — abstention 신호 비교: 생성기 confidence (B-9)

**측정 조건(B-9)**: B-8이 "검색 점수가 아닌 별도 answerability 신호가 필요하다"로 끝나, 그 후보로 **생성기 자신의 confidence**를 측정했다. `generate_answer`가 생성 답의 평균 토큰 logprob를 함께 반환하게 하고(`return_confidence`), `baseline_search`가 질문별 `gen_confidence`를 기록한다. 기권은 사후(post-hoc) 적용 — confidence가 임계값 τ_g 미만이면 답을 "Unanswerable"로 대체한다. 그래서 confidence만 수집한 뒤 τ_g 스윕·F1을 저장된 행에서 오프라인 계산했다(기권 행 F1: unanswerable→1.0, answerable→0). 평가셋·가중치는 B-7과 동일(202문, `_search`). B-8과 달리 confidence는 모델별이라 4 레짐을 각각 수집했다. 신호만 견주려 v1(리랭커)과의 비교도 같은 사후 override 기준으로 본다(B-8의 빈-근거 재생성 수치와는 메커니즘이 달라 소폭 차이가 있다).

### confidence 분포 — 평균 토큰 logprob, 유형별 median

| 모델 | unanswerable (29) | answerable (173) |
|---|---|---|
| base | −0.292 | −0.269 |
| full | −0.724 | −0.653 |
| LoRA | −0.658 | −0.552 |
| QLoRA | −0.540 | −0.508 |

방향은 B-8과 같다(unanswerable이 조금 더 낮음 = 덜 확신). 하지만 median 차가 0.02~0.11로 분포 폭(사분위 0.3~0.5)보다 훨씬 작아 겹침이 크다.

### Answer-F1 이득 — 같은 오기권 예산 vs 상한 (202문)

v1(B-8)의 오기권 예산(τ=−5에서 6.9%) 이하로 묶었을 때와, 제약 없이 eval에 맞춘 전역 최적(상한)을 함께 본다.

| 모델 | base F1 | 예산 내 Δ (τ_g·기권재현·오기권) | 전역 best Δ (오기권) |
|---|---|---|---|
| base | 0.249 | +0.014 (−0.67·14%·5%) | +0.022 (16%) |
| full | 0.278 | +0.000 (−2.02·0%·1%) | +0.021 (22%) |
| LoRA | 0.265 | +0.012 (−1.35·10%·5%) | +0.032 (29%) |
| QLoRA | 0.275 | +0.022 (−1.03·17%·6%) | +0.037 (14%) |

같은 오기권 예산에서 이득은 +0.00~+0.022로, B-8의 리랭커 τ=−5(+0.00~+0.01)와 **동급이고 일관된 우위가 없다**(full은 0). 전역 best(+0.02~0.04)는 오기권 14~29%를 받아들여야 나오는 상한이다. 게다가 최적 τ_g가 −0.67~−2.02로 모델마다 달라, 리랭커 점수(모델 독립)와 달리 공유 임계값이 안 통한다.

### 왜 약한가 — 자발적 'Unanswerable' 방출 (저장 데이터 진단)

| 모델 | unanswerable 자발 기권 | answerable 오방출 |
|---|---|---|
| base | 7/29 (24%) | 13/173 |
| full | 5/29 (17%) | 7/173 |
| LoRA | 2/29 (7%) | 3/173 |
| QLoRA | 8/29 (28%) | 14/173 |

모델은 unanswerable의 72~93%에서 스스로 기권하지 못하고 **환각**한다. 특히 LoRA는 29문 중 2문만 "Unanswerable"을 낸다 — 검색 데이터 학습이 "항상 답하라"를 강화해 자발적 기권을 오히려 줄였다. 환각할 때 confidence가 answerable보다 살짝 낮긴 하나(평균 gap 0.07~0.18, PEFT에서 더 큼) 폭이 작아 임계값으로 가르기 어렵다.

### 해석 (정직)

- **생성기 confidence도 abstention 신호로 약하다.** 같은 오기권 예산에서 리랭커 점수와 동급(+0.01 안팎)이고, 모델별 임계값이 필요해 운용상 오히려 불리하다. 리트리버·생성기 두 신호 모두 unanswerable을 못 가린다.
- **"생성기는 아는가?"의 답은 대체로 '모른다'.** 모델은 unanswerable에서 자신 있게 틀린 답을 내며, 파인튜닝이 자발적 기권을 더 줄인다. 실패 모드를 생성 측에서 재확인했다.
- B-8+B-9를 합치면, 검색 unanswerable 붕괴를 닫으려면 근접 신호(리랭커 점수·생성 confidence)가 아니라 **전용 answerability 판단**(별도 분류기 등)이 필요하다는 결론이 두 각도에서 수렴한다.
