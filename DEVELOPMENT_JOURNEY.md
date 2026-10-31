# finetune-tradeoffs-eval — 개발 일지

Qasper 논문 QA의 **생성 단계**를 소형 LLM(Qwen2.5-1.5B)에 파인튜닝하면서, **full FT · LoRA · QLoRA** 세 적응 방법의 품질/비용 트레이드오프를 같은 척도로 측정한다. 사실·정량 로그만 남긴다(어필 서술은 README).

## 문서 맵

- [README.md](README.md) — 포트폴리오 개요(어필용)
- [EVALUATION.md](EVALUATION.md) — 평가 방법·상세 지표 표

문서 구조·기록 관례는 `devlog-convention` 관례를 따른다(Part 0/A/B/C, 상태표, 체크박스 로드맵, ID 의사결정).

## 상태표 (최신 위)

| 날짜(ISO) | 분류 | 내용 | 관련ID |
|---|---|---|---|
| 2026-10-31 | result | answerability 게이트 결합 — top_score+confidence 로지스틱 결합(tune 학습)을 같은 오기권 예산(6.9%)서 단독과 비교(동작점 test 정렬, paired bootstrap 10k): 결합이 리랭커 단독 게이팅을 재현하고 conf 단독보다 전 레짐 f1_gain 낮음, 결합−단독 Δ CI 모두 0 포함(test unanswerable 20). 두 약신호 결합도 못 닫음 | B-12 |
| 2026-10-30 | result | F1 신뢰구간 — paired bootstrap(202문, 10k, seed 0): oracle LoRA/QLoRA의 base 상회만 유의(Δ CI 0 제외), 레짐 상위 간·검색은 전부 CI 겹침(QLoRA−LoRA −0.004). "QLoRA 최고/단조↑"는 평균 순위일 뿐 통계적 미지지 | B-11 |
| 2026-10-30 | chore | 재현성 자산 — split manifest(train 42/val 8/test 78, train↔평가셋 비중복 검증)·결과 CSV(runs/ 집계)·experiment_manifest(환경·하이퍼파라미터), format_sft `--evidence-mode {oracle,search}` 분리 | — |
| 2026-10-30 | result | LoRA rank 민감도 — r∈{4,8,16,32}: F1 평평(rank 과잉, r4가 1/4 파라미터로 r16 동급), r32 +0.03은 시드 아티팩트(시드 3개서 r16 분산 0.042 > 격차 0.008). VRAM·시간 불변·저장만 선형 | B-10 |
| 2026-10-29 | milestone | Phase 2 종료 — 검색 축 완결: F1 ~0.2↓(recall 0.60)·unanswerable 붕괴 주동인, abstention 두 신호(리랭커·생성 confidence) 모두 약함(근접 신호로 못 닫음), 검색 full 최강건 | C-2 |
| 2026-10-29 | result | abstention 신호 비교 — 생성기 confidence(평균 토큰 logprob): 같은 오기권 예산서 F1 +0.00~+0.02로 리랭커(B-8)와 동급·일관 우위 없음, τ_g 모델별. 모델이 unanswerable의 72~93%서 환각(LoRA 자발 기권 2/29) | B-9 |
| 2026-10-29 | result | abstention 실험 — 리랭커 top-1 로짓 τ=−5로 검색 unanswerable 붕괴 완화: F1 base/LoRA/QLoRA +0.01·full 상쇄, unans 0.24→0.38(29문 중 6문만 포착). 리랭커 점수는 abstention 신호로 약함 | B-8 |
| 2026-10-29 | result | Phase 2 확대 — eval N 53→202(dev 60편 추가, 재학습 없음): 전 FT가 oracle·검색서 base 상회, oracle QLoRA 최고(단조↑)·검색 full 최고, recall 0.60, extractive FT 이득·unanswerable 붕괴 재확인 | B-7 |
| 2026-10-28 | result | Phase 2 — 검색 축 3 레짐: 검색이 F1 ~0.2↓(recall 0.55), unanswerable 붕괴가 주동인, full 최강건·QLoRA 우위 상실 | B-6 |
| 2026-10-28 | result | 개선 반복 — REF 마커 정리 + eval n 확대, 통제로 정리 단독 효과 분리(modest, extractive에서 FT 유효, QLoRA만 base 상회) | B-5 |
| 2026-10-27 | result | Phase 1 품질 재측정 + 회귀 원인 진단 — 지배 유형 중립, 회귀는 소수 유형에 집중 | B-4 |
| 2026-10-27 | result | Phase 1 학습 비용 실측(3 레짐) — full 10.85GB 스필, LoRA/QLoRA wall-clock ~5배↓ | B-3 |
| 2026-10-26 | feat | SFT 데이터 포맷·분할(train42/val8/held-out10, 114샘플, completion-only) | B-2 |
| 2026-10-26 | result | 베이스라인(튜닝 전 1.5B) 측정 — held-out F1 0.543, 충실율 0.769 | B-1 |
| 2026-10-25 | decision | 프로젝트 착수 — 기술 결정 Part 0 확정 | P0-1~P0-8 |

---

## Part 0 — 기획·기술 결정

결정은 선호가 아니라 트레이드오프·근거·제약·기각한 대안으로 적는다. 뒤집을 때는 원 결정을 "폐기"로 표시하고 새 항목을 추가한다.

### P0-1 — 문제·목표

paper-rag-eval(B-20)에서 "코퍼스 라우팅·검색을 통과해 **gold 근거를 top-5에 확보한 뒤에도** Answer-F1이 ~0.48에 머문다 = 이 작동점의 병목은 검색이 아니라 **생성**"이라고 측정했다. 그 병목을 **학습**으로 공략한다.

- **핵심 질문**: 소형 모델(Qwen2.5-1.5B)에서 full FT / LoRA / QLoRA가 Qasper QA 생성 품질을 각각 얼마나 올리나, 그리고 비용(VRAM·학습 시간·학습 파라미터 수) 대비 트레이드오프는 어떻게 갈리나.
- **thesis**: "어느 방법이 낫다"는 단언이 아니라 **측정된 트레이드오프**다. 실무 기본값이 왜 LoRA/QLoRA인지 숫자로 보인다.
- 기각: 기성 모델 교체(체크포인트 스왑)는 학습이 아니라 모델 선택이고 paper-rag B-8(생성 모델 축 ablation)에서 이미 다뤘으므로 새 기여가 아니다.

### P0-2 — 평가셋·채점 (품질 + 비용 2축)

- **평가셋**: Qasper (paper-rag와 동일 — gold evidence + reference answer).
- **품질**: Qasper Answer-F1 + 로컬 NLI 충실율/비함의율. paper-rag 하네스를 재사용한다(P0-7). 보류(Unanswerable) 규칙도 동일.
- **비용**: peak VRAM · wall-clock 학습 시간 · 학습 파라미터 수(trainable params). 품질만이 아니라 비용을 함께 재야 "트레이드오프"가 된다.
- before/after 비교 기준선 = **튜닝 전 Qwen2.5-1.5B-Instruct**(same prompt). 상한 참조로 paper-rag의 7B 수치도 같은 척도라 병기 가능.

### P0-3 — 베이스 모델

**Qwen2.5-1.5B-Instruct.**
- paper-rag(Qwen2.5-7B)와 같은 Qwen 계열이라 서사가 이어지고, 0.5B보다 성능 여유가 있다.
- 10GB에서 **full FT가 가능한 상한에 가깝다** — grad checkpointing + 8-bit(또는 paged) Adam으로 적재.
- 기각: 0.5B(절대 성능 낮음), Llama-3.2-1B(계열 불연속·라이선스 조건 확인 부담).
- 제약·컨티전시: 1.5B full FT가 OOM이면 1B 또는 0.5B로 폴백한다. 폴백 여부는 첫 B-로그(실측 VRAM)로 결정한다 — 지금 단정하지 않는다.

### P0-4 — 적응 방법 축 (핵심 비교)

동일 모델·동일 데이터에서 **full FT / LoRA / QLoRA** 3 레짐을 격리 비교한다.
- full FT: 전체 파라미터 학습. grad checkpointing + 8-bit Adam으로 10GB 적재.
- LoRA: fp16 베이스 + 저랭크 어댑터.
- QLoRA: 4-bit 베이스 + 저랭크 어댑터.
- 레짐만 바꾸고 데이터·프롬프트·시드·에폭·eval은 고정해 방법 효과를 격리한다.

### P0-5 — 학습 타깃 축 (근거 조건)

입력→출력 포맷을 두 조건으로 비교한다.
- (a) **oracle 근거 + 질문 → reference 답**: 생성만 격리. 측정된 병목(P0-1)에 정확히 대응.
- (b) **검색 근거 + 질문 → reference 답**: dense→리랭커 top-5 근거로 end-to-end 현실 조건.
- **단계화**: Phase 1 = oracle 축(3 레짐) 완성 → Phase 2 = 검색 축(3 레짐). Phase 1만으로도 완결된 비교가 나오고, Phase 2에서 "노이즈 근거에 대한 강건성"이 추가된다.

### P0-6 — 데이터 범위·분할

- Qasper train split에서 SFT 학습셋을 만든다. paper-rag 하네스 호환을 위해 **단일 주석 질문**만 쓴다(다중 주석은 보류 처리되므로 제외).
- 학습/검증/held-out을 **논문 단위로 분리**(질문 누출 방지). held-out은 학습에 쓰지 않은 논문.
- 정확한 분량(질문·논문 수)은 첫 B-로그에서 실측 후 확정한다 — 미측정 수치를 지금 쓰지 않는다.

### P0-7 — 평가 하네스 재사용

paper-rag-eval의 채점 모듈([src/eval/answer.py](../RAG/src/eval/answer.py) Answer-F1, [src/eval/nli.py](../RAG/src/eval/nli.py) NLI, Qasper ingest)을 이 저장소로 **복사해 self-contained**로 둔다. 출처를 명시한다.
- 근거: 포트폴리오 저장소는 독립 실행 가능해야 한다. 패키지 의존보다 복사+출처 표기가 재현·열람에 낫다.

### P0-8 — 스택·하드웨어

- Python, 로컬 RTX 3080(10GB). `transformers` + `peft` + `bitsandbytes` + `trl`(또는 수동 `Trainer`). 버전은 재현 위해 고정.
- 채점용 모델(NLI DeBERTa, 임베더/리랭커는 검색 근거 조건에서만)은 paper-rag와 동일 체크포인트.

---

## Part A — 개요·설계

(채울 예정) 데이터 포맷팅 → 학습(3 레짐 × 2 근거 조건) → 평가(품질 Answer-F1·NLI + 비용 VRAM·시간·params) → 비교표·회귀.

## Part B — 개발 로그

### B-1 — 베이스라인(튜닝 전 1.5B) 측정

학습 결과를 Δ로 읽으려면 before 기준선이 필요하다(P0-2). 튜닝 전 Qwen2.5-1.5B-Instruct를 same prompt로 held-out 10편/33문(단일주석)에 돌려 oracle 근거 + 질문 → 답을 생성하고 Qasper Answer-F1 + 로컬 NLI로 채점했다.

**결과: Answer-F1 0.543, 충실율 0.769 / 환각율 0.231(scored 26, 보류 7 제외). 추론 VRAM ~3.1GB.** 상세 [EVALUATION.md](EVALUATION.md).

원칙: before 기준선은 이후 모든 Δ의 기준 — 같은 프롬프트·평가셋·하네스로 고정한다.

### B-2 — SFT 데이터 포맷·분할

3 레짐을 같은 데이터로 학습하려면 chat 포맷 SFT 셋과 질문 누출 없는 분할이 필요하다. 논문 단위로 train 42 / val 8(seed 0) + held-out 10(평가 전용)으로 나누고, 단일 주석 질문만 써서 oracle 근거 + 질문 → reference 답을 chat 3턴(system/user/assistant)으로 만들었다. 손실은 completion-only(assistant 응답 토큰에만, `response_template` = `<|im_start|>assistant\n`)로 두어 프롬프트 토큰은 학습에서 제외했다.

**결과: [scripts/format_sft.py](scripts/format_sft.py)가 `data/sft/train.jsonl` 114 샘플 생성.** 분할·포맷을 고정해 이후 레짐 효과만 격리한다.

원칙: 질문 누출은 논문 단위 분할로 막고, 생성 품질 학습은 프롬프트 토큰 손실을 제외(completion-only)한다.

### B-3 — Phase 1 학습: 3 레짐 비용 실측

P0-4 방법 축. 레짐만 바꾸고(데이터·시드·에폭·eval 고정) full FT / LoRA / QLoRA의 비용을 같은 척도로 쟀다. TRL `SFTTrainer`로 full = bf16 전체 학습, LoRA = bf16 base + 저랭크 어댑터(r16/α32/all-linear), QLoRA = 4-bit NF4 base + 어댑터. 공통으로 grad checkpointing + `adamw_bnb_8bit`, batch 1 × accum 8, 3 epoch, `max_seq_len` 1280. 함정: LoRA + gradient checkpointing에서 frozen base의 입력 임베딩에 grad 진입점이 없어 backward가 끊긴다("element 0 … does not require grad") → `enable_input_require_grads()` 호출로 해결(QLoRA는 `prepare_model_for_kbit_training`이 동일 처리). 코드 [scripts/train_sft.py](scripts/train_sft.py).

**결과: full 523.6s / 10.85GB / 1.54B(100%), LoRA 109.3s / 5.12GB / 18.46M(~1.2%), QLoRA 142.8s / 4.34GB / 18.46M. full은 10GB 물리 VRAM을 초과해 공유 메모리로 스필(7~12s/step)되는 반면 LoRA·QLoRA는 적재되어 wall-clock ~5배↓. QLoRA는 메모리 최소지만 4-bit 역양자화로 LoRA보다 느리다.** 표 [EVALUATION.md](EVALUATION.md).

원칙: 적응 방법의 비용 우위는 trainable 축소가 옵티마이저·활성 메모리를 줄여 물리 VRAM 안에 들어가는 데서 나온다 — 최소 메모리 ≠ 최고 속도.

### B-4 — Phase 1 품질 재측정 + 회귀 원인 진단

비용만으론 트레이드오프가 반쪽이다. 학습 가중치/어댑터를 평가 경로에 물려 held-out 품질을 base와 같은 척도로 쟀다. full은 저장된 모델 디렉터리를 그대로 로드, LoRA/QLoRA는 bf16 base에 `PeftModel`로 어댑터를 merge해 같은 oracle 평가(held-out 33문)에 넣었다(QLoRA도 bf16 base에 얹어 LoRA와 동일 조건 비교, train 4-bit와의 미세 불일치는 통상 관행대로 수용). 코드 [src/generate/qwen.py](src/generate/qwen.py)·[scripts/baseline_oracle.py](scripts/baseline_oracle.py).

**결과: Answer-F1 full 0.472 / LoRA 0.471 / QLoRA 0.517 — 세 레짐 모두 base 0.543에 못 미친다.** 유형별로 분해하니 **extractive(n22)·abstractive(n4)는 사실상 불변, 회귀는 boolean(n2: 0.50→0.00)·unanswerable(n5: 1.00→0.80) 소수 유형에 집중**됐다. 육안 확인 결과 파인튜닝 모델이 Yes/No·Unanswerable로 짧게 답할 문항에서 근거 내용을 추출해버린다(예: "Yes"→내용 추출, "Unanswerable"→"No") = SFT(extractive 편중)가 지시 따르기를 약화. BIBREF 인용표지 누출은 1~2문으로 부차적. **정직한 결론: 이 설정의 트레이드오프는 지배 유형 품질 중립 + 비용 대폭↓이며, 측정된 회귀는 소수 유형의 지시 약화 + n=33 fragile aggregate로 설명된다. 레짐 순위는 과신하지 않는다.** 상세·표 [EVALUATION.md](EVALUATION.md).

원칙: 집계 지표의 하락은 유형·n으로 분해해 원인을 봐야 하고, train_loss는 일반화를 대변하지 못한다.

### B-5 — 개선 반복: REF 마커 정리 + eval n 확대 (정리 단독 효과 분리)

B-4의 회귀 원인 가설(SFT의 extractive 편중, BIBREF 누출)을 데이터로 검증했다. train 114샘플의 타깃 유형을 세어보니 boolean 19 + unanswerable 16 = 31%로 이미 충분해 "유형 편중" 가설은 약했다(→ 유형 균형 보강은 기각). 대신 **근거(모델 입력)의 39%(45/114)가 Qasper placeholder 마커로 오염**돼 있었다(BIBREF 121·TABREF 29·FIGREF 26·SECREF 8·FLOAT 8·EQREF 1). B-4가 "부차적"이라 본 것은 출력 쪽(타깃 3/114)이고, 입력 쪽 오염은 훨씬 컸다. 해결: REF 패밀리(`[A-Z]{2,}REF\d*`) + FLOAT/FORMULA를 제거하는 정규화를 **학습 타깃·근거와 추론 근거에 동일 적용**(train/inference 입력 분포 일치가 관건)하고 데이터·3 레짐을 재생성·재학습했다. 평가는 미학습 셋을 held-out 10편(33문) → val 8 + held-out 18편(53문)으로 확대했다(val은 튜닝에 쓰지 않아 재활용). 정리와 eval셋 변화가 동시에 들어가는 교란을 막으려, **같은 held-out 33문에 정리만 적용한 통제 실행**을 따로 돌려 정리 단독 효과를 분리했다. 코드 [src/data/sample.py](src/data/sample.py) `clean_refs`.

**결과(3조건 base 대비 gap): uncleaned33 full −0.071·LoRA −0.072·QLoRA −0.026 → cleaned33 −0.040·−0.039·+0.011 → cleaned53 −0.015·−0.014·+0.027.** 정리 단독으로 gap이 절반쯤 좁고(base가 내려가고 FT가 소폭↑), QLoRA만 base를 근소 상회한다. 가장 단단한 발견은 **지배 유형 extractive에서 정리 후 FT가 base를 뒤집는다(base 0.43 → FT 0.46~0.50, 두 eval셋 일관)** = BIBREF 누출이 FT의 extractive 이득을 가리고 있었다. boolean(n2)은 cleaned에서도 0.00으로 남지만 극소 n이라 수치 결론엔 쓰지 않는다. 표 [EVALUATION.md](EVALUATION.md).

원칙: 진단 가설은 분포를 직접 세어 검증하고, 개입이 둘이면 통제 실행으로 요인을 분리한다.

### B-6 — Phase 2: 검색 축 3 레짐 (oracle×검색 매트릭스 완성)

P0-5의 검색 축을 채웠다. within-paper 검색(dense bge-m3 top-20 → 리랭커 bge-reranker-v2-m3 top-5, `clean_refs` 동일)으로 train·eval 근거를 oracle에서 교체하고, B-3와 동일 비용 구조로 3 레짐을 재학습·재평가했다(학습은 파라미터 불변; 검색은 데이터 생성·추론 비용만 추가). 평가 cleaned 53문, 추가 지표 evidence recall. within-paper로 범위를 잡아 논문 라우팅 오류를 배제하고 생성의 노이즈 강건성만 격리했다. 코드 [src/retrieve/](src/retrieve/)·[scripts/baseline_search.py](scripts/baseline_search.py).

**결과**: 검색이 F1을 전 레짐 ~0.19–0.25↓(base 0.529→0.287, full 0.514→0.322, LoRA 0.515→0.279, QLoRA 0.556→0.304), evidence recall 0.553(top-5가 gold 근거의 ~55%만 회수). 유형 분해로 드롭의 주동인은 **unanswerable 붕괴**(oracle 1.00→검색 0.11~0.22): 검색이 항상 top-5를 반환해 모델이 "근거 없음"을 못 가리고 답을 짓는다. 반면 지배 유형 extractive는 검색에서도 FT>base 유지(base 0.29 vs FT 0.34~0.37). 레짐 순위는 oracle(QLoRA 1위)에서 검색(full 드롭 최소 −0.192, QLoRA 드롭 최대 −0.252)으로 바뀐다 — 단 검색 레짐 간 폭은 n=53 노이즈 범위. 표 [EVALUATION.md](EVALUATION.md).

원칙: 품질 드롭은 집계가 아니라 유형으로 분해해 메커니즘(unanswerable)을 찾고, 결정적 지표(recall)로 원인을 정량화한다.

### B-7 — Phase 2 확대 평가 (eval N 53→202, 재학습 없음)

B-4~B-6 내내 레짐 순위와 FT−base 집계에 "n 작음" 단서를 달았다. 결론을 단단히 하려고 **평가셋만** 키웠다: Qasper dev split(현 학습·평가와 완전 분리)에서 단일주석 질문이 많은 순으로 60편을 뽑아 기존 eval 18편(held-out 10 + val 8)에 더해 78편/202문으로 확대했다(기존 53문은 상위집합으로 그대로 포함). 모델은 B-5(oracle `_clean`)·B-6(검색 `_search`) 가중치를 재사용해 **재학습 없이** oracle·검색 × 4모델 + NLI만 다시 돌렸다. dev raw는 공식 HF 형식(qas·full_text가 리스트)이라 load_paper 평탄화 스키마로 변환하고 top-level id를 주입, 적재 검증을 통과한 논문만 채택했다.

**결과: 전 레짐이 oracle·검색 양쪽에서 base를 상회한다(oracle base 0.430 < full 0.467 < LoRA 0.478 < QLoRA 0.482 단조↑; 검색 base 0.249 < LoRA 0.265 < QLoRA 0.275 < full 0.278).** 53문에서 full·LoRA가 base에 −0.01 못 미치던 집계는 작은 n의 artifact였다. evidence recall 0.603(gold 있는 156문, 여전히 bimodal). B-6의 두 기둥 — 지배 유형 extractive에서 FT 이득(oracle 0.39→0.48, 검색 0.26→0.33), 검색의 unanswerable 붕괴(oracle ~1.0→검색 0.07~0.28) — 가 큰 셋에서 재확인됐다. 레짐 간 폭은 여전히 modest라 "FT>base, oracle 최고 QLoRA·검색 최고 full"까지만 단언한다. 절대 F1이 53문보다 낮은 건 dev 60편의 abstractive 비중·난이도 때문이고 레짐 간 상대 비교엔 무관하다. 표 [EVALUATION.md](EVALUATION.md).

원칙: 작은 n의 순위·집계는 과신하지 않고, 재학습 없이 평가셋을 키워 결론의 강건성을 검증한다.

### B-8 — abstention 실험 (검색 unanswerable 붕괴 완화)

B-6·B-7에서 검색 F1 드롭의 주동인으로 짚은 **unanswerable 붕괴**(검색이 항상 근거를 반환)를 완화하려 abstention을 넣었다. `baseline_search.py`에 `--abstain-threshold`를 추가해 리랭커 top-1 로짓이 τ 미만이면 근거를 비워 생성한다(oracle의 unanswerable 경로 재사용). test 튜닝을 피해 τ=None으로 top_score 분포를 먼저 수집했는데 unanswerable·answerable이 심하게 겹쳤다(median −1.65 vs −0.38). 순효과가 최대인 **τ=−5**를 택해 4 레짐에 적용(점수 기반이라 동일 18/202 기권).

**결과: F1 base/LoRA/QLoRA +0.01·full −0.001(상쇄).** unanswerable이 0.24→0.38로 올랐으나 29문 중 6문만 잡혔고(τ를 올리면 answerable 피해 급증), boolean·extractive·abstractive는 오기권으로 미세 손실. 결론: **리랭커 점수는 abstention 신호로 약하다** — 실패 모드는 실재하나 임계값만으로는 닫히지 않고, 별도 answerability 신호가 필요하다. 표 [EVALUATION.md](EVALUATION.md).

원칙: 개선 시도의 음성·미미한 결과도 측정해 정직하게 남긴다.

### B-9 — abstention 신호 비교: 생성기 confidence

B-8이 "검색 점수가 아닌 별도 answerability 신호가 필요하다"로 끝나, 그 후보로 생성기 자신의 confidence를 쟀다. `generate_answer`가 평균 토큰 logprob를 함께 반환하게 하고(`return_confidence`) `baseline_search`가 `gen_confidence`를 기록한다. 기권은 사후 적용(confidence가 τ_g 미만이면 답을 "Unanswerable"로 대체)이라 confidence만 수집한 뒤 τ_g 스윕·F1을 오프라인 계산했다(기권 행: unanswerable→1.0, answerable→0). confidence는 모델별이라 4 레짐을 각각 수집했다.

**결과: 생성기 confidence도 약한 신호.** unanswerable confidence median이 answerable보다 조금 낮을 뿐(gap 0.02~0.11 ≪ 분포 폭) 겹침이 크다. v1(B-8)의 오기권 예산(6.9%) 이하로 묶으면 F1 이득은 +0.00~+0.02로 리랭커 τ=−5(+0.00~+0.01)와 동급이고 일관된 우위가 없다(full은 0). 전역 best(+0.02~0.04)는 오기권 14~29%를 받아들인 상한이고, 최적 τ_g가 −0.67~−2.02로 모델마다 달라 공유 임계값이 안 통한다. 저장 데이터 진단: 모델은 unanswerable의 72~93%에서 환각하고, 파인튜닝은 자발적 "Unanswerable" 방출을 오히려 줄였다(LoRA 2/29). 표 [EVALUATION.md](EVALUATION.md).

원칙: 대안 신호의 음성 결과도 측정해 남긴다.

### B-10 — LoRA rank 민감도 + 시드 검증

LoRA 핵심 노브 rank r의 트레이드오프를 재려 r ∈ {4,8,16,32}를 학습·평가했다(`--rank`, alpha=2r로 용량만 격리, search SFT·202문). r=16은 기존 LoRA(search 0.265 재현)로 일관성 확인. 단일 시드의 작은 차를 거르려 r16·r32를 시드 {42,100,202}로 재학습(어댑터 초기화 전 `set_seed`)해 분산을 쟀다.

**결과: rank는 F1에 영향 없음.** 시드 42에서 r4~16은 ±0.01 내 평평(비단조)하고 r32만 +0.03처럼 보였으나, 시드 3개로 보면 r16 분산(range 0.042)이 r32−r16 평균차(0.008)의 5배라 구간이 겹친다 — r32 이득은 단일 시드 아티팩트. rank 16은 이 과제·1.5B에 과잉이고 r4가 파라미터 1/4로 동급이다. 비용은 trainable params만 rank에 선형이고 VRAM(5.22→5.62 GB)·wall-clock(~120s)은 불변(연산은 base forward 지배). 부수로 시드 변동 ±0.04가 레짐 격차(≤0.04)와 맞먹어 작은 레짐 차도 같은 눈금으로 읽어야 함을 보였다. 표 [EVALUATION.md](EVALUATION.md).

원칙: 작은 노브·격차는 시드로 검증하고, 노이즈면 음성으로 남긴다.

### B-12 — answerability 게이트: 두 약신호 결합

B-8·B-9가 리랭커 점수·생성 confidence 둘 다 약하다로 끝나, 전용 신호에 투자하기 전 가장 싼 수를 먼저 쟀다 — 두 약신호를 **하나의 로지스틱 회귀로 결합**하면 단독보다 나은지다(`gate_answerability.py`). tune(18편 53문)에서 두 신호를 표준화·학습하고, 세 신호(top_score 단독·confidence 단독·결합)를 test(60편 149문, unanswerable 20)에서 **같은 오기권 예산(6.9%)에 동작점을 맞춰** 비교했다(`gate_compare.py`). tune 임계값은 test 기권율이 0.06~0.14로 흩어져 신호 비교가 불공정하므로, 세 신호 모두 test answerable의 6.9%만 기권하도록 동작점을 test에서 정렬했다(A안: 운용 추정이 아니라 신호 간 공정 비교용 진단). 질문 단위 paired bootstrap(10k, seed 0)으로 신호별 CI와 결합−단독 Δ를 구했다.

**결과: 결합도 못 닫는다.** 같은 예산에서 결합 f1_gain은 base +0.004·full −0.006·lora −0.003·qlora −0.001로, 더 나은 단독(confidence +0.007·−0.004·+0.003·+0.023)보다 전 레짐에서 낮다(결합−conf Δ 모두 음수). 결합은 사실상 리랭커 단독 게이트를 재현했고(full은 모든 리샘플에서 게이팅 동일), 결합−단독 Δ의 95% CI는 catch·f1_gain 모두 0을 포함한다 — test unanswerable이 20개뿐이라 넓기도 하나, 두 신호 분포가 겹치는 한(B-8·B-9) 결합이 가를 수 없다는 게 핵심이다. 표 [EVALUATION.md](EVALUATION.md).

원칙: 비싼 수(전용 모델) 전에 싼 결합부터 측정해 음성으로 거른다.

## Part C — 회고

### C-1 — 배운 것

- **비용 우위의 메커니즘은 "파라미터가 적다"가 아니라 "물리 VRAM에 들어간다"였다.** 초기 가정은 LoRA/QLoRA가 단순히 조금 가볍다는 것이었는데, 실제로는 full이 10GB를 **초과**해 공유 메모리로 스필되며 7~12s/step으로 느려졌고, 물리 VRAM 안에 들어간 LoRA·QLoRA가 ~5배 빨랐다. 또 메모리가 가장 작은 QLoRA가 가장 빠르지는 않았다(4-bit 역양자화 비용).
- **train_loss는 일반화를 대변하지 못한다.** 세 레짐의 train_loss(0.62~0.63)는 비슷했지만 held-out 품질은 모두 base보다 낮았다. 집계 F1 하락(0.543→0.47~0.52)도 유형별로 분해하니 지배 유형은 불변이고 소수 유형(boolean n2·unanswerable n5)에 회귀가 집중된 것이었다 — 작은 n에서는 1~2문이 지표를 크게 흔든다.
- **진단 가설은 숫자로 검증하고, 개입이 둘이면 통제로 분리한다.** B-4는 회귀 원인을 "SFT의 extractive 편중"으로 적었지만 train 분포를 실제로 세어보니 boolean+unanswerable이 31%여서 그 가설은 약했다 — 진짜 오염은 근거 입력의 39%를 차지한 BIBREF류 마커였다. 정리 후 지배 유형 extractive에서 FT가 base를 뒤집어(0.43→0.46~0.50) 누출이 이득을 가리고 있었음을 보였다. 다만 정리와 eval셋 확대를 동시에 넣으면 효과를 혼동하므로 같은 33문 통제 실행으로 분리했고, 그 결과 정리 단독 효과는 modest(gap 절반, QLoRA만 base 근소 상회)임이 드러났다 — 큰 결론("FT가 base를 크게 못 넘음")은 유지됐다.

- **근거 품질이 적응 방법보다 F1을 더 가른다.** oracle→검색에서 전 레짐이 ~0.2 떨어졌는데(recall 0.55) 이는 레짐 간 차이(≤0.04)보다 훨씬 크다. 적응 방법의 상대 우위도 근거 조건에 의존했다(oracle 최고 QLoRA, 검색 최강건 full). 특히 검색이 **항상** 근거를 반환하는 탓에 unanswerable이 1.00→0.22로 무너졌다 — "근거 없음"을 모델이 못 가리는 것이 노이즈 근거의 핵심 실패 모드다. 생성 병목 위에 검색 병목이 겹친다.

- **작은 n의 순위는 재측정으로 검증했다.** 53문에서 full/LoRA가 base에 −0.01 뒤처진 집계에 "노이즈" 단서를 달았는데(B-7), 재학습 없이 eval을 202문으로 키우니 전 FT가 base를 상회하고 oracle이 base<full<LoRA<QLoRA로 단조 증가했다 — 단서가 옳았고, 확대가 결론을 단단히 했다. 반대로 레짐 간 미세 폭·검색의 unanswerable 붕괴는 큰 셋에서도 그대로여서 과장 없이 유지된다.

- **실패 모드를 닫으려는 시도도 음성으로 측정했다.** 검색의 unanswerable 붕괴를 리랭커 점수 임계값(τ=−5) abstention으로 완화해봤지만, 점수가 두 유형을 못 가려(분포 겹침 큼) 전체 F1은 +0.01에 그쳤고 full은 상쇄됐다(B-8). "실재하는 실패 모드"와 "가까운 신호로 닫힘"은 별개임을 수치로 확인했다 — 제대로 닫으려면 검색 점수가 아닌 전용 answerability 신호가 필요하다.

- **대안 신호도 측정해 같은 결론을 재확인했다.** B-8의 "별도 answerability 신호가 필요하다"를 받아, 리트리버 점수 대신 생성기 자신의 confidence(평균 토큰 logprob)를 abstention 신호로 시험했다(B-9). 같은 오기권 예산에서 이득은 리랭커와 동급(+0.01 안팎)이고 임계값이 모델별이라 오히려 불리했다 — 모델은 unanswerable에서 자신 있게 틀리고, 파인튜닝이 자발적 기권을 더 줄였다. "생성기는 아는가"의 답은 대체로 '모른다'였고, 두 신호가 수렴해 **근접 신호로는 안 닫힌다**를 굳혔다.

- **작은 노브의 이득도 시드로 검증했다.** LoRA rank 스윕(B-10)에서 r32가 +0.03 나아 보였지만, 시드 3개로 돌리니 r16의 시드 분산(0.042)이 그 격차(0.008)를 삼켜 둘이 겹쳤다 — 단일 시드 아티팩트였다. 동시에 시드 변동 ±0.04가 레짐 간 격차와 맞먹음을 확인해, "작은 차이는 재측정으로 거른다"는 원칙을 노브 축에서도 반복했다. rank는 품질에 영향 없고(과잉), 낮추면 저장 파라미터만 공짜로 절약된다.

### C-2 — Phase 2 종합 (종료)

Phase 2(검색 근거 축)를 종료한다. oracle→검색으로 근거 조건을 바꾸자 전 레짐 F1이 ~0.2 떨어졌고(evidence recall 0.60), 드롭의 주동인은 unanswerable 붕괴(검색이 항상 근거를 반환해 "근거 없음"을 못 가림)였다. 레짐 간 상대 우위도 근거 조건에 의존했다(oracle 최고 QLoRA, 검색 최강건 full). 붕괴를 닫으려 abstention을 두 신호로 시험했으나 — 리랭커 top-1 로짓(B-8)·생성기 평균 토큰 logprob(B-9) 모두 unanswerable·answerable을 못 가려(분포 겹침) 같은 오기권 예산에서 F1 이득이 +0.01 안팎에 그쳤다. 근접 신호로는 실패 모드가 닫히지 않으며, 제대로 닫으려면 전용 answerability 판단(별도 분류기 등)이 필요하다 — 측정으로 드러난 한계이자 다음 단계의 과제다. Phase 2의 순수 기여는 "근거 품질이 적응 방법보다 F1을 더 가른다"를 oracle×검색 매트릭스로 정량화하고, 그 위에 abstention 한계까지 음성으로 남긴 것이다.

(로드맵은 아래.)

---

## 로드맵

- [x] P0 확정 및 저장소 스캐폴딩
- [x] 평가 하네스 이식(answer.py·nli.py·Qasper ingest) + 베이스라인(튜닝 전 1.5B) 측정 (B-1)
- [x] 데이터 포맷팅 파이프라인(oracle 근거 → SFT 샘플) + 분할 확정 (B-2)
- [x] Phase 1 — oracle 축: full FT / LoRA / QLoRA 학습·평가·비용 측정 (B-3·B-4)
- [x] Phase 1 비교표(품질×비용) + held-out 일반화 + 유형별 진단 ([EVALUATION.md](EVALUATION.md))
- [x] 개선 반복 — REF 마커 정리(근거 입력 39% 오염 제거) + eval n 확대(33→53) + 통제로 정리 단독 효과 분리 (B-5). 유형 균형은 train에 boolean/unanswerable이 이미 31%라 기각.
- [x] Phase 2 — 검색 축: 동일 3 레짐 학습·평가 + oracle×검색 매트릭스·evidence recall (B-6). 검색이 F1 ~0.2↓(recall 0.55), unanswerable 붕괴가 주동인, full 최강건·QLoRA 우위 상실.
- [x] Phase 2 확대 — eval N 53→202(dev 60편, 재학습 없음): 전 FT가 base 상회·oracle QLoRA 최고(단조↑)·검색 full 최고, extractive FT 이득·unanswerable 붕괴 재확인 (B-7).
- [x] abstention 실험 — 리랭커 top-1 로짓 τ=−5로 검색 unanswerable 붕괴 완화 시도: F1 +0.01(full 상쇄), 리랭커 점수는 abstention 신호로 약함 — 별도 answerability 신호 필요 (B-8).
- [x] abstention 신호 비교 — 생성기 confidence(평균 토큰 logprob)도 abstention 신호로 약함: 같은 오기권 예산서 리랭커와 동급(+0.01 안팎)·τ_g 모델별, 모델이 unanswerable서 환각(B-9).
- [x] **Phase 2 종료** — 검색 축(근거 조건) 완결. 종합: 검색 F1 ~0.2↓(recall 0.60)·unanswerable 붕괴 주동인, abstention 두 신호(리랭커·생성 confidence) 모두 약해 근접 신호로는 못 닫음 → 전용 answerability 필요 (C-2).
- [x] LoRA rank 민감도 + 시드 검증 — r∈{4,8,16,32}: rank는 F1에 영향 없음(r4=r16, r32 +0.03은 시드 아티팩트), VRAM·시간 불변·저장만 선형. 시드 변동 ±0.04가 레짐 격차와 맞먹음 (B-10).
- [x] 재현성 자산 — `format_sft --evidence-mode {oracle,search}` 분리, split manifest([manifests/](manifests/): train 42/val 8/test 78, train↔평가셋 비중복), 결과 CSV([reports/](reports/): runs/ 측정값 집계), 설정 매니페스트([experiment_manifest.json](experiment_manifest.json): 환경·하이퍼파라미터).
- [x] F1 신뢰구간 — paired bootstrap(202문, 10k, seed 0)로 레짐 CI·레짐 간 Δ CI. oracle LoRA/QLoRA의 base 상회만 유의, 레짐 상위 간·검색은 전부 CI 겹침 → "QLoRA 최고/단조↑"는 평균 순위일 뿐 (B-11). [reports/f1_ci.csv](reports/f1_ci.csv)·[f1_pairwise.csv](reports/f1_pairwise.csv).
- [x] answerability 게이트 — top_score+confidence 로지스틱 결합(B-12): 같은 오기권 예산서 결합이 단독을 못 이김(리랭커 재현·conf보다 낮음), 결합−단독 Δ CI 모두 0 포함(test unanswerable 20) → 전용 신호 과제 유지. [reports/gate_compare.csv](reports/gate_compare.csv).
