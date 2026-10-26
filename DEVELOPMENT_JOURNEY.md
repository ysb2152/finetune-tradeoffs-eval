# finetune-tradeoffs-eval — 개발 일지

Qasper 논문 QA의 **생성 단계**를 소형 LLM(Qwen2.5-1.5B)에 파인튜닝하면서, **full FT · LoRA · QLoRA** 세 적응 방법의 품질/비용 트레이드오프를 같은 척도로 측정한다. 사실·정량 로그만 남긴다(어필 서술은 README).

## 문서 맵

- [README.md](README.md) — 포트폴리오 개요(어필용)
- [EVALUATION.md](EVALUATION.md) — 평가 방법·상세 지표 표

문서 구조·기록 관례는 `devlog-convention` 관례를 따른다(Part 0/A/B/C, 상태표, 체크박스 로드맵, ID 의사결정).

## 상태표 (최신 위)

| 날짜(ISO) | 분류 | 내용 | 관련ID |
|---|---|---|---|
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

## Part C — 회고

### C-1 — 배운 것

- **비용 우위의 메커니즘은 "파라미터가 적다"가 아니라 "물리 VRAM에 들어간다"였다.** 초기 가정은 LoRA/QLoRA가 단순히 조금 가볍다는 것이었는데, 실제로는 full이 10GB를 **초과**해 공유 메모리로 스필되며 7~12s/step으로 느려졌고, 물리 VRAM 안에 들어간 LoRA·QLoRA가 ~5배 빨랐다. 또 메모리가 가장 작은 QLoRA가 가장 빠르지는 않았다(4-bit 역양자화 비용).
- **train_loss는 일반화를 대변하지 못한다.** 세 레짐의 train_loss(0.62~0.63)는 비슷했지만 held-out 품질은 모두 base보다 낮았다. 집계 F1 하락(0.543→0.47~0.52)도 유형별로 분해하니 지배 유형은 불변이고 소수 유형(boolean n2·unanswerable n5)에 회귀가 집중된 것이었다 — 작은 n에서는 1~2문이 지표를 크게 흔든다.

(로드맵은 아래.)

---

## 로드맵

- [x] P0 확정 및 저장소 스캐폴딩
- [x] 평가 하네스 이식(answer.py·nli.py·Qasper ingest) + 베이스라인(튜닝 전 1.5B) 측정 (B-1)
- [x] 데이터 포맷팅 파이프라인(oracle 근거 → SFT 샘플) + 분할 확정 (B-2)
- [x] Phase 1 — oracle 축: full FT / LoRA / QLoRA 학습·평가·비용 측정 (B-3·B-4)
- [x] Phase 1 비교표(품질×비용) + held-out 일반화 + 유형별 진단 ([EVALUATION.md](EVALUATION.md))
- [ ] Phase 2 — 검색 축: 동일 3 레짐
- [ ] 개선 반복: SFT 데이터 유형 균형(boolean/unanswerable 보강)·BIBREF 제거·eval n 확대
