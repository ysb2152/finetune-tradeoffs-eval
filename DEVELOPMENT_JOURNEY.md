# finetune-tradeoffs-eval — 개발 일지

Qasper 논문 QA의 **생성 단계**를 소형 LLM(Qwen2.5-1.5B)에 파인튜닝하면서, **full FT · LoRA · QLoRA** 세 적응 방법의 품질/비용 트레이드오프를 같은 척도로 측정한다. 사실·정량 로그만 남긴다(어필 서술은 README).

## 문서 맵

- [README.md](README.md) — 포트폴리오 개요(어필용)
- [EVALUATION.md](EVALUATION.md) — 평가 방법·상세 지표 표

문서 구조·기록 관례는 `devlog-convention` 관례를 따른다(Part 0/A/B/C, 상태표, 체크박스 로드맵, ID 의사결정).

## 상태표 (최신 위)

| 날짜(ISO) | 분류 | 내용 | 관련ID |
|---|---|---|---|
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

(B-1부터 시간순으로 채운다.)

## Part C — 회고

(배운 것 + 로드맵.)

---

## 로드맵

- [ ] P0 확정 및 저장소 스캐폴딩
- [ ] 평가 하네스 이식(answer.py·nli.py·Qasper ingest) + 베이스라인(튜닝 전 1.5B) 측정
- [ ] 데이터 포맷팅 파이프라인(oracle/검색 근거 → SFT 샘플) + 분할 확정
- [ ] Phase 1 — oracle 축: full FT / LoRA / QLoRA 학습·평가·비용 측정
- [ ] Phase 2 — 검색 축: 동일 3 레짐
- [ ] 비교표(품질×비용) + held-out 일반화 + 회고
