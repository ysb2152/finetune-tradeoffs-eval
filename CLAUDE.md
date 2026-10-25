# finetune-tradeoffs-eval

Qasper 논문 QA의 **생성 단계**를 소형 LLM(Qwen2.5-1.5B-Instruct)에 파인튜닝하며 **full FT · LoRA · QLoRA**의 품질/비용 트레이드오프를 같은 척도로 측정하는 평가 주도 프로젝트. 자매 프로젝트 [paper-rag-eval](https://github.com/ysb2152/paper-rag-eval)이 측정한 "생성 병목"을 학습으로 공략한다.

## 문서 맵

- [README.md](README.md) — 포트폴리오 개요(어필용)
- [DEVELOPMENT_JOURNEY.md](DEVELOPMENT_JOURNEY.md) — 개발 일지(사실·정량, Part 0 결정 P0-1~8)
- [EVALUATION.md](EVALUATION.md) — 평가 방법·지표

문서·기록은 `devlog-convention` 관례를 따른다(Part 0/A/B/C, 상태표, 체크박스 로드맵, ID 의사결정). 구조를 이 파일에 다시 적지 않는다.

## 확정 스펙 (요약 — 상세·근거는 DEVELOPMENT_JOURNEY Part 0)

- **문제(P0-1)**: gold 근거를 확보하고도 Answer-F1 ~0.48에 머무는 생성 병목을 학습으로 공략. thesis = "무엇이 낫다"가 아니라 **측정된 트레이드오프**.
- **평가셋·채점(P0-2)**: Qasper. 품질 = Answer-F1 + 로컬 NLI(충실율/비함의율), 비용 = peak VRAM·wall-clock·trainable params.
- **베이스 모델(P0-3)**: Qwen2.5-1.5B-Instruct. 1.5B full FT가 10GB에서 OOM이면 1B/0.5B 폴백(첫 실측으로 결정).
- **비교 축(P0-4·5)**: 방법 = full FT / LoRA / QLoRA, 근거 = oracle / 검색. Phase 1 oracle → Phase 2 검색으로 단계화.
- **하네스 재사용(P0-7)**: paper-rag `src/eval/answer.py`·`src/eval/nli.py`·Qasper ingest를 복사해 self-contained로. 출처 명시.
- **스택(P0-8)**: Python + RTX 3080(10GB), `transformers`+`peft`+`bitsandbytes`+`trl`. 버전 고정.

## 공개 타이밍

GitHub 공개 저장소는 Phase 1 결과가 나온 뒤 생성한다. 그 전까지 로컬 커밋으로만 추적한다.

## 코드·커밋 규칙

- 미측정 수치를 문서에 넣지 않는다. 공개 기록엔 실제 결정·검증 결과만.
- 커밋 전 파일별 범위·공개 문구·커밋 메시지 초안을 확인받는다. 서명은 세션 지침(co-author)을 따른다.
- 개인 학습 메모·전략은 gitignored `local-notes/`에 둔다.
