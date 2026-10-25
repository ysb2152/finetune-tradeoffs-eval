# finetune-tradeoffs-eval — QA 생성기 파인튜닝 방법 비교

> Qasper 논문 QA의 **생성 단계**를 소형 LLM(Qwen2.5-1.5B)에 파인튜닝하면서, **full FT · LoRA · QLoRA**가 품질과 비용에서 어떻게 갈리는지 같은 척도로 측정한다.

> ⚠️ 진행 중 — 기술 결정(Part 0) 확정 단계. 아직 측정 결과 없음. 수치는 측정되는 대로 채운다.

개발 과정은 [DEVELOPMENT_JOURNEY](DEVELOPMENT_JOURNEY.md)에, 평가 방법과 지표는 [EVALUATION](EVALUATION.md)에 정리한다.

## 한눈에

- **문제**: 자매 프로젝트 [paper-rag-eval](https://github.com/ysb2152/paper-rag-eval)에서 "gold 근거를 확보하고도 Answer-F1이 ~0.48 = 병목은 검색이 아니라 생성"임을 측정했다. 이 생성 병목을 **학습**으로 공략한다.
- **핵심 질문**: 소형 모델에서 full FT / LoRA / QLoRA가 생성 품질을 각각 얼마나 올리나, 비용(VRAM·학습 시간·학습 파라미터) 대비 트레이드오프는?
- **thesis**: "무엇이 낫다"가 아니라 **측정된 트레이드오프**. 실무 기본값이 왜 LoRA/QLoRA인지 숫자로 보인다.
- **스택**: 로컬 GPU(RTX 3080, 10GB)에서 `transformers`+`peft`+`bitsandbytes`+`trl`. 채점은 Qasper Answer-F1 + 로컬 NLI(paper-rag 하네스 재사용).

## 구현 범위

- **비교 축 1 — 적응 방법**: full FT · LoRA · QLoRA (동일 모델·데이터, 레짐만 격리).
- **비교 축 2 — 근거 조건**: oracle 근거(생성만 격리) · 검색 근거(end-to-end). Phase 1 oracle → Phase 2 검색으로 단계화.
- **측정**: 품질(Answer-F1·NLI 충실율/비함의율) × 비용(peak VRAM·wall-clock·trainable params). 튜닝 전 1.5B 대비 before/after, 논문 단위 held-out으로 일반화 확인.

## 핵심 성과표

_측정 후 채운다. 미측정 수치는 넣지 않는다._

## 데이터·라이선스

평가·학습: **Qasper** (Allen Institute for AI). 평가 하네스(Answer-F1·NLI)는 [paper-rag-eval](https://github.com/ysb2152/paper-rag-eval)에서 가져와 self-contained로 포함(출처 명시).
