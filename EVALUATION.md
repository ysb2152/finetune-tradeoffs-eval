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

## 결과

_측정되는 대로 B-로그 번호와 함께 표로 채운다._
