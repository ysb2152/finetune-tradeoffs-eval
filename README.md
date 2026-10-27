# finetune-tradeoffs-eval — QA 생성기 파인튜닝 방법 비교

> Qasper 논문 QA의 **생성 단계**를 소형 LLM(Qwen2.5-1.5B)에 파인튜닝하면서, **full FT · LoRA · QLoRA**가 품질과 비용에서 어떻게 갈리는지 같은 척도로 측정한다.

> 상태: Phase 1(oracle 근거, 3 레짐) + 개선 반복(마커 정리·eval 확대) 완료. Phase 2(검색 근거) 예정.

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

## 표

측정 조건: RTX 3080(10GB), Qwen2.5-1.5B-Instruct, oracle 근거 + 질문 → reference 답. 학습 train 42편(114샘플), 평가 미학습 held-out+val 18편(53문). 상세·3조건 분해는 [EVALUATION](EVALUATION.md).

**비용 (학습, 42 step / 3 epoch)**

| 레짐 | wall-clock | peak VRAM | trainable params |
|---|---|---|---|
| full FT | 523.6s | 10.85 GB | 1.54B (100%) |
| LoRA | **109.3s** | 5.12 GB | 18.5M (~1.2%) |
| QLoRA | 142.8s | **4.34 GB** | 18.5M (~1.2%) |

LoRA·QLoRA는 trainable을 full의 ~1.2%로 줄여 물리 10GB 안에 적재 → full 대비 **wall-clock ~5배↓**. full은 10GB를 초과해 공유 메모리로 스필(7~12s/step). 메모리 최소(QLoRA) ≠ 최고 속도(LoRA).

**품질 (Answer-F1, 미학습 53문)**

| 모델 | Answer-F1 | base 대비 |
|---|---|---|
| base (튜닝 전) | 0.529 | — |
| full FT | 0.514 | −0.015 |
| LoRA | 0.515 | −0.014 |
| QLoRA | **0.556** | **+0.027** |

지배 유형 **extractive(전체의 62%)에서 파인튜닝이 base를 개선**(0.43 → 0.46~0.50), QLoRA는 종합적으로 base를 상회한다.

## 결론 (정직)

- **트레이드오프**: LoRA/QLoRA는 full 대비 비용 ~5배↓·VRAM 절반 이하로, 지배 유형 품질을 지키거나 소폭 개선한다. 실무 기본값이 왜 LoRA/QLoRA인지 숫자로 드러난다.
- **과장하지 않음**: 전체 F1에서 full/LoRA는 base에 근소하게 못 미치고(−0.01), QLoRA만 base를 넘는다. 극적 개선이 아니라 **비용 대폭↓ + 지배 유형 품질 유효**가 정직한 요약이다.
- **측정 주도**: 초기 측정에서 파인튜닝이 품질을 해치는 듯 보였으나, 원인을 **근거 입력의 39%를 오염시킨 인용 마커 누출**로 진단하고, 정리 + eval 확대 + **통제 실행으로 요인을 분리**했다. 결론을 작은 n·단일 측정으로 과신하지 않는 과정 자체가 이 저장소의 핵심이다.

## 재현 (실행법)

환경: Python + CUDA GPU. `pip install -r requirements.txt` (transformers 4.46.3 / torch 2.5.1+cu121 고정). Qasper 논문 JSON을 `data/papers`(학습 풀)·`data/papers-heldout`(평가 전용)에 둔다. cp949 콘솔이면 각 명령 앞에 `PYTHONIOENCODING=utf-8`.

```bash
# 1) SFT 학습 데이터 생성 (train 분할의 단일주석 질문 → chat 샘플)
python -m scripts.format_sft data/papers --out data/sft/train.jsonl
```
```bash
# 2) 레짐별 학습 (full / lora / qlora), 가중치·metrics는 --out 디렉터리에
python -m scripts.train_sft --regime lora data/sft/train.jsonl --out runs/lora
```
```bash
# 3) 생성 품질(Answer-F1) 평가 — 튜닝 전 base는 모델 인자 없이,
#    full은 --model-path, LoRA/QLoRA는 --adapter-path
python -m scripts.baseline_oracle data/papers-heldout --adapter-path runs/lora --out runs/eval/lora_oracle.json
```
```bash
# 4) 충실성(NLI) 평가 — 3)의 출력 JSON을 입력으로
python -m scripts.baseline_nli runs/eval/lora_oracle.json --out runs/eval/lora_nli.json
```

## 데이터·라이선스

평가·학습: **Qasper** (Allen Institute for AI). 평가 하네스(Answer-F1·NLI)는 [paper-rag-eval](https://github.com/ysb2152/paper-rag-eval)에서 가져와 self-contained로 포함(출처 명시). 코드는 MIT 라이선스([LICENSE](LICENSE)).
