# finetune-tradeoffs-eval — QA 생성기 파인튜닝 방법 비교

> Qasper 논문 QA의 **생성 단계**를 소형 LLM(Qwen2.5-1.5B)에 파인튜닝하면서, **full FT · LoRA · QLoRA**가 품질과 비용에서 어떻게 갈리는지 같은 척도로 측정한다.

> 상태: **Phase 2 종료.** oracle·검색 두 근거 조건에서 full·LoRA·QLoRA 학습·평가 완료(미학습 202문), 개선 반복(인용 마커 정리·eval 확대) 포함. 추가 ablation: LoRA rank 민감도, F1 신뢰구간(paired bootstrap), 학습 시드 강건성(oracle 전 레짐 3시드), abstention 신호(리랭커·생성 confidence).

개발 과정은 [DEVELOPMENT_JOURNEY](DEVELOPMENT_JOURNEY.md)에, 평가 방법과 지표는 [EVALUATION](EVALUATION.md)에 정리한다.

## 한눈에

- **문제**: 자매 프로젝트 [paper-rag-eval](https://github.com/ysb2152/paper-rag-eval)에서 "gold 근거를 확보하고도 Answer-F1이 약 0.48 = 병목은 검색이 아니라 생성"임을 측정했다. 이 생성 병목을 **학습**으로 공략한다.
- **핵심 질문**: 소형 모델에서 full FT / LoRA / QLoRA가 생성 품질을 각각 얼마나 올리나, 비용(VRAM·학습 시간·학습 파라미터) 대비 트레이드오프는?
- **thesis**: "무엇이 낫다"가 아니라 **측정된 트레이드오프**. 핵심 결과 — LoRA/QLoRA는 full 대비 **학습 약 4–5배↓·VRAM 절반 이하**이면서 oracle 품질은 동급(레짐 우열은 CI·시드 양쪽에서 미결). 실무 기본값이 왜 LoRA/QLoRA인지 숫자로 드러난다.
- **스택**: 로컬 GPU(RTX 3080, 10GB)에서 `transformers`+`peft`+`bitsandbytes`+`trl`. 채점은 Qasper Answer-F1 + 로컬 NLI(paper-rag 하네스 재사용).

## 구현 범위

- **비교 축 1 — 적응 방법**: full FT · LoRA · QLoRA (동일 모델·데이터, 레짐만 격리).
- **비교 축 2 — 근거 조건**: oracle 근거(생성만 격리) · 검색 근거(end-to-end). Phase 1 oracle → Phase 2 검색으로 단계화.
- **측정**: 품질(Answer-F1·NLI 충실율/비함의율) × 비용(peak VRAM·wall-clock·trainable params). 튜닝 전 1.5B 대비 before/after, 논문 단위 held-out으로 일반화 확인.

## 표

측정 조건: RTX 3080(10GB), Qwen2.5-1.5B-Instruct, oracle 근거 + 질문 → reference 답. 학습 train 42편(114샘플), 평가 미학습 78편(202문; dev 60 + held-out 10 + val 8, 재학습 없이 확대). 상세·3조건 분해는 [EVALUATION](EVALUATION.md).

**비용 (검색 근거 학습, 42 step / 3 epoch)**

| 레짐 | wall-clock | peak VRAM | trainable params |
|---|---|---|---|
| full FT | 627.1s | 11.01 GB | 1.54B (100%) |
| LoRA | **121.5s** | 5.35 GB | 18.5M (약 1.2%) |
| QLoRA | 166.0s | **4.58 GB** | 18.5M (약 1.2%) |

LoRA·QLoRA는 trainable을 full의 약 1.2%로 줄여 물리 10GB 안에 적재 → full 대비 **wall-clock 약 4–5배↓**(LoRA 5.2배·QLoRA 3.8배). full은 10GB를 초과해 공유 메모리로 스필(7–12s/step). 메모리 최소(QLoRA) ≠ 최고 속도(LoRA).

**품질 (Answer-F1, 미학습 202문, paired bootstrap 95% CI)**

| 모델 | Answer-F1 [95% CI] | base 대비 Δ [95% CI] |
|---|---|---|
| base (튜닝 전) | 0.430 [0.38, 0.48] | — |
| full FT | 0.467 [0.41, 0.52] | +0.037 [−0.01, +0.09] |
| LoRA | 0.478 [0.42, 0.53] | +0.048 [+0.01, +0.09] |
| QLoRA | 0.482 [0.43, 0.54] | +0.053 [+0.01, +0.09] |

oracle에서 **LoRA·QLoRA의 base 상회는 유의**(Δ의 95% CI가 0을 제외), full−base는 경계(CI가 0을 살짝 포함, P(Δ>0)=0.93). 반면 **레짐 상위끼리(full/LoRA/QLoRA)는 CI가 겹쳐 우열을 가릴 수 없다**(QLoRA−LoRA Δ=+0.004 [−0.03, +0.04]) — QLoRA 평균이 가장 높지만 뚜렷한 우위로 판단하지 않는다. 이 이득은 지배 유형 extractive(52%, 0.39→0.48)가 주도한다. 53문에서 full/LoRA가 base에 −0.01 못 미치던 집계는 작은 n의 artifact였다. 위 oracle 수치는 단일 학습 시드이나, seed 42 재학습본이 B-7 수치를 ±0.01로 재현하고, 3시드 range(최대 0.037)가 레짐 평균차(≤0.009)보다 크다(B-14). 상세는 [reports/f1_pairwise.csv](reports/f1_pairwise.csv).

**품질 — oracle vs 검색 (Phase 2, 동일 202문)**

| 모델 | oracle F1 | 검색 F1 |
|---|---|---|
| base | 0.430 | 0.249 |
| full FT | 0.467 | 0.278 |
| LoRA | 0.478 | 0.265 |
| QLoRA | 0.482 | 0.275 |

근거를 oracle→검색(within-paper dense→리랭커 top-5)으로 바꾸면 전 레짐 F1이 **약 0.2↓**(evidence recall 0.60 — top-5가 gold 근거의 약 60%만 회수). 단 **검색에서는 어떤 레짐도 base와 유의차가 없다**(full−base Δ=+0.029 [−0.02, +0.08]) — oracle에서 보이던 FT 이득이 노이즈 근거에서 CI 안으로 사라진다. 레짐 평균 순위는 근거 조건에 따라 바뀌나(oracle 최고 QLoRA, 검색 최고 full) 이 역시 CI 겹침 범위의 변동이다. 상세·유형별 분해는 [EVALUATION](EVALUATION.md).

## 결론 (정직)

- **트레이드오프**: LoRA/QLoRA는 full 대비 비용 약 4–5배↓·VRAM 절반 이하로, 지배 유형 품질을 지키거나 소폭 개선한다. 실무 기본값이 왜 LoRA/QLoRA인지 숫자로 드러난다.
- **과장하지 않음**: paired bootstrap 95% CI로 보면 oracle에서 **LoRA/QLoRA의 base 상회만 유의**하고(Δ CI가 0 제외), full−base·레짐 상위 간 우열·검색의 모든 개선은 CI가 겹쳐 구분되지 않는다. 극적 개선이 아니라 **비용 대폭↓ + oracle FT의 소폭·유의한 base 상회(extractive 주도), 레짐 우열은 노이즈**가 정직한 요약이다.
- **측정 주도**: 초기 측정에서 파인튜닝이 품질을 해치는 듯 보였으나, 원인을 **근거 입력의 39%를 오염시킨 인용 마커 누출**로 진단하고, 정리 + eval 확대 + **통제 실행으로 요인을 분리**했다. 결론을 작은 n·단일 측정으로 과신하지 않고 **재학습 없이 eval을 202문으로 확대해 재검증**했다(전 레짐이 base를 상회; oracle 레짐 순위는 이후 3시드(B-14)서 노이즈 안으로 확인). 이 과정 자체가 이 저장소의 핵심이다.
- **두 번째 병목 = 검색(Phase 2)**: 근거를 oracle→검색으로 바꾸자 전 레짐 F1이 약 0.2 떨어졌다(recall 0.60). 즉 **근거 품질이 적응 방법보다 품질을 더 가른다** — paper-rag가 찾은 생성 병목 위에 검색 병목이 겹치고, 파인튜닝만으로는 닫히지 않는다. 드롭의 주동인은 unanswerable 붕괴(검색이 항상 근거를 반환해 "근거 없음"을 못 가림)다.
- **한계·노브까지 측정(과장 방지)**: unanswerable 붕괴를 닫으려 abstention을 두 신호(리랭커 점수·생성 confidence)로 시험했으나 둘 다 약해(오기권이 회복만큼 따라옴) 근접 신호로는 못 닫았다 — 전용 answerability 판단이 필요하다는 한계를 음성으로 남겼다. 또 LoRA rank(4→32)는 F1에 영향이 없고(r4가 r16과 동급, 시드 검증으로 r32의 소폭 이득은 노이즈로 확인) 비용은 저장 파라미터에만 선형이다. 시드 변동(±0.04)이 레짐 간 격차와 맞먹는다. oracle 전 레짐을 3시드로 재학습해도 레짐 평균차(≤0.009)가 시드 range 안이고 단일시드 수치가 ±0.01로 재현돼(B-14), 레짐 우열이 노이즈라는 결론을 **eval 리샘플(CI)과 학습 시드 두 축에서** 확인했다 — 작은 차이를 단일 측정으로 과신하지 않는다.

## 재현 (실행법)

환경: Python + CUDA GPU. `pip install -r requirements.txt` (transformers 4.46.3 / torch 2.5.1+cu121 고정). Qasper 논문 JSON을 `data/papers`(학습 풀)·`data/papers-heldout`(평가 전용)에 둔다. cp949 콘솔이면 각 명령 앞에 `PYTHONIOENCODING=utf-8`.

```bash
# 1) SFT 학습 데이터 생성 (train 분할의 단일주석 질문 → chat 샘플; --evidence-mode 기본 search, oracle 선택)
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
```bash
# 5) F1 신뢰구간 — runs/eval78의 질문별 f1로 paired bootstrap(레짐 CI·레짐 간 Δ CI)
python -m scripts.bootstrap_ci
```

정확한 설정값(모델·패키지 버전·학습/평가 하이퍼파라미터)은 [experiment_manifest.json](experiment_manifest.json), 논문 분할(train 42/val 8/test 78, train과 평가셋 비중복)은 [manifests/](manifests/), 레짐×근거별 측정 결과는 [reports/quality.csv](reports/quality.csv)·[reports/cost.csv](reports/cost.csv), F1 신뢰구간은 [reports/f1_ci.csv](reports/f1_ci.csv)·[reports/f1_pairwise.csv](reports/f1_pairwise.csv)에 있다.

## 데이터·라이선스

평가·학습: **Qasper** (Allen Institute for AI). 평가 하네스(Answer-F1·NLI)는 [paper-rag-eval](https://github.com/ysb2152/paper-rag-eval)에서 가져와 self-contained로 포함(출처 명시). 코드는 MIT 라이선스([LICENSE](LICENSE)).
