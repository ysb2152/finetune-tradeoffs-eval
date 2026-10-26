# 이식 출처: paper-rag-eval (ysb2152/paper-rag-eval) src/eval/nli.py 를 그대로 복사했다.
# 근거는 P0-7(하네스 재사용). 채점 모델은 paper-rag와 동일 체크포인트(P0-8).
# 로컬 NLI로 생성한 답이 근거에 함의되는지(충실성)를 판정한다. Answer-F1은 reference와의
# 겹침만 보니, 답이 근거에서 실제로 나온 말인지(환각 아닌지)는 따로 잰다. 근거를 전제, 답을
# 가설로 두고 entailment면 충실·neutral이면 근거 밖에서 지어냄·contradiction이면 근거와 반대.
# 방향 주의 — 근거가 답보다 정보가 많아 반대로(답→근거) 보면 충실한 답도 neutral로 오판된다.
# 라벨 순서는 모델마다 다르므로 config.id2label을 읽어 쓴다.

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


# DeBERTa MNLI 모델 로드

def load_nli(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, use_safetensors=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer, device


# 전제(근거)가 가설(답)을 함의하는지 판정. 라벨별 확률과 최상위 라벨을 돌려준다

@torch.no_grad()
def entailment(
    premise: str,
    hypothesis: str,
    model,
    tokenizer,
    device: str,
    max_length: int = 512,
) -> dict:
    inputs = tokenizer(
        premise, hypothesis, truncation=True, max_length=max_length, return_tensors="pt"
    ).to(device)
    probs = torch.softmax(model(**inputs).logits[0], dim=-1)
    label_probs = {
        model.config.id2label[i].lower(): float(probs[i]) for i in range(probs.shape[0])
    }
    top_label = max(label_probs, key=label_probs.get)
    return {
        "label": top_label,
        "entailment": label_probs.get("entailment", 0.0),
        "probs": label_probs,
    }
