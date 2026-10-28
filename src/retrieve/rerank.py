import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.ingest.qasper import Paragraph
from src.retrieve.embed import SearchResult

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


# bge-reranker 로드. device 미지정이면 cuda 있으면 cuda 없으면 cpu

def load_reranker(model_name: str = MODEL_NAME,device:str = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if device is None:
        device= "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer, device


# (질문, 문단) 쌍을 교차 인코더에 넣어 관련도 점수로 다시 정렬, top_k만 남긴다

def rerank(
    query: str,
    candidates: list[Paragraph],
    model,
    tokenizer,
    device: str,
    top_k: int = 5,
    max_length: int = 512,
    batch_size: int = 8,
) -> list[SearchResult]:
    if not candidates:
        return []

    pairs = [(query, paragraph.text) for paragraph in candidates]
    scores: list[float] = []
    # 후보 전체를 배치로 나눠 추론한다. 배치 크기는 한 번에 GPU로 보내는 쌍의 수일 뿐,
    # 검토하는 후보 수(len(candidates))와는 다르다.
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs).logits.view(-1).float()
            scores.extend(logits.tolist())

    # 점수가 같으면 후보로 들어온 순서(dense 순위)를 유지한다.
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    return [SearchResult(candidates[i], scores[i]) for i in order[:top_k]]



