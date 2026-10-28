import torch
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass
from src.ingest.qasper import Paragraph

@dataclass
class SearchResult:
    paragraph: Paragraph
    score: float

MODEL_NAME = "BAAI/bge-m3"


# bge-m3 임베더 로드. device 미지정이면 cuda 있으면 cuda 없으면 cpu

def load_embedder(model_name: str = MODEL_NAME,device:str=None):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if device is None:
        device="cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer, device

# 문장 목록을 정규화된 dense 벡터로 바꾼다. 반환은 (문장 수, 차원) 텐서

@torch.no_grad()
def encode(
    texts: list[str],
    model,
    tokenizer,
    device: str,
    max_length: int = 512,
    batch_size: int = 8,
) -> torch.Tensor:
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        cls = model(**inputs).last_hidden_state[:, 0]
        cls = torch.nn.functional.normalize(cls, p=2, dim=1)
        vectors.append(cls.cpu())
    return torch.cat(vectors, dim=0)


# 문단들을 임베딩해 질문 벡터와 코사인 유사도 top_k를 SearchResult로 돌려준다

def search_embed(
    paragraphs: list[Paragraph],
    query: str,
    model,
    tokenizer,
    device: str,
    top_k: int = 5,
    max_length: int = 512,
    batch_size: int = 8,
) -> list[SearchResult]:
    if top_k < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")

    documents = [paragraph for paragraph in paragraphs if paragraph.text.strip()]
    if not documents:
        return []

    doc_vectors = encode(
        [paragraph.text for paragraph in documents],
        model,
        tokenizer,
        device,
        max_length,
        batch_size,
    )
    query_vector = encode([query], model, tokenizer, device, max_length, batch_size)[0]

    scores = doc_vectors @ query_vector
    # 점수가 같으면 원문 순서를 유지해 재실행 시 순위가 바뀌지 않게 한다.
    order = sorted(range(len(documents)), key=lambda i: -scores[i])
    return [SearchResult(documents[i], float(scores[i])) for i in order[:top_k]]

