import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import argparse
from pathlib import Path

MODEL_NAME = "microsoft/deberta-large-mnli"


def load_signal_module(model_name: str = MODEL_NAME, device: str = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer, device


def mnli_scoring(modes):
    pairs = []
    question_ids = []
    gold_unanswerables = []
    batch_size = 8
    max_length = 512
    model, tokenizer, device = load_signal_module()
    evidence=[]

    with open("data/phase3/retrieval_dump.jsonl", encoding="utf-8") as f:
        for retrieval_dump in f:
            dicts = json.loads(retrieval_dump)


            for ele in dicts["evidence"]:
                if modes == "template":
                    pairs.append((ele["text"], "This passage answers the question: "+dicts["question"]))
                else:
                    pairs.append((ele["text"],dicts["question"]))
            evidence.append(len(dicts["evidence"]))
            question_ids.append(dicts["question_id"])
            gold_unanswerables.append(dicts["gold_unanswerable"])

    scores = []

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
            logits = model(**inputs).logits.softmax(dim=-1)
            ent_idx=model.config.label2id["ENTAILMENT"]
            scores.extend(logits[:,ent_idx].tolist())

    return_list = []
    cursor=0
    for i in range(len(question_ids)):
        return_dict = dict()
        return_dict["mnli_max"] = max(scores[cursor:cursor+evidence[i]])
        return_dict["mnli_mean"] = sum(scores[cursor:cursor+evidence[i]]) / evidence[i]
        return_dict["question_id"] = question_ids[i]
        return_dict["gold_unanswerable"] = gold_unanswerables[i]
        return_list.append(return_dict)
        cursor+=evidence[i]
    return return_list


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", type=str, choices=["plain","template"], required=True)
    parser.add_argument("--out", type=Path, help="결과 JSON 저장 경로", required=True)

    args = parser.parse_args()

    results = mnli_scoring(args.modes)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"questions={len(results)}  out: {args.out}")


if __name__ == "__main__":
    main()