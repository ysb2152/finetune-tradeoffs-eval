import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
from scripts.signal_utils import aggregate_by_counts,build_pair

MODEL_NAME = "cross-encoder/qnli-electra-base"



def load_signal_module(model_name: str = MODEL_NAME,device:str = None):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if device is None:
        device= "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer, device




def qnli_scoring():

    pairs=[]
    question_ids=[]
    gold_unanswerables=[]
    batch_size=8
    max_length=512
    model, tokenizer, device = load_signal_module()
    evidence=[]
    with open("data/phase3/retrieval_dump.jsonl",encoding="utf-8") as f:
        for retrieval_dump in f:
            dicts=json.loads(retrieval_dump)
            evidence.append(len(dicts["evidence"]))
            for ele in dicts["evidence"]:
                pairs.append(build_pair(dicts["question"], ele["text"], "qnli"))

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
            logits = model(**inputs).logits.view(-1).float()
            scores.extend(logits.tolist())

    return_list=[]
    max_and_sum_list = aggregate_by_counts(scores, evidence)
    for i in range(len(question_ids)):
        return_dict=dict()
        return_dict["qnli_max"],return_dict["qnli_mean"]=max_and_sum_list[i]
        return_dict["question_id"]=question_ids[i]
        return_dict["gold_unanswerable"]=gold_unanswerables[i]
        return_list.append(return_dict)

    return return_list


def main():
    results = qnli_scoring()
    with open("data/phase3/qnli_scores.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"questions={len(results)}  out: data/phase3/qnli_scores.jsonl")


if __name__ == "__main__":
    main()