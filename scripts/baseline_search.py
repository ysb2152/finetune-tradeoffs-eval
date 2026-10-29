import argparse
import json
from pathlib import Path

from src.data.sample import single_annotation_questions
from src.generate.qwen import MODEL_NAME, load_generator, generate_answer
from src.eval.answer import answer_f1_detail
from src.retrieve.embed import load_embedder
from src.retrieve.rerank import load_reranker
from src.retrieve.pipeline import within_paper
from src.data.sample import clean_refs



def run(papers_dir: Path , max_new_tokens: int = 64, model_path=None, adapter_path=None,candidate_n=20 ,top_k=5 , abstain_threshold: float = None ) -> dict:

    model, tokenizer = load_generator(model_name=model_path or MODEL_NAME, adapter_path=adapter_path)
    embed_bundle = load_embedder()
    rerank_bundle = load_reranker()
    rows = []
    papers=set()

    for (paper,question) in single_annotation_questions(papers_dir.glob("*.json")):

        gold_ids = []
        for answer in question.answers:
            for evidence in answer.evidence:
                gold_ids.extend(evidence.paragraph_ids)
        gold_ids=set(gold_ids)
        papers.add(paper.id)
        results = within_paper(paper.paragraphs,question.text,embed_bundle,rerank_bundle,candidate_n,top_k)
        if results:
            top_score=results[0].score
        else:
            top_score=float("-inf")
        if abstain_threshold and top_score < abstain_threshold:
            abstained=True
        else:
            abstained=False
        if abstained:
            evidence=[]
        else:
            evidence = [clean_refs(r.paragraph.text) for r in results]
        retrieved_ids = [r.paragraph.id for r in results]
        cnt=len(gold_ids.intersection(set(retrieved_ids)))

        if len(gold_ids) !=0:

            evidence_recall= cnt/ len(gold_ids)

        else:
            evidence_recall=None

        prediction = generate_answer(question.text, evidence, model, tokenizer, max_new_tokens)
        f1, matched_type = answer_f1_detail(prediction, question)
        rows.append({
                "paper_id": paper.id,
                "question_id": question.id,
                "question": question.text,
                "evidence": evidence,
                "retrieved_ids": retrieved_ids,
                "top_score" : top_score,
                "abstained" : abstained,
                "evidence_recall": evidence_recall,
                "prediction": prediction,
                "matched_type": matched_type,
                "has_evidence": bool(evidence),
                "f1": f1,
            })

    count = len(rows)
    recall_list = [row["evidence_recall"] for row in rows if row["evidence_recall"] is not None]
    types = sorted({row["matched_type"] for row in rows})


    if recall_list:
        mean_evidence_recall = sum(recall_list) / len(recall_list)
    else:
        mean_evidence_recall=None

    by_type = {
        answer_type: {
            "count": sum(1 for row in rows if row["matched_type"] == answer_type),
            "mean_f1": sum(row["f1"] for row in rows if row["matched_type"] == answer_type)
            / sum(1 for row in rows if row["matched_type"] == answer_type),
        }
        for answer_type in types
    }
    abstained_average = {

        "unanswerable_abstained": sum(1 for row in rows if row["matched_type"] == "unanswerable" and row["abstained"]) / sum(1 for row in rows if row["matched_type"] == "unanswerable"),
        "answerable_abstained": sum (1 for row in rows if row["matched_type"] != "unanswerable" and row["abstained"]) / sum(1 for row in rows if row["matched_type"] != "unanswerable")

    }
    return {
        "model": str(adapter_path or model_path or MODEL_NAME),
        "papers": len(papers),
        "questions": count,
        "max_new_tokens": max_new_tokens,
        "mean_f1": sum(row["f1"] for row in rows) / count if count else None,
        "mean_evidence_recall": mean_evidence_recall,
        "by_type": by_type,
        "abstained_type": abstained_average ,
        "rows": rows,
    }




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("papers_dir", type=Path, help="Qasper 논문 폴더")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--model-path", type=Path, help="full 파인튜닝 모델 디렉터리")
    parser.add_argument("--adapter-path", type=Path, help="LoRA/QLoRA 어댑터 디렉터리")
    parser.add_argument("--out", type=Path, help="결과 JSON 저장 경로")
    parser.add_argument("--abstain-threshold", type=float, default=None)
    args = parser.parse_args()

    report = run(args.papers_dir, args.max_new_tokens, args.model_path, args.adapter_path,abstain_threshold=args.abstain_threshold)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
