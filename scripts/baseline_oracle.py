import argparse
import json
from pathlib import Path

from src.data.sample import single_annotation_questions, oracle_evidence_paragraphs
from src.generate.qwen import MODEL_NAME, load_generator, generate_answer
from src.eval.answer import answer_f1_detail

# oracle 근거로 1.5B가 답을 생성해 유형별·전체 평균 F1을 낸다

def run(papers_dir: Path , max_new_tokens: int = 64) -> dict:
    model, tokenizer = load_generator()

    rows = []
    papers=set()

    for (paper,question) in single_annotation_questions(papers_dir.glob("*.json")):
        papers.add(paper.id)
        evidence = oracle_evidence_paragraphs(question)
        prediction = generate_answer(question.text, evidence, model, tokenizer, max_new_tokens)
        f1, matched_type = answer_f1_detail(prediction, question)
        rows.append({
                "paper_id": paper.id,
                "question_id": question.id,
                "question": question.text,
                "evidence": evidence,
                "prediction": prediction,
                "matched_type": matched_type,
                "has_evidence": bool(evidence),
                "f1": f1,
            })

    count = len(rows)
    types = sorted({row["matched_type"] for row in rows})
    by_type = {
        answer_type: {
            "count": sum(1 for row in rows if row["matched_type"] == answer_type),
            "mean_f1": sum(row["f1"] for row in rows if row["matched_type"] == answer_type)
            / sum(1 for row in rows if row["matched_type"] == answer_type),
        }
        for answer_type in types
    }

    return {
        "model": MODEL_NAME,
        "papers": len(papers),
        "questions": count,
        "max_new_tokens": max_new_tokens,
        "mean_f1": sum(row["f1"] for row in rows) / count if count else None,
        "by_type": by_type,
        "rows": rows,
    }


# CLI: 논문들 받아 oracle 생성 리포트를 JSON으로 출력(선택: --out 저장)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("papers_dir", type=Path, help="Qasper 논문 폴더")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", type=Path, help="결과 JSON 저장 경로")
    args = parser.parse_args()

    report = run(args.papers_dir, args.max_new_tokens)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
