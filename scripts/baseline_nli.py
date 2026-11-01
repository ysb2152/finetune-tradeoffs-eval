import argparse
import json
from pathlib import Path

from src.eval.nli import MODEL_NAME, entailment, load_nli

# 답이 보류('Unanswerable')인지 판정

def is_abstention(prediction: str) -> bool:
    return prediction.strip().lower().startswith("unanswerable")


# 생성 결과 JSON의 답을 근거에 대해 NLI 채점, 충실율·비함의율을 낸다(보류는 제외해 따로 센다)

def run(run_path: str, max_length: int = 512) -> dict:
    report = json.loads(Path(run_path).read_text(encoding="utf-8"))
    model, tokenizer, device = load_nli()

    scored = []
    abstained = 0
    skipped_no_evidence = 0
    for row in report["rows"]:
        prediction = row["prediction"]
        evidence = row.get("evidence") or []
        if is_abstention(prediction):
            abstained += 1
            continue
        if not evidence:
            skipped_no_evidence += 1
            continue
        result = entailment("\n\n".join(evidence), prediction, model, tokenizer, device, max_length)
        scored.append({
            "question_id": row["question_id"],
            "prediction": prediction,
            "label": result["label"],
            "entailment": result["entailment"],
        })

    count = len(scored)
    faithful = sum(1 for row in scored if row["label"] == "entailment")
    contradiction = sum(1 for row in scored if row["label"] == "contradiction")
    return {
        "source": run_path,
        "nli_model": MODEL_NAME,
        "scored_answers": count,
        "abstained": abstained,
        "skipped_no_evidence": skipped_no_evidence,
        "faithful_rate": faithful / count if count else None,
        "non_entailment_rate": (count - faithful) / count if count else None,
        "contradiction_rate": contradiction / count if count else None,
        "mean_entailment": sum(row["entailment"] for row in scored) / count if count else None,
        "answers": scored,
    }


# CLI: 생성 결과 JSON을 받아 충실성 리포트를 JSON으로 출력(선택: --out 저장)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_path", help="generate/retrieved 실험이 저장한 결과 JSON")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--out", type=Path, help="결과 JSON 저장 경로")
    args = parser.parse_args()

    result = run(args.run_path, args.max_length)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
