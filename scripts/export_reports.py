"""공개용 리포트 CSV를 runs/eval78 결과에서 만든다. 재학습·재평가 없이 기존
질문별 결과만 집계한다. 세 가지를 낸다.
- eval_per_question.csv: 질문×레짐×근거별 F1(+검색 근거는 answerability 신호)
- f1_by_type_149.csv: pristine 149문의 답 유형별 평균 F1
- oracle_seed_robustness.csv: oracle 3시드(42/100/202) 레짐별 F1과 요약(B-14)
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

REGIMES = ["base", "full", "lora", "qlora"]
EVIDENCES = ["oracle", "search"]
SEEDS = [42, 100, 202]


def load_rows(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["rows"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", default="runs/eval78")
    parser.add_argument("--seed-dir", default="runs/eval78_seeds")
    parser.add_argument("--papers", default="manifests/gate_split.json")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    seed_dir = Path(args.seed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    split = json.loads(Path(args.papers).read_text(encoding="utf-8"))
    pristine = set(split["test"]["papers"])

    # 1) 질문별 flat CSV. 검색 근거는 _search_conf.json에서 answerability 신호를 붙인다.
    per_q = []
    for ev in EVIDENCES:
        conf_by_qid = {}
        if ev == "search":
            for rg in REGIMES:
                for r in load_rows(eval_dir / f"{rg}_search_conf.json"):
                    conf_by_qid[(rg, r["question_id"])] = r
        for rg in REGIMES:
            for r in load_rows(eval_dir / f"{rg}_{ev}.json"):
                conf = conf_by_qid.get((rg, r["question_id"]), {})
                per_q.append({
                    "paper_id": r["paper_id"],
                    "question_id": r["question_id"],
                    "regime": rg,
                    "evidence": ev,
                    "answer_type": r["matched_type"],
                    "f1": round(r["f1"], 4),
                    "in_pristine149": r["paper_id"] in pristine,
                    "top_score": round(conf["top_score"], 4) if "top_score" in conf else "",
                    "gen_confidence": round(conf["gen_confidence"], 4) if "gen_confidence" in conf else "",
                    "abstained": conf.get("abstained", ""),
                })
    write_csv(out_dir / "eval_per_question.csv", per_q)

    # 2) pristine 149문의 유형별 평균 F1.
    by_type = []
    for ev in EVIDENCES:
        for rg in REGIMES:
            rows = [r for r in load_rows(eval_dir / f"{rg}_{ev}.json")
                    if r["paper_id"] in pristine]
            buckets = {}
            for r in rows:
                buckets.setdefault(r["matched_type"], []).append(r["f1"])
            for mtype in sorted(buckets):
                vals = buckets[mtype]
                by_type.append({
                    "regime": rg, "evidence": ev, "answer_type": mtype,
                    "n": len(vals), "mean_f1": round(sum(vals) / len(vals), 4),
                })
    write_csv(out_dir / "f1_by_type_149.csv", by_type)

    # 3) oracle 3시드 강건성(202문, B-14). base는 시드 무관이라 단일값.
    seed_rows = []
    for rg in REGIMES:
        if rg == "base":
            f1 = json.loads((seed_dir / "base_oracle.json").read_text(encoding="utf-8"))["mean_f1"]
            seed_rows.append({"regime": rg, "seed": "all", "answer_f1": round(f1, 4),
                              "mean": round(f1, 4), "std": "", "range": ""})
            continue
        vals = []
        for s in SEEDS:
            f1 = json.loads((seed_dir / f"{rg}_s{s}_oracle.json").read_text(encoding="utf-8"))["mean_f1"]
            vals.append(f1)
            seed_rows.append({"regime": rg, "seed": s, "answer_f1": round(f1, 4),
                              "mean": "", "std": "", "range": ""})
        seed_rows.append({
            "regime": rg, "seed": "summary",
            "answer_f1": "", "mean": round(statistics.mean(vals), 4),
            "std": round(statistics.stdev(vals), 4), "range": round(max(vals) - min(vals), 4),
        })
    write_csv(out_dir / "oracle_seed_robustness.csv", seed_rows)

    print(f"eval_per_question.csv: {len(per_q)} rows | "
          f"f1_by_type_149.csv: {len(by_type)} rows | "
          f"oracle_seed_robustness.csv: {len(seed_rows)} rows")


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
