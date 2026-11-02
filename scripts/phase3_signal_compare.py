"""Phase 3a 신호 비교: 리랭커 점수 기준선 vs QNLI·MNLI 전용 신호.

retrieval_dump.jsonl(리랭커 점수 집계), qnli_scores.jsonl, mnli_scores.jsonl,
mnli_tmpl_scores.jsonl을 같은 gold unanswerable 라벨에 대해 AUROC·부트스트랩
95% CI(10000회, seed 0)로 재서 나란히 출력한다. positive=answerable, 점수 높을수록
answerable 방향(0.5=분리 없음). 전용 신호가 무료 리랭커 기준선을 유의하게 넘는지
쌍체 부트스트랩으로 확인한다(같은 재표본에서 두 신호 AUROC 차 Δ=신호−rerank_max,
CI가 0을 포함하면 구분 못 함).

두 가지로 나눠 본다.
- full: 원래 질문의 answerability(운용 조건 — 검색이 돌려준 근거만으로 가르는가).
- retrieval_hit: answerable 중 gold 근거가 top-k에 회수된 질문 + 전체 unanswerable
  (검색 실패를 떼어낸 메커니즘 진단 — 근거가 들어와 있을 때도 신호가 약한가).

--csv 로 두 집합 집계를 공개용 표로 저장한다.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.phase3_baseline_auroc import auroc, bootstrap_ci


def auroc_ci(scores, answerable, reps, seed):
    a = auroc(scores, answerable)
    lo, hi = bootstrap_ci(scores, answerable, reps, seed)
    return a, lo, hi


def paired_delta(scores, base, answerable, reps, seed):
    # 같은 재표본 인덱스를 두 신호에 함께 적용해 AUROC 차의 분포를 낸다(쌍체).
    rng = np.random.default_rng(seed)
    n = len(scores)
    diffs = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        a1 = auroc(scores[idx], answerable[idx])
        a0 = auroc(base[idx], answerable[idx])
        if not (np.isnan(a1) or np.isnan(a0)):
            diffs.append(a1 - a0)
    diffs = np.array(diffs)
    delta = auroc(scores, answerable) - auroc(base, answerable)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return delta, lo, hi, (diffs > 0).mean()


def analyze(tag, signals, answerable, sel, reps, seed):
    # sel: 이 부분집합에 포함할 행의 불리언 마스크.
    sub = {name: s[sel] for name, s in signals.items()}
    ans = answerable[sel]
    base = sub["rerank_max"]
    dedicated = [n for n in signals if not n.startswith("rerank_")]

    print(f"[{tag}] n={sel.sum()} answerable={ans.sum()} unanswerable={(~ans).sum()}")
    rows = []
    for name, s in sub.items():
        a, lo, hi = auroc_ci(s, ans, reps, seed)
        print(f"  {name:16s} AUROC={a:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")
        rows.append({"subset": tag, "signal": name, "auroc": round(a, 4),
                     "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})

    print(f"  쌍체 부트스트랩 (vs rerank_max):")
    deltas = {}
    for name in dedicated:
        d, lo, hi, p = paired_delta(sub[name], base, ans, reps, seed)
        deltas[name] = (d, lo, hi, p)
        print(f"  {name:16s} Δ={d:+.3f}  95% CI=[{lo:+.3f}, {hi:+.3f}]  P(Δ>0)={p:.2f}")

    for r in rows:
        d = deltas.get(r["signal"])
        r["delta_vs_rerank_max"] = round(d[0], 4) if d else ""
        r["delta_ci_lo"] = round(d[1], 4) if d else ""
        r["delta_ci_hi"] = round(d[2], 4) if d else ""
        r["p_delta_gt0"] = round(d[3], 4) if d else ""
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=Path("data/phase3/retrieval_dump.jsonl"))
    parser.add_argument("--qnli", type=Path, default=Path("data/phase3/qnli_scores.jsonl"))
    parser.add_argument("--mnli", type=Path, default=Path("data/phase3/mnli_scores.jsonl"))
    parser.add_argument("--mnli-tmpl", type=Path, default=Path("data/phase3/mnli_tmpl_scores.jsonl"))
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, default=None, help="두 부분집합 집계를 공개용 CSV로 저장")
    args = parser.parse_args()

    dump = [json.loads(l) for l in open(args.dump, encoding="utf-8")]
    qnli = [json.loads(l) for l in open(args.qnli, encoding="utf-8")]
    mnli = [json.loads(l) for l in open(args.mnli, encoding="utf-8")]
    mnli_tmpl = [json.loads(l) for l in open(args.mnli_tmpl, encoding="utf-8")]
    # 같은 질문·순서인지 question_id로 확인한다.
    ids = [d["question_id"] for d in dump]
    assert [q["question_id"] for q in qnli] == ids, "qnli 질문 순서 불일치"
    assert [m["question_id"] for m in mnli] == ids, "mnli 질문 순서 불일치"
    assert [m["question_id"] for m in mnli_tmpl] == ids, "mnli_tmpl 질문 순서 불일치"

    answerable = np.array([not d["gold_unanswerable"] for d in dump])
    signals = {
        "rerank_max": np.array([max(e["rerank_score"] for e in d["evidence"]) for d in dump]),
        "rerank_mean": np.array([np.mean([e["rerank_score"] for e in d["evidence"]]) for d in dump]),
        "qnli_max": np.array([q["qnli_max"] for q in qnli]),
        "qnli_mean": np.array([q["qnli_mean"] for q in qnli]),
        "mnli_max": np.array([m["mnli_max"] for m in mnli]),
        "mnli_mean": np.array([m["mnli_mean"] for m in mnli]),
        "mnli_tmpl_max": np.array([m["mnli_max"] for m in mnli_tmpl]),
        "mnli_tmpl_mean": np.array([m["mnli_mean"] for m in mnli_tmpl]),
    }

    # 검색 성공 부분집합: unanswerable은 모두 포함(근거가 없으니 hit 무의미),
    # answerable은 gold 근거가 top-k에 회수된 질문만.
    hit = np.array([d.get("retrieval_hit", False) for d in dump])
    subset_sel = (~answerable) | hit

    rows = []
    rows += analyze("full", signals, answerable, np.ones(len(dump), dtype=bool), args.reps, args.seed)
    print()
    rows += analyze("retrieval_hit", signals, answerable, subset_sel, args.reps, args.seed)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\ncsv: {args.csv}")


if __name__ == "__main__":
    main()
