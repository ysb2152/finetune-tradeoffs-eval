"""Phase 3a 기준선: 검색 점수 집계가 answerable/unanswerable을 가르는지 본다.

전용 신호(C)가 넘어야 할 무료 기준선. retrieval_dump.jsonl 의 질문별 top-5 리랭커
점수를 max·mean 으로 집계해, gold unanswerable 라벨에 대한 AUROC와 부트스트랩
95% CI(10000회, seed 0)를 구한다. unanswerable이면 관련 문단이 약해 점수가 낮다는
가정이라, 점수가 높을수록 answerable로 보는 방향(positive=answerable)으로 AUROC를
재므로 0.5=분리 없음, 1에 가까울수록 분리가 잘 된다.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def auroc(scores, pos_mask):
    # 순위 기반 AUROC(동점은 평균 순위). positive=answerable, 점수 높을수록 answerable.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # 동점 평균 순위 처리
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2
            ranks[order[i : j + 1]] = avg
        i = j + 1
    n_pos = pos_mask.sum()
    n_neg = len(scores) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos_mask].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def bootstrap_ci(scores, pos_mask, reps=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(scores)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        a = auroc(scores[idx], pos_mask[idx])
        if not np.isnan(a):
            vals.append(a)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=Path("data/phase3/retrieval_dump.jsonl"))
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(args.dump, encoding="utf-8")]
    # positive=answerable 이므로 unanswerable의 반대.
    answerable = np.array([not r["gold_unanswerable"] for r in rows])
    s_max = np.array([max(e["rerank_score"] for e in r["evidence"]) for r in rows])
    s_mean = np.array([np.mean([e["rerank_score"] for e in r["evidence"]]) for r in rows])

    n = len(rows)
    print(f"n={n} answerable={answerable.sum()} unanswerable={(~answerable).sum()}")
    for name, s in (("max", s_max), ("mean", s_mean)):
        a = auroc(s, answerable)
        lo, hi = bootstrap_ci(s, answerable, args.reps, args.seed)
        print(f"rerank_{name}: AUROC={a:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
