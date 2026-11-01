"""질문 단위 paired bootstrap으로 레짐별 Answer-F1 평균의 95% 신뢰구간과
레짐 간 차이(Δ)의 신뢰구간을 구한다. 재학습 없이 runs/eval78의 질문별 f1만 리샘플한다."""
import argparse
import csv
import json
import random
from pathlib import Path

REGIMES = ["base", "full", "lora", "qlora"]
EVIDENCES = ["oracle", "search"]
# base는 비교 기준, 상위 레짐끼리는 우열이 노이즈인지 보려고 서로도 비교한다.
COMPARISONS = [("full", "base"), ("lora", "base"), ("qlora", "base"),
               ("full", "lora"), ("lora", "qlora"), ("full", "qlora")]


def percentile(sorted_vals, p):
    # 정렬된 표본에서 선형보간 백분위
    k = (len(sorted_vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", default="runs/eval78")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    # 특정 논문 부분집합(예: gate_split test의 pristine 149문)으로 제한할 때 사용한다.
    parser.add_argument("--papers", help="논문 분할 JSON 경로 (gate_split 형식)")
    parser.add_argument("--subset", default="test", help="--papers 안에서 쓸 키 (예: test/tune)")
    parser.add_argument("--suffix", default="", help="출력 파일명 접미사 (예: _149)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # --papers를 주면 해당 논문들의 질문만 남긴다. 질문→논문 매핑은 결과 행에서 가져온다.
    keep_papers = None
    if args.papers:
        split = json.loads(Path(args.papers).read_text(encoding="utf-8"))
        keep_papers = set(split[args.subset]["papers"])

    # (레짐,근거) 셀마다 question_id → f1
    cells = {}
    for ev in EVIDENCES:
        for rg in REGIMES:
            d = json.loads((eval_dir / f"{rg}_{ev}.json").read_text(encoding="utf-8"))
            rows = d["rows"]
            if keep_papers is not None:
                rows = [r for r in rows if r["paper_id"] in keep_papers]
            cells[(rg, ev)] = {r["question_id"]: r["f1"] for r in rows}

    rng = random.Random(args.seed)
    ci_rows = []
    pair_rows = []

    for ev in EVIDENCES:
        # paired의 전제: 같은 근거 조건의 네 레짐이 동일 질문 집합을 공유해야 한다.
        common = sorted(set.intersection(*(set(cells[(rg, ev)]) for rg in REGIMES)))
        n = len(common)
        vecs = {rg: [cells[(rg, ev)][q] for q in common] for rg in REGIMES}
        point = {rg: sum(vecs[rg]) / n for rg in REGIMES}

        boot_means = {rg: [] for rg in REGIMES}
        boot_diffs = {c: [] for c in COMPARISONS}
        for _ in range(args.n_boot):
            # 같은 인덱스 추출을 네 레짐에 함께 적용해야 Δ의 paired 분산이 나온다.
            idx = [rng.randrange(n) for _ in range(n)]
            m = {rg: sum(vecs[rg][i] for i in idx) / n for rg in REGIMES}
            for rg in REGIMES:
                boot_means[rg].append(m[rg])
            for a, b in COMPARISONS:
                boot_diffs[(a, b)].append(m[a] - m[b])

        for rg in REGIMES:
            s = sorted(boot_means[rg])
            ci_rows.append({
                "regime": rg, "evidence": ev, "n_questions": n,
                "mean_f1": round(point[rg], 4),
                "ci95_low": round(percentile(s, 0.025), 4),
                "ci95_high": round(percentile(s, 0.975), 4),
            })
        for a, b in COMPARISONS:
            s = sorted(boot_diffs[(a, b)])
            p_gt0 = sum(1 for x in s if x > 0) / args.n_boot
            pair_rows.append({
                "evidence": ev, "comparison": f"{a}-{b}",
                "delta": round(point[a] - point[b], 4),
                "ci95_low": round(percentile(s, 0.025), 4),
                "ci95_high": round(percentile(s, 0.975), 4),
                "p_delta_gt_0": round(p_gt0, 4),
            })

    with open(out_dir / f"f1_ci{args.suffix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ci_rows[0].keys()))
        w.writeheader()
        w.writerows(ci_rows)
    with open(out_dir / f"f1_pairwise{args.suffix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)

    print(f"f1_ci.csv: {len(ci_rows)} rows | f1_pairwise.csv: {len(pair_rows)} rows "
          f"(paired bootstrap, {args.n_boot} resamples, seed {args.seed})")


if __name__ == "__main__":
    main()
