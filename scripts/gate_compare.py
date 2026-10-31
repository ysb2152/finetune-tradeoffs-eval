"""세 answerability 신호(리랭커 top_score 단독·생성 confidence 단독·둘을 결합한 로지스틱 게이트)를
같은 over-abstention 동작점에서 비교한다. 결합 게이트 가중치는 tune에서 고정 학습하고,
세 신호 모두 test에서 answerable 중 예산 비율만큼만 기권하도록 임계값을 맞춘 뒤(A안: 동작점 정렬)
catch와 F1 이득을 잰다. 질문 단위 paired bootstrap으로 신호별 CI와 결합−단독 Δ의 CI를 구한다."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from gate_answerability import load_conf_rows, split_rows, features, labels, fit_gate, gate_scores

REGIMES = ["base", "full", "lora", "qlora"]
SIGNALS = ["top_score", "gen_confidence", "combined"]
# 결합 게이트가 단독 신호를 이기는지가 B-12의 질문이라 결합을 양쪽 단독과 짝지어 비교한다.
COMPARISONS = [("combined", "top_score"), ("combined", "gen_confidence")]


def percentile(sorted_vals, p):
    # 정렬된 표본에서 선형보간 백분위 (bootstrap_ci.py와 동일 관례)
    k = (len(sorted_vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def operating_point(gateness, y, f1, budget):
    # answerable(y==0) 중 예산 비율만큼만 기권하도록 임계값을 test에서 정렬한다.
    # gateness가 클수록 기권 대상이고, answerable 상위 k개만 넘기는 중간값을 임계값으로 쓴다.
    ans_scores = np.sort(gateness[y == 0])[::-1]
    n_ans = len(ans_scores)
    k = round(budget * n_ans)
    if k <= 0:
        threshold = np.inf
    elif k >= n_ans:
        threshold = -np.inf
    else:
        threshold = (ans_scores[k - 1] + ans_scores[k]) / 2
    gated = gateness > threshold

    n_unans = int((y == 1).sum())
    catch = int((gated & (y == 1)).sum()) / n_unans if n_unans else 0.0
    over_abstention = int((gated & (y == 0)).sum()) / n_ans if n_ans else 0.0

    f1_after = np.where(gated, np.where(y == 1, 1.0, 0.0), f1)
    f1_before = float(f1.mean())
    return {
        "catch": catch,
        "over_abstention": over_abstention,
        "f1_before": f1_before,
        "f1_after": float(f1_after.mean()),
        "f1_gain": float(f1_after.mean()) - f1_before,
        "n_gated": int(gated.sum()),
    }


def build_signals(regime, split_path):
    # test 행에 대한 세 신호의 gateness 벡터와 라벨·f1을 만든다.
    # 단독 신호는 값이 낮을수록 기권 대상이라 부호를 뒤집고, 결합은 tune에서 학습한 게이트의 unanswerable 확률을 쓴다.
    rows = load_conf_rows(regime)
    tune_rows, test_rows = split_rows(rows, split_path)
    model, mean, std = fit_gate(tune_rows)

    X = features(test_rows)
    y = labels(test_rows)
    f1 = np.array([r["f1"] for r in test_rows])
    gateness = {
        "top_score": -X[:, 0],
        "gen_confidence": -X[:, 1],
        "combined": gate_scores(model, mean, std, test_rows),
    }
    return gateness, y, f1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_path", type=Path)
    parser.add_argument("--budget", type=float, default=0.069)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    point_rows = []
    pair_rows = []

    for regime in REGIMES:
        gateness, y, f1 = build_signals(regime, args.split_path)
        n = len(y)

        point = {s: operating_point(gateness[s], y, f1, args.budget) for s in SIGNALS}

        boot = {s: {"catch": [], "f1_gain": []} for s in SIGNALS}
        boot_diff = {c: {"catch": [], "f1_gain": []} for c in COMPARISONS}
        for _ in range(args.n_boot):
            # 같은 인덱스를 세 신호에 함께 적용해야 결합−단독 Δ의 paired 분산이 나온다.
            idx = np.array([rng.randrange(n) for _ in range(n)])
            m = {s: operating_point(gateness[s][idx], y[idx], f1[idx], args.budget) for s in SIGNALS}
            for s in SIGNALS:
                boot[s]["catch"].append(m[s]["catch"])
                boot[s]["f1_gain"].append(m[s]["f1_gain"])
            for a, b in COMPARISONS:
                boot_diff[(a, b)]["catch"].append(m[a]["catch"] - m[b]["catch"])
                boot_diff[(a, b)]["f1_gain"].append(m[a]["f1_gain"] - m[b]["f1_gain"])

        for s in SIGNALS:
            p = point[s]
            row = {"regime": regime, "signal": s,
                   "catch": round(p["catch"], 4),
                   "f1_before": round(p["f1_before"], 4),
                   "f1_after": round(p["f1_after"], 4),
                   "f1_gain": round(p["f1_gain"], 4),
                   "over_abstention": round(p["over_abstention"], 4),
                   "n_gated": p["n_gated"]}
            for metric in ("catch", "f1_gain"):
                sv = sorted(boot[s][metric])
                row[f"{metric}_ci95_low"] = round(percentile(sv, 0.025), 4)
                row[f"{metric}_ci95_high"] = round(percentile(sv, 0.975), 4)
            point_rows.append(row)

        for a, b in COMPARISONS:
            for metric in ("catch", "f1_gain"):
                sv = sorted(boot_diff[(a, b)][metric])
                delta = point[a][metric] - point[b][metric]
                p_gt0 = sum(1 for x in sv if x > 0) / args.n_boot
                pair_rows.append({
                    "regime": regime, "comparison": f"{a}-{b}", "metric": metric,
                    "delta": round(delta, 4),
                    "ci95_low": round(percentile(sv, 0.025), 4),
                    "ci95_high": round(percentile(sv, 0.975), 4),
                    "p_delta_gt_0": round(p_gt0, 4),
                })

    with open(args.out_dir / "gate_compare.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(point_rows[0].keys()))
        w.writeheader()
        w.writerows(point_rows)
    with open(args.out_dir / "gate_compare_pairwise.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)

    print(f"gate_compare.csv: {len(point_rows)} rows | gate_compare_pairwise.csv: {len(pair_rows)} rows "
          f"(paired bootstrap, {args.n_boot} resamples, seed {args.seed}, budget {args.budget})")


if __name__ == "__main__":
    main()
