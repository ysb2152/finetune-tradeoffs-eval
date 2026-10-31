"""NLI 충실성(근거→생성답 entailment)이 answerability 게이트 신호로 쓸 만한지 진단한다.
전용 신호를 새로 만들기 전에, 이미 계산해 둔 nli_*_search.json의 entailment를 재사용해
test 분할에서 answerable과 unanswerable을 가르는지 본다. 기권 대상은 '충실성 낮음'이므로
gateness=1-entailment로 두고, 모델이 스스로 "Unanswerable"로 답해 NLI 점수가 없는 행은
이미 기권한 셈이라 gateness=1(최대)로 대입한다. 임계값과 무관한 분리력은 AUROC로 재되,
자기기권을 뺀 순수 신호(AUROC_scored)와 자기기권을 포함한 값(AUROC_all)을 함께 낸다."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REGIMES = ["base", "full", "lora", "qlora"]


def load_split(split_path):
    splits = json.loads(Path(split_path).read_text(encoding="utf-8"))
    return set(splits["test"]["papers"])


def diagnose(regime, test_papers):
    conf = json.loads(Path(f"runs/eval78/{regime}_search_conf.json").read_text(encoding="utf-8"))
    nli = json.loads(Path(f"runs/eval78/nli_{regime}_search.json").read_text(encoding="utf-8"))
    entail = {a["question_id"]: a["entailment"] for a in nli["answers"]}

    rows = [r for r in conf["rows"] if r["paper_id"] in test_papers]
    y = np.array([1 if r["matched_type"] == "unanswerable" else 0 for r in rows])
    # NLI 점수가 없는 행 = 모델 자기기권. 충실성 신호로는 결측이라 분리할 때 두 갈래로 본다.
    scored = np.array([r["question_id"] in entail for r in rows])
    gateness = np.array([1.0 - entail[r["question_id"]] if r["question_id"] in entail else 1.0
                         for r in rows])

    ans_e = [entail[r["question_id"]] for r in rows
             if r["matched_type"] != "unanswerable" and r["question_id"] in entail]
    unans_e = [entail[r["question_id"]] for r in rows
               if r["matched_type"] == "unanswerable" and r["question_id"] in entail]

    sa_unans = int(((~scored) & (y == 1)).sum())
    return {
        "regime": regime,
        "n_test": len(rows),
        "n_unans": int((y == 1).sum()),
        "ans_entail_mean": round(float(np.mean(ans_e)), 4),
        "unans_entail_mean": round(float(np.mean(unans_e)), 4),
        "auroc_scored": round(float(roc_auc_score(y[scored], gateness[scored])), 4),
        "auroc_all": round(float(roc_auc_score(y, gateness)), 4),
        "self_abstain": int((~scored).sum()),
        "self_abstain_unans": sa_unans,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_path", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    test_papers = load_split(args.split_path)
    out_rows = [diagnose(r, test_papers) for r in REGIMES]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "nli_gate_diagnostic.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    for r in out_rows:
        print(r)


if __name__ == "__main__":
    main()
