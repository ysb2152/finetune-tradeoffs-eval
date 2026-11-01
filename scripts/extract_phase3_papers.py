"""Phase 3 answerability 평가셋을 Qasper train split에서 추출한다. dev는 다중 주석이라
단일 주석 unanswerable이 희박해(잔여 14개) 고갈이고, train은 질문당 단일 주석이라
수율이 높다. SFT에 쓴 논문(data/papers)과 기존 eval(eval78·heldout)을 모두 제외해
모델이 학습에 안 쓴 held-out만 쓴다.

raw Qasper 레코드를 src.ingest.qasper.load_paper가 읽는 per-paper 스키마로 바꿔
data/papers-phase3/<id>.json 로 저장하고, 선택 내역을 manifests/phase3_split.json 에 남긴다.
고정 seed로 논문을 섞어 단일 주석 unanswerable이 목표치에 닿을 때까지 논문 단위로 담는다.

원본은 Qasper v0.3 train(allenai/qasper)이다. data/ 는 추적하지 않으므로, 내려받은
qasper-train-v0.3.json 을 --raw 로 지정하거나 data/qasper-raw/ 아래 두면 된다.
"""
import argparse
import json
import random
from pathlib import Path


def transform(pid, rec):
    # raw: full_text=섹션 객체 리스트, qas=질문 리스트. per-paper는 병렬 리스트 형태를 쓴다.
    full_text = {
        "section_name": [s["section_name"] for s in rec["full_text"]],
        "paragraphs": [s["paragraphs"] for s in rec["full_text"]],
    }
    qas = {
        "question_id": [q["question_id"] for q in rec["qas"]],
        "question": [q["question"] for q in rec["qas"]],
        # 주석 래퍼(annotation_id/worker_id)를 벗기고 answer 본체만 리스트로 모은다.
        "answers": [{"answer": [w["answer"] for w in q["answers"]]} for q in rec["qas"]],
    }
    return {
        "id": pid,
        "title": rec["title"],
        "abstract": rec["abstract"],
        "full_text": full_text,
        "qas": qas,
        "figures_and_tables": rec.get("figures_and_tables"),
    }


def single_unans(rec):
    # 단일 주석 질문 수와 그중 unanswerable 수 (평가 하네스가 단일 주석만 쓴다).
    single = unans = 0
    for q in rec["qas"]:
        if len(q["answers"]) == 1:
            single += 1
            if q["answers"][0]["answer"]["unanswerable"]:
                unans += 1
    return single, unans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/qasper-raw/qasper-train-v0.3.json")
    parser.add_argument("--target-unans", type=int, default=75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="data/papers-phase3")
    parser.add_argument("--manifest", default="manifests/phase3_split.json")
    args = parser.parse_args()

    train = json.loads(Path(args.raw).read_text(encoding="utf-8-sig"))

    exclude = set()
    for d in ("data/papers", "data/papers-eval78", "data/papers-heldout"):
        exclude.update(p.stem for p in Path(d).glob("*.json"))

    ids = sorted(train.keys())
    random.Random(args.seed).shuffle(ids)

    selected = []
    tot_single = tot_unans = 0
    for pid in ids:
        if pid in exclude:
            continue
        single, unans = single_unans(train[pid])
        if single == 0:
            continue
        selected.append(pid)
        tot_single += single
        tot_unans += unans
        if tot_unans >= args.target_unans:
            break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for pid in selected:
        (out_dir / f"{pid}.json").write_text(
            json.dumps(transform(pid, train[pid]), ensure_ascii=False), encoding="utf-8"
        )

    manifest = {
        "split": "phase3-answerability",
        "source": Path(args.raw).name,
        "source_split": "qasper train (held-out: SFT·eval78·heldout 전부 제외)",
        "seed": args.seed,
        "target_unanswerable": args.target_unans,
        "n_papers": len(selected),
        "n_single_questions": tot_single,
        "n_unanswerable": tot_unans,
        "papers": selected,
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"selected papers={len(selected)} single_q={tot_single} unanswerable={tot_unans}")
    print(f"out: {out_dir}  manifest: {args.manifest}")


if __name__ == "__main__":
    main()
