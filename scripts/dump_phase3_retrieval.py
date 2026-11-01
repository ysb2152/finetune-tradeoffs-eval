"""Phase 3 answerability 신호 설계용 검색 덤프를 만든다.

data/papers-phase3 의 242편에서 단일 주석 질문을 모두 꺼내, 하네스의 검색 조건과
같은 파이프라인(within_paper, candidate_n=20·top_k=5)으로 질문별 상위 문단을 뽑는다.
질문마다 검색 근거(문단 id·섹션·리랭커 점수·clean_refs 본문)와 gold answerability
라벨(단일 주석의 unanswerable 여부·답 유형)을 JSONL 한 줄로 남긴다.

이 덤프가 '질문↔근거' 전용 신호의 입력이다. 출력에 Qasper 본문이 들어가므로
data/ 아래(추적 제외)에 두고, 재현은 manifests/phase3_split.json + 추출기 +
이 스크립트로 한다.
"""
import argparse
import json
from pathlib import Path

from src.data.sample import single_annotation_questions, clean_refs
from src.eval.answer import answer_type
from src.retrieve.embed import load_embedder
from src.retrieve.rerank import load_reranker
from src.retrieve.pipeline import within_paper


def dump(papers_dir, out, candidate_n, top_k):
    paths = sorted(papers_dir.glob("*.json"))
    pairs = single_annotation_questions(paths)

    embed_bundle = load_embedder()
    rerank_bundle = load_reranker()

    out.parent.mkdir(parents=True, exist_ok=True)
    n_unans = 0
    with open(out, "w", encoding="utf-8") as f:
        for i, (paper, question) in enumerate(pairs, 1):
            results = within_paper(
                paper.paragraphs, question.text, embed_bundle, rerank_bundle, candidate_n, top_k
            )
            evidence = [
                {
                    "paragraph_id": r.paragraph.id,
                    "section": r.paragraph.section,
                    "rerank_score": r.score,
                    "text": clean_refs(r.paragraph.text),
                }
                for r in results
            ]
            # 평가 하네스처럼 단일 주석만 쓰므로 gold 라벨은 그 한 답에서 바로 나온다.
            answer = question.answers[0]
            n_unans += answer.unanswerable
            record = {
                "paper_id": paper.id,
                "question_id": question.id,
                "question": question.text,
                "gold_unanswerable": answer.unanswerable,
                "answer_type": answer_type(answer),
                "evidence": evidence,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"{i}/{len(pairs)}")

    print(f"questions={len(pairs)} unanswerable={n_unans}")
    print(f"out: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers-dir", type=Path, default=Path("data/papers-phase3"))
    parser.add_argument("--out", type=Path, default=Path("data/phase3/retrieval_dump.jsonl"))
    parser.add_argument("--candidate-n", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    dump(args.papers_dir, args.out, args.candidate_n, args.top_k)


if __name__ == "__main__":
    main()
