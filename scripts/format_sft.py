import argparse
import json
from pathlib import Path
from src.data.sample import paper_split, single_annotation_questions, to_chat_sample
from src.retrieve.embed import load_embedder
from src.retrieve.rerank import load_reranker
from src.data.sample import search_evidence_paragraphs


def format_sft(train_paths,out):

    embed_bundle=load_embedder()
    rerank_bundle=load_reranker()



    out.parent.mkdir(parents=True,exist_ok=True)
    with open(out,"w",encoding="utf-8") as f:
        for (paper,question) in single_annotation_questions(train_paths):
            evidence = search_evidence_paragraphs(paper, question, embed_bundle, rerank_bundle)
            sample=to_chat_sample(question,evidence)
            f.write(json.dumps(sample,ensure_ascii=False) +"\n")



# CLI: 논문 폴더를 받아 train 분할의 단일 주석 질문을 SFT chat 샘플(JSONL)로 저장

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("papers_dir", type=Path, help="Qasper 논문 폴더")
    parser.add_argument("--val", type=int, default=8)
    parser.add_argument("--out", type=Path, default=0,help="결과 JSON 저장 경로")
    parser.add_argument("--seed", type=int, default=0, help="분할 seed")
    args = parser.parse_args()

    train_paths,_ = paper_split(args.papers_dir, args.val,args.seed)
    format_sft(train_paths,args.out)

if __name__ == "__main__":
    main()
