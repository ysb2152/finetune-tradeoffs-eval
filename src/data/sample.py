from src.ingest.qasper import Question,Paper,load_paper
from pathlib import Path
from src.generate.qwen import build_messages
from src.eval.answer import answer_string
from src.retrieve.pipeline import within_paper
import random
import re

def oracle_evidence_paragraphs(question:Question) -> list[str]:
    seen = set()
    parts = []
    for answer in question.answers:
        for evidence in answer.evidence:
            text = evidence.text.strip()
            text= clean_refs(text)
            if text and text not in seen:
                seen.add(text)
                parts.append(text)
    return parts

def single_annotation_questions(paths:Path) -> list:

    papers = [load_paper(p) for p in paths]
    single_list=[]

    for paper in papers:
        for question in paper.questions:
            if len(question.answers)==1:

                single_list.append((paper,question))

    return single_list


def paper_split(path,val=8,seed=0):

    paths = sorted([p for p in path.glob("*.json")])
    rng=random.Random(seed)
    rng.shuffle(paths)
    val_paths=paths[:val]
    train_paths=paths[val:]

    return (train_paths,val_paths)

def to_chat_sample(question, evidence):

    system,user=build_messages(question.text,evidence)
    assistant={"role":"assistant","content":clean_refs(answer_string(question.answers[0]))}

    return {"messages":[system,user,assistant]}

def clean_refs(text: str) -> str:

    text=re.sub(r"[A-Z]{2,}"+r"REF"+r"\d*[a-z]*"r"|\bFLOAT\b|\bFORMULA\b"," ",text)
    text=" ".join(text.split())

    return text

def search_evidence_paragraphs(paper,question,embed_bundle,rerank_bundle,candidate_n=20,top_k=5) -> list[str]:

    top_k_Searchlist = within_paper(paper.paragraphs,question.text,embed_bundle,rerank_bundle,candidate_n,top_k)
    evidence_list=[]

    for searchresult in top_k_Searchlist:

        evidence_list.append(clean_refs(searchresult.paragraph.text))


    return evidence_list


