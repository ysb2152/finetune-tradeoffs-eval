from src.ingest.qasper import Question,Paper,load_paper
from pathlib import Path
from src.generate.qwen import build_messages
from src.eval.answer import answer_string
import random

def oracle_evidence_paragraphs(question:Question) -> list[str]:
    seen = set()
    parts = []
    for answer in question.answers:
        for evidence in answer.evidence:
            text = evidence.text.strip()
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

def to_chat_sample(question):

    evidence=oracle_evidence_paragraphs(question)
    system,user=build_messages(question.text,evidence)
    assistant={"role":"assistant","content":answer_string(question.answers[0])}

    return {"messages":[system,user,assistant]}
