# 이식 출처: paper-rag-eval (ysb2152/paper-rag-eval) src/eval/answer.py 를 그대로 복사했다.
# Qasper 표준 token-level Answer-F1. 근거는 P0-7(하네스 재사용) 참고.
import re
import string
from collections import Counter
from src.ingest.qasper import Answer,Question

# 소문자화·관사 제거·문장부호 제거·공백 정리로 채점용 정규화

def normalize_answer(text:str)->str:

    text = text.lower()
    # 관사 제거
    def remove_a(text):
        return re.sub(r"\b(a|an|the)\b"," ",text)

    # 문장부호 제거
    def remove_q(text):
        sets=set(string.punctuation)
        remains=[ch for ch in text if ch not in sets]
        remains="".join(remains)
        return str(remains)

    return " ".join(remove_a(remove_q(text)).split())



# F1 Score 계산
def token_f1(prediction: str, ground_truth: str) -> float:

    prediction=normalize_answer(prediction).split()
    ground_truth=normalize_answer(ground_truth).split()
    sums=sum((Counter(prediction) & Counter(ground_truth)).values())
    if sums==0:
        return 0.0
    precision=sums/len(prediction)
    recall=sums/len(ground_truth)

    return 2*precision*recall/(precision+recall)
# 주석자 답을 문자열로 for 채점
def answer_string(answer: Answer) -> str:
    if answer.unanswerable:
        return "Unanswerable"
    if answer.extractive_spans:
        return ", ".join(answer.extractive_spans)
    if answer.free_form_answer:
        return answer.free_form_answer
    if answer.yes_no:
        return "Yes"
    if answer.yes_no is not None:
        return "No"

    return ""

# 답 유형 분류 — unanswerable/extractive/abstractive/boolean/none

def answer_type(answer: Answer) -> str:

    if answer.unanswerable:
        return "unanswerable"
    if answer.extractive_spans:
        return "extractive"
    if answer.free_form_answer:
        return "abstractive"
    if answer.yes_no is not None:
        return "boolean"
    return "none"
# F1 Score Max 값을 튜플로 (Score,답 유형) 으로 리턴
def answer_f1_detail(prediction, question) -> tuple[float, str]:
    if not question.answers:
        raise ValueError("answers가 비어있습니다")
    tup_list=[(token_f1(prediction,answer_string(answer)),answer_type(answer)) for answer in question.answers]
    max_val=max(tup_list,key=lambda x: x[0])
    return max_val

# 튜플에서 Score만
def answer_f1(prediction,question)->float:
    return answer_f1_detail(prediction,question)[0]
