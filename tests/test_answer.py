import pytest
from src.eval.answer import (
    normalize_answer,
    token_f1,
    answer_string,
    answer_type,
    answer_f1,
    answer_f1_detail,
)
from src.ingest.qasper import Answer, Question


def make_answer(unanswerable=False, extractive_spans=None, free_form_answer="",
                yes_no=None):
    return Answer(unanswerable, extractive_spans or [], free_form_answer, yes_no, [])


# normalize_answer: 소문자화·관사 제거·문장부호 제거

def test_normalize_lowercases_and_strips_punctuation():
    assert normalize_answer("Yes!") == "yes"


def test_normalize_removes_articles():
    assert normalize_answer("the cat and a dog") == "cat and dog"


# token_f1

def test_f1_exact_match_is_one():
    assert token_f1("neural machine translation", "neural machine translation") == 1.0


def test_f1_no_overlap_is_zero():
    assert token_f1("cats", "dogs") == 0.0


def test_f1_ignores_articles_and_case():
    assert token_f1("The Model", "model") == 1.0


def test_f1_partial_overlap():
    # 공통 토큰 2개(y, z), 예측 3·정답 3 → precision=recall=2/3 → f1=2/3
    # (관사 a/an/the는 정규화에서 빠지므로 일반 토큰으로 둔다)
    assert token_f1("x y z", "y z w") == pytest.approx(2 / 3)


# answer_string: 보류/추출/자유서술/boolean 변환

def test_answer_string_unanswerable():
    assert answer_string(make_answer(unanswerable=True)) == "Unanswerable"


def test_answer_string_extractive_joins_spans():
    assert answer_string(make_answer(extractive_spans=["A", "B"])) == "A, B"


def test_answer_string_free_form():
    assert answer_string(make_answer(free_form_answer="a short answer")) == "a short answer"


def test_answer_string_yes_and_no():
    assert answer_string(make_answer(yes_no=True)) == "Yes"
    assert answer_string(make_answer(yes_no=False)) == "No"


def test_answer_string_empty_when_nothing_set():
    assert answer_string(make_answer()) == ""


# answer_type 분류

@pytest.mark.parametrize("answer,expected", [
    (make_answer(unanswerable=True), "unanswerable"),
    (make_answer(extractive_spans=["x"]), "extractive"),
    (make_answer(free_form_answer="free"), "abstractive"),
    (make_answer(yes_no=False), "boolean"),
    (make_answer(), "none"),
])
def test_answer_type(answer, expected):
    assert answer_type(answer) == expected


# answer_f1_detail: 주석 여러 개면 최대 F1을 그 유형과 함께 반환

def test_answer_f1_detail_takes_max_over_annotations():
    q = Question("q1", "?", [
        make_answer(extractive_spans=["wrong"]),
        make_answer(free_form_answer="correct answer"),
    ])
    score, matched_type = answer_f1_detail("correct answer", q)
    assert score == 1.0
    assert matched_type == "abstractive"


def test_answer_f1_raises_on_empty_answers():
    with pytest.raises(ValueError):
        answer_f1("anything", Question("q1", "?", []))
