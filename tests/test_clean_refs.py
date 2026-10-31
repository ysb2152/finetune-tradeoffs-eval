import pytest
from src.data.sample import clean_refs


# Qasper placeholder 마커 제거 — REF 패밀리와 FLOAT/FORMULA

@pytest.mark.parametrize("marker", ["BIBREF", "BIBREF3", "TABREF2", "FIGREF0",
                                    "SECREF", "EQREF1"])
def test_removes_ref_family(marker):
    assert clean_refs(f"see {marker} for detail") == "see for detail"


def test_removes_float_and_formula():
    assert clean_refs("as in FLOAT and FORMULA here") == "as in and here"


def test_collapses_whitespace():
    assert clean_refs("a   b\n c") == "a b c"


def test_keeps_plain_text():
    assert clean_refs("a normal sentence") == "a normal sentence"


def test_lowercase_formula_is_kept():
    # 대문자 placeholder만 노린 패턴이라 일반 단어 formula/float(소문자)는 보존돼야 한다
    assert clean_refs("the formula for float values") == "the formula for float values"


def test_ref_without_prefix_is_kept():
    # [A-Z]{2,}REF 라서 앞에 2글자 이상 대문자가 없는 "REF" 단독은 지우지 않는다
    assert clean_refs("REF is a word") == "REF is a word"


def test_formula_substring_should_not_be_deleted():
    assert clean_refs("the word FORMULAIC stays") == "the word FORMULAIC stays"
