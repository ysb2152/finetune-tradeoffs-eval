from scripts.signal_utils import aggregate_by_counts, build_pair


# build_pair: 모델별로 질문↔근거를 어떤 순서·형식으로 묶는지

def test_build_pair_qnli_keeps_question_first():
    assert build_pair("Q", "P", "qnli") == ("Q", "P")


def test_build_pair_plain_puts_passage_first():
    assert build_pair("Q", "P", "plain") == ("P", "Q")


def test_build_pair_template_prefixes_question():
    assert build_pair("Q", "P", "template") == ("P", "This passage answers the question: Q")


# aggregate_by_counts: 평탄한 점수를 질문별 근거 개수만큼 끊어 (max, mean)으로 집계

def test_aggregate_uneven_counts():
    # 질문별 근거 개수가 2·3·1로 다른 경우 경계가 어긋나지 않아야 한다.
    scores = [1.0, 9.0, 2.0, 2.0, 2.0, 5.0]
    assert aggregate_by_counts(scores, [2, 3, 1]) == [(9.0, 5.0), (2.0, 2.0), (5.0, 5.0)]


def test_aggregate_single_evidence_each():
    # 질문마다 근거 1개면 max·mean 모두 그 값 그대로.
    assert aggregate_by_counts([3.0, 7.0], [1, 1]) == [(3.0, 3.0), (7.0, 7.0)]
