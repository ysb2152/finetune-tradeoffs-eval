import numpy as np
import pytest

from scripts.phase3_baseline_auroc import auroc, bootstrap_ci
from scripts.phase3_signal_compare import paired_delta


# auroc: positive=answerable, 점수 높을수록 positive 방향

def test_auroc_perfect_separation():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    pos = np.array([False, False, True, True])  # 상위 두 점수가 positive
    assert auroc(scores, pos) == 1.0


def test_auroc_reversed_is_zero():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    pos = np.array([True, True, False, False])  # positive가 하위 점수
    assert auroc(scores, pos) == 0.0


def test_auroc_no_separation_half():
    scores = np.array([5.0, 5.0, 5.0, 5.0])  # 전부 동점
    pos = np.array([True, False, True, False])
    assert auroc(scores, pos) == 0.5


def test_auroc_ties_average():
    # 동점이 섞인 경우 평균 순위로 0.5가 나오는 작은 예.
    scores = np.array([1.0, 1.0, 2.0, 2.0])
    pos = np.array([False, True, False, True])
    assert auroc(scores, pos) == pytest.approx(0.5)


def test_auroc_nan_when_one_class_missing():
    scores = np.array([1.0, 2.0, 3.0])
    pos = np.array([True, True, True])  # negative가 없음
    assert np.isnan(auroc(scores, pos))


# bootstrap_ci: 완전 분리면 거의 항상 1.0이라 CI가 1 근방

def test_bootstrap_ci_perfect_separation_near_one():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    pos = np.array([False, False, False, True, True, True])
    lo, hi = bootstrap_ci(scores, pos, reps=200, seed=0)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


# paired_delta: 같은 재표본에서 두 신호 AUROC 차

def test_paired_delta_identical_signals_zero():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    answerable = np.array([False, False, False, True, True, True])
    delta, lo, hi, p = paired_delta(scores, scores, answerable, reps=200, seed=0)
    assert delta == 0.0 and lo == 0.0 and hi == 0.0
    assert p == 0.0  # 0을 초과하는 차가 하나도 없음


def test_paired_delta_better_signal_positive():
    answerable = np.array([False, False, True, True, True, True])
    base = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])   # 분리 없음(AUROC 0.5)
    better = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])  # 완전 분리(AUROC 1.0)
    delta, lo, hi, p = paired_delta(better, base, answerable, reps=200, seed=0)
    assert delta > 0
