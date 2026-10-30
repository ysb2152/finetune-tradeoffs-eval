from src.data.sample import paper_split


def make_papers(tmp_path, n):
    # paper_split 은 *.json 경로만 섞어 나누므로 내용은 비워도 된다
    for i in range(n):
        (tmp_path / f"paper_{i:02d}.json").write_text("{}", encoding="utf-8")
    return tmp_path


def test_train_and_val_are_disjoint_and_cover_all(tmp_path):
    make_papers(tmp_path, 20)
    train, val = paper_split(tmp_path, val=8, seed=0)
    assert len(val) == 8
    assert len(train) == 12
    assert set(train).isdisjoint(set(val))
    assert set(train) | set(val) == set(tmp_path.glob("*.json"))


def test_same_seed_is_deterministic(tmp_path):
    make_papers(tmp_path, 20)
    assert paper_split(tmp_path, 8, 0) == paper_split(tmp_path, 8, 0)


def test_different_seed_changes_split(tmp_path):
    make_papers(tmp_path, 20)
    _, val0 = paper_split(tmp_path, 8, 0)
    _, val1 = paper_split(tmp_path, 8, 1)
    assert val0 != val1
