import json
from src.ingest.qasper import load_paper


def write_paper(tmp_path, data):
    path = tmp_path / "paper.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def base_data(**overrides):
    data = {
        "id": "P1",
        "title": "A Title",
        "abstract": "An abstract.",
        "full_text": {
            "section_name": ["Intro", "Method"],
            "paragraphs": [["intro p0", "intro p1"], ["method p0"]],
        },
        "qas": {
            "question_id": ["q1"],
            "question": ["What?"],
            "answers": [{"answer": [{
                "unanswerable": False,
                "extractive_spans": ["intro p0"],
                "free_form_answer": "",
                "yes_no": None,
                "evidence": ["intro p0", "not a real paragraph"],
            }]}],
        },
    }
    data.update(overrides)
    return data


def test_paragraph_id_scheme(tmp_path):
    paper = load_paper(write_paper(tmp_path, base_data()))
    ids = [p.id for p in paper.paragraphs]
    assert ids == ["P1:s0:p0", "P1:s0:p1", "P1:s1:p0"]


def test_evidence_maps_to_paragraph_ids(tmp_path):
    paper = load_paper(write_paper(tmp_path, base_data()))
    evidence = paper.questions[0].answers[0].evidence
    # 본문에 있는 근거는 문단 id로 매핑되고, 없는 텍스트는 빈 리스트가 된다
    assert evidence[0].paragraph_ids == ["P1:s0:p0"]
    assert evidence[1].paragraph_ids == []


def test_empty_questions(tmp_path):
    data = base_data(qas={"question_id": [], "question": [], "answers": []})
    paper = load_paper(write_paper(tmp_path, data))
    assert paper.questions == []
    assert len(paper.paragraphs) == 3
