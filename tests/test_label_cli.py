import json

from aegis.eval.label_cli import _manual_search, _preview, append_label, load_existing_labels


def test_preview_truncates_long_text_with_ellipsis():
    text = "word " * 100
    result = _preview(text, length=20)
    assert len(result) == 23
    assert result.endswith("...")


def test_preview_collapses_newlines_and_extra_whitespace():
    assert _preview("line one\n\n   line two") == "line one line two"


def test_preview_leaves_short_text_untouched():
    assert _preview("short text") == "short text"


def test_manual_search_matches_on_text_or_heading_case_insensitive():
    chunks = [
        {"id": "a", "doc_id": "d1", "heading_path": ["CORS"], "text": "Use CORSMiddleware to enable it."},
        {"id": "b", "doc_id": "d2", "heading_path": ["Unrelated"], "text": "Something about databases."},
    ]
    assert [c["id"] for c in _manual_search(chunks, "cors")] == ["a"]


def test_manual_search_caps_at_fifteen_results():
    chunks = [{"id": str(i), "doc_id": "d", "heading_path": [], "text": "matchme"} for i in range(30)]
    assert len(_manual_search(chunks, "matchme")) == 15


def test_load_existing_labels_and_append_round_trip(tmp_path):
    path = tmp_path / "labels_in_progress.jsonl"
    assert load_existing_labels(path) == {}

    append_label(path, {"query_id": "golden::issue::1", "stratum": "how_to", "status": "accepted", "ground_truth_chunk_ids": ["c1"]})
    append_label(path, {"query_id": "golden::issue::2", "stratum": "how_to", "status": "skipped", "ground_truth_chunk_ids": []})

    loaded = load_existing_labels(path)
    assert set(loaded) == {"golden::issue::1", "golden::issue::2"}
    assert loaded["golden::issue::1"]["status"] == "accepted"


def test_append_label_is_resumable_across_multiple_writes(tmp_path):
    path = tmp_path / "labels_in_progress.jsonl"
    append_label(path, {"query_id": "a", "stratum": "how_to", "status": "accepted", "ground_truth_chunk_ids": ["x"]})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["query_id"] == "a"
