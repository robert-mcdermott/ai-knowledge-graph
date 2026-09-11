import pytest

from src.knowledge_graph.llm import extract_json_from_text, strip_reasoning

TRIPLE = {"subject": "james watt", "predicate": "refined", "object": "steam engine"}


def test_clean_array():
    assert extract_json_from_text('[{"subject": "a", "predicate": "p", "object": "b"}]') == [
        {"subject": "a", "predicate": "p", "object": "b"}
    ]


def test_code_fence_with_language():
    text = 'Here you go:\n```json\n[{"subject": "a", "predicate": "p", "object": "b"}]\n```\nDone.'
    assert extract_json_from_text(text) == [{"subject": "a", "predicate": "p", "object": "b"}]


def test_think_block_is_ignored():
    text = '<think>maybe [1, 2] or {"x": 1}</think>\n[{"subject": "a", "predicate": "p", "object": "b"}]'
    assert extract_json_from_text(text) == [{"subject": "a", "predicate": "p", "object": "b"}]
    assert strip_reasoning("<think>hidden</think>visible") == "visible"


def test_prose_around_array():
    text = 'Sure! Here are the triples: [{"subject": "a", "predicate": "p", "object": "b"}] Let me know.'
    assert extract_json_from_text(text) == [{"subject": "a", "predicate": "p", "object": "b"}]


def test_object_expected_returns_dict_not_inner_list():
    text = 'Sure! {"steam engine": ["steam engines", "the steam engine"]}'
    assert extract_json_from_text(text, expect="object") == {
        "steam engine": ["steam engines", "the steam engine"]
    }


def test_object_expected_but_array_given_returns_none():
    assert extract_json_from_text('[{"a": 1}]', expect="object") is None


def test_array_expected_but_mapping_given_returns_none():
    # A dict with several list values is an entity mapping, not a wrapped array.
    assert extract_json_from_text('{"a": ["x"], "b": ["y"]}') is None


def test_json_mode_wrapper_is_unwrapped():
    assert extract_json_from_text('{"triples": [{"subject": "a", "predicate": "p", "object": "b"}]}') == [
        {"subject": "a", "predicate": "p", "object": "b"}
    ]


def test_single_bare_triple_becomes_list():
    assert extract_json_from_text('{"subject": "a", "predicate": "p", "object": "b"}') == [
        {"subject": "a", "predicate": "p", "object": "b"}
    ]


def test_colon_inside_string_value_survives_repair():
    text = '[{subject: "time: 10am", predicate: "p", object: "o"}]'
    assert extract_json_from_text(text) == [{"subject": "time: 10am", "predicate": "p", "object": "o"}]


def test_trailing_commas_are_repaired():
    text = '[{"subject": "a", "predicate": "p", "object": "b",},]'
    assert extract_json_from_text(text) == [{"subject": "a", "predicate": "p", "object": "b"}]


def test_brackets_inside_strings_do_not_confuse_matching():
    text = '[{"subject": "arr[0]", "predicate": "p", "object": "b"}]'
    assert extract_json_from_text(text) == [{"subject": "arr[0]", "predicate": "p", "object": "b"}]


def test_truncated_array_salvages_complete_objects():
    text = ('[{"subject": "a", "predicate": "p", "object": "b"},'
            ' {"subject": "c", "predicate": "q", "object": "d"}, {"subject": "e", "pred')
    assert extract_json_from_text(text) == [
        {"subject": "a", "predicate": "p", "object": "b"},
        {"subject": "c", "predicate": "q", "object": "d"},
    ]


@pytest.mark.parametrize("text", ["", "   ", None, "no json here", "<think>only thoughts</think>"])
def test_nothing_usable_returns_none(text):
    assert extract_json_from_text(text) is None


def test_invalid_expect_rejected():
    with pytest.raises(ValueError):
        extract_json_from_text("[]", expect="thing")
