from knowledge_graph.prompts import language_instruction, system_prompt_for


def test_language_instruction_only_for_non_english():
    assert language_instruction("auto") == ""
    assert language_instruction("English") == ""
    assert language_instruction(None) == ""
    assert "Chinese" in language_instruction("Chinese")


def test_system_prompt_for_appends_language():
    plain = system_prompt_for("main_system", {"extraction": {"language": "auto"}})
    zh = system_prompt_for("main_system", {"extraction": {"language": "Chinese"}})
    assert zh.startswith(plain) and "in Chinese" in zh
    assert system_prompt_for("hub_inference_system", {}) == system_prompt_for("hub_inference_system", None)
