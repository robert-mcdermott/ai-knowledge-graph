import pytest

from src.knowledge_graph.config import ConfigError, load_config, resolve_secret, validate_config


def base():
    return {"llm": {"model": "m", "base_url": "http://x"}}


def test_defaults_are_applied():
    cfg = validate_config(base())
    assert cfg["llm"]["max_tokens"] == 32768
    assert cfg["llm"]["token_param"] == "auto"
    assert cfg["chunking"] == {"chunk_size": 500, "overlap": 50}
    assert cfg["inference"]["apply_transitive"] is False


def test_missing_model_is_an_error():
    with pytest.raises(ConfigError, match="model"):
        validate_config({"llm": {"base_url": "http://x"}})


def test_overlap_must_be_smaller_than_chunk_size():
    cfg = base()
    cfg["chunking"] = {"chunk_size": 100, "overlap": 100}
    with pytest.raises(ConfigError, match="overlap"):
        validate_config(cfg)


def test_bad_token_param():
    cfg = base()
    cfg["llm"]["token_param"] = "tokens"
    with pytest.raises(ConfigError, match="token_param"):
        validate_config(cfg)


def test_resolve_secret_forms(monkeypatch):
    monkeypatch.setenv("K", "v")
    assert resolve_secret("env:K") == "v"
    assert resolve_secret("${K}") == "v"
    assert resolve_secret("sk-plain") == "sk-plain"
    assert resolve_secret(None) is None
    monkeypatch.delenv("K")
    with pytest.raises(ConfigError, match="K"):
        resolve_secret("env:K")


def test_load_config_round_trip(tmp_path, capsys):
    path = tmp_path / "c.toml"
    path.write_text('[llm]\nmodel = "m"\nbase_url = "http://x"\n[chunking]\nchunk_size = 10\noverlap = 2\n')
    cfg = load_config(str(path))
    assert cfg["chunking"]["overlap"] == 2 and cfg["llm"]["timeout"] == 300


def test_load_config_reports_invalid_and_missing(tmp_path, capsys):
    path = tmp_path / "c.toml"
    path.write_text('[llm]\nmodel = "m"\nbase_url = "http://x"\n[chunking]\nchunk_size = 5\noverlap = 9\n')
    assert load_config(str(path)) is None
    assert "overlap" in capsys.readouterr().out
    assert load_config(str(tmp_path / "nope.toml")) is None
    assert "not found" in capsys.readouterr().out
