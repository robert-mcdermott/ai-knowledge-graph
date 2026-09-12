"""Configuration loading, validation and defaults for the knowledge graph generator."""
from __future__ import annotations

import os
import re
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - older interpreters
    import tomli as tomllib  # type: ignore[no-redef]

_ENV_BRACES = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")

DEFAULTS: dict[str, dict[str, Any]] = {
    "llm": {
        "max_tokens": 32768,
        "temperature": 0.2,
        "timeout": 300,
        "max_retries": 3,
        "token_param": "auto",
        "json_mode": False,
        "concurrency": 4,
        "cache_dir": ".kg-cache",
    },
    "chunking": {"chunk_size": 500, "overlap": 50},
    "standardization": {"enabled": True, "use_llm_for_entities": True, "merge_word_subsets": False},
    "inference": {
        "enabled": True,
        "use_llm_for_inference": True,
        "apply_transitive": False,
        "transitive_max_hub_degree": 10,
        "transitive_max_per_subject": 5,
        "lexical": False,
        "lexical_min_word_length": 5,
        "max_inferred_ratio": 0.5,
        "taxonomy": True,
        "llm_bridge": True,
        "max_bridge_components": 20,
        "bridge_groups_per_call": 5,
        "llm_hub": True,
        "hub_entities": 25,
        "hub_max_new": 25,
    },
    "visualization": {"edge_smooth": False, "name_communities": True, "show_inferred": True},
}


class ConfigError(ValueError):
    """Raised when the configuration is missing or invalid."""


def resolve_secret(value: Any) -> Any:
    """Resolve ``env:NAME`` / ``${NAME}`` references to environment variables.

    Plain strings are returned unchanged so existing configs keep working.
    """
    if not isinstance(value, str):
        return value
    name = None
    if value.startswith("env:"):
        name = value[4:].strip()
    else:
        match = _ENV_BRACES.match(value)
        if match:
            name = match.group(1)
    if name is None:
        return value
    resolved = os.environ.get(name)
    if not resolved:
        raise ConfigError(f"Environment variable {name!r} referenced by the config is not set")
    return resolved


def apply_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Fill in missing sections/keys from :data:`DEFAULTS` (in place) and return the config."""
    for section, values in DEFAULTS.items():
        table = config.setdefault(section, {})
        for key, default in values.items():
            table.setdefault(key, default)
    return config


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate types and ranges, resolve secrets, and return the config.

    Raises:
        ConfigError: with a message that names the offending key.
    """
    if not isinstance(config, dict):
        raise ConfigError("Configuration must be a table")
    apply_defaults(config)

    llm = config["llm"]
    for key in ("model", "base_url"):
        if not llm.get(key):
            raise ConfigError(f"[llm] {key} is required")
    llm["api_key"] = resolve_secret(llm.get("api_key"))
    if not isinstance(llm["max_tokens"], int) or llm["max_tokens"] <= 0:
        raise ConfigError("[llm] max_tokens must be a positive integer")
    if llm.get("temperature") is not None and not isinstance(llm["temperature"], (int, float)):
        raise ConfigError("[llm] temperature must be a number (or omitted)")
    if llm["token_param"] not in ("auto", "max_tokens", "max_completion_tokens"):
        raise ConfigError("[llm] token_param must be 'auto', 'max_tokens' or 'max_completion_tokens'")
    if not isinstance(llm.get("extra_body", {}), dict):
        raise ConfigError("[llm] extra_body must be a table")
    if not isinstance(llm["concurrency"], int) or llm["concurrency"] < 1:
        raise ConfigError("[llm] concurrency must be a positive integer")

    chunking = config["chunking"]
    size, overlap = chunking["chunk_size"], chunking["overlap"]
    if not isinstance(size, int) or size <= 0:
        raise ConfigError("[chunking] chunk_size must be a positive integer")
    if not isinstance(overlap, int) or overlap < 0:
        raise ConfigError("[chunking] overlap must be a non-negative integer")
    if overlap >= size:
        raise ConfigError(f"[chunking] overlap ({overlap}) must be smaller than chunk_size ({size})")

    inference = config["inference"]
    ratio = inference["max_inferred_ratio"]
    if not isinstance(ratio, (int, float)) or ratio < 0:
        raise ConfigError("[inference] max_inferred_ratio must be a number >= 0")
    for key in ("transitive_max_hub_degree", "transitive_max_per_subject", "lexical_min_word_length",
                "max_bridge_components", "bridge_groups_per_call", "hub_entities", "hub_max_new"):
        if not isinstance(inference[key], int) or inference[key] < 0:
            raise ConfigError(f"[inference] {key} must be a non-negative integer")
    groups = inference.get("transitive_predicate_groups")
    if groups is not None:
        from knowledge_graph.entity_standardization import TRANSITIVE_PREDICATE_GROUPS
        unknown = set(groups) - set(TRANSITIVE_PREDICATE_GROUPS)
        if unknown:
            raise ConfigError(f"[inference] unknown transitive_predicate_groups {sorted(unknown)}; "
                              f"choose from {sorted(TRANSITIVE_PREDICATE_GROUPS)}")
    return config


def load_config(config_file: str = "config.toml") -> dict[str, Any] | None:
    """Load, default and validate a TOML config. Returns ``None`` (after printing) on failure."""
    try:
        with open(config_file, "rb") as f:
            config = tomllib.load(f)
    except FileNotFoundError:
        print(f"Error: config file not found: {config_file}")
        return None
    except (OSError, tomllib.TOMLDecodeError) as e:
        print(f"Error loading config file {config_file}: {e}")
        return None
    try:
        return validate_config(config)
    except ConfigError as e:
        print(f"Invalid configuration in {config_file}: {e}")
        return None
