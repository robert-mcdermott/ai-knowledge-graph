"""Configuration utilities for the knowledge graph generator."""
import tomli
import os

def load_config(config_file="config.toml"):
    """
    Load configuration from TOML file or environment variables.
    Environment variables take precedence over config file.

    Args:
        config_file: Path to the TOML configuration file

    Returns:
        Dictionary containing the configuration
    """
    # Default configuration
    config = {
        "llm": {
            "model": "llama-3.1-70b-versatile",
            "api_key": "dummy-key",
            "base_url": "http://localhost:11434/v1/chat/completions",
            "max_tokens": 8192,
            "temperature": 0.8
        },
        "chunking": {
            "chunk_size": 100,
            "overlap": 20
        },
        "standardization": {
            "enabled": True,
            "use_llm_for_entities": True
        },
        "inference": {
            "enabled": True,
            "use_llm_for_inference": True,
            "apply_transitive": True
        },
        "visualization": {
            "edge_smooth": False
        }
    }

    # Try to load from config file
    if os.path.exists(config_file):
        try:
            with open(config_file, "rb") as f:
                file_config = tomli.load(f)
                # Merge file config with defaults
                for section in file_config:
                    if section in config:
                        config[section].update(file_config[section])
                    else:
                        config[section] = file_config[section]
        except Exception as e:
            print(f"Warning: Error loading config file: {e}. Using defaults and environment variables.")

    # Override with environment variables if present
    if os.getenv("LLM_MODEL"):
        config["llm"]["model"] = os.getenv("LLM_MODEL")
    if os.getenv("LLM_API_KEY"):
        config["llm"]["api_key"] = os.getenv("LLM_API_KEY")
    if os.getenv("LLM_BASE_URL"):
        config["llm"]["base_url"] = os.getenv("LLM_BASE_URL")
    if os.getenv("LLM_MAX_TOKENS"):
        config["llm"]["max_tokens"] = int(os.getenv("LLM_MAX_TOKENS"))
    if os.getenv("LLM_TEMPERATURE"):
        config["llm"]["temperature"] = float(os.getenv("LLM_TEMPERATURE"))

    return config 