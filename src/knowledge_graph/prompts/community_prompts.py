"""Phase 4: community naming prompts."""

COMMUNITY_NAMING_SYSTEM_PROMPT = """
You label clusters of related entities from a knowledge graph with short, specific names.
"""


def get_community_naming_user_prompt(communities_text):
    return f"""
Below are clusters of entities detected in a knowledge graph, each with its most connected members.
Give every cluster a short name (2-4 words) that describes what its members have in common.
Be specific ("Steam power and railways", not "Technology"). Do not reuse a name for two clusters.

{communities_text}

Return ONLY a JSON object mapping the cluster number (as a string) to its name, for example:
{{
  "1": "Steam power and railways",
  "2": "Labour movements"
}}
"""
