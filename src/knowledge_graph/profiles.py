"""Small, editable extraction profiles; no extra runtime dependencies."""
PROFILES = {
    "general": {"label": "General reading", "description": "People, ideas, places and their connections.",
                "instruction": "Extract useful, specific relationships explicitly supported by the passages."},
    "research": {"label": "Research literature", "description": "Methods, findings, datasets and limitations.",
                 "instruction": "Prioritize research questions, methods, datasets, findings and limitations. "
                                "Preserve uncertainty and attribution. A reported hypothesis is not an established fact."},
    "organizations": {"label": "Organizations & products", "description": "Companies, products, people and events.",
                      "instruction": "Prioritize organizations, products, roles, partnerships and acquisitions. "
                                     "Attach dates to their specific relationship, and distinguish plans from completed events."},
}


def profile_prompt(config):
    extraction = config.get("extraction", {})
    profile = PROFILES[extraction.get("profile", "general")]
    extra = profile["instruction"]
    if extraction.get("entity_types"):
        extra += "\nAllowed entity types: " + ", ".join(extraction["entity_types"])
    if extraction.get("predicates"):
        extra += "\nPrefer these predicates when accurate: " + ", ".join(extraction["predicates"])
    return extra
