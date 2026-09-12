"""Phase 3 & 4: Relationship inference prompts."""

RELATIONSHIP_INFERENCE_SYSTEM_PROMPT = """
You are an expert in knowledge representation and inference.
Your task is to infer plausible relationships between disconnected entities in a knowledge graph.
"""


def get_relationship_inference_user_prompt(entities1, entities2, triples_text):
    return f"""
I have a knowledge graph with two disconnected communities of entities.

Community 1 entities: {entities1}
Community 2 entities: {entities2}

Here are some existing relationships involving these entities:
{triples_text}

Please infer 2-3 plausible relationships between entities from Community 1 and entities from Community 2.
Return your answer as a JSON array of triples in the following format:

[
  {{
    \"subject\": \"entity from community 1\",
    \"predicate\": \"inferred relationship\",
    \"object\": \"entity from community 2\"
  }},
  ...
]

Only include highly plausible relationships with clear predicates.
IMPORTANT: The inferred relationships (predicates) MUST be no more than 3 words maximum. Preferably 1-2 words. Never more than 3.
For predicates, use short phrases that clearly describe the relationship.
IMPORTANT: Make sure the subject and object are different entities - avoid self-references.
"""


WITHIN_COMMUNITY_INFERENCE_SYSTEM_PROMPT = """
You are an expert in knowledge representation and inference.
Your task is to infer plausible relationships between semantically related entities that are not yet connected in a knowledge graph.
"""


def get_within_community_inference_user_prompt(pairs_text, triples_text):
    return f"""
I have a knowledge graph with several entities that appear to be semantically related but are not directly connected.

Here are some pairs of entities that might be related:
{pairs_text}

Here are some existing relationships involving these entities:
{triples_text}

Please infer plausible relationships between these disconnected pairs.
Return your answer as a JSON array of triples in the following format:

[
  {{
    \"subject\": \"entity1\",
    \"predicate\": \"inferred relationship\",
    \"object\": \"entity2\"
  }},
  ...
]

Only include highly plausible relationships with clear predicates.
IMPORTANT: The inferred relationships (predicates) MUST be no more than 3 words maximum. Preferably 1-2 words. Never more than 3.
IMPORTANT: Make sure that the subject and object are different entities - avoid self-references.
"""


BRIDGE_INFERENCE_SYSTEM_PROMPT = """
You are an expert in knowledge representation with broad general knowledge.
Your task is to connect small isolated groups of entities to the main body of a knowledge graph
using relationships that are well established facts, not speculation.
CRITICAL: predicates MUST be 1-3 words. Subject and object MUST be different entities taken verbatim from the lists given.
"""


def get_bridge_inference_user_prompt(main_entities, groups_text):
    return f"""
A knowledge graph has one large connected component and several small isolated groups.

Main component entities (connect TO these):
{main_entities}

Isolated groups (connect FROM these). Each group lists its entities and some of its existing relationships:
{groups_text}

For EACH isolated group, propose 1-3 relationships that link one of its entities to one of the main
component entities. Use only relationships that are well known facts or clearly implied by the
existing relationships shown. Use the entity names exactly as written above.

Return ONLY a JSON array of objects with \"subject\", \"predicate\" and \"object\":

[
  {{\"subject\": \"entity from an isolated group\", \"predicate\": \"enabled\", \"object\": \"entity from the main component\"}},
  ...
]

Rules: predicates are 1-3 words; subject and object must differ; do not invent new entity names;
skip a group if no plausible link exists.
"""


HUB_INFERENCE_SYSTEM_PROMPT = """
You are an expert in knowledge representation with broad general knowledge.
Your task is to add well known relationships between the most important entities of a knowledge graph
that the source text did not state explicitly.
CRITICAL: predicates MUST be 1-3 words. Subject and object MUST be different entities taken verbatim from the list given.
"""


def get_hub_inference_user_prompt(entities_text, existing_text, max_new):
    return f"""
Below are the most important entities of a knowledge graph, followed by the relationships between them
that are already known.

Entities:
{entities_text}

Known relationships between these entities:
{existing_text}

Using general knowledge, propose up to {max_new} additional relationships between pairs of these
entities that are widely accepted facts (for example: \"internet\" enabled \"e-commerce\";
\"steam engine\" powered \"railways\"). Do not repeat or reverse relationships already listed.
Prefer specific verbs (enabled, powered, preceded, invented, depends on) over vague ones (related to).

Return ONLY a JSON array of objects with \"subject\", \"predicate\" and \"object\", using the entity
names exactly as written above:

[
  {{\"subject\": \"entity\", \"predicate\": \"enabled\", \"object\": \"other entity\"}},
  ...
]
"""
