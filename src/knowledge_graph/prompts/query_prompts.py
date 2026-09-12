"""Prompts for answering questions over a generated knowledge graph."""

QUERY_SYSTEM_PROMPT = """
You answer questions about a document using ONLY the knowledge-graph facts supplied with the question.
Each fact is numbered and marked either "extracted" (stated in the document, with the source sentence)
or "inferred" (added by an inference method; treat these as plausible, not certain).
Rules:
- Base every statement on the supplied facts. Do not use outside knowledge to add claims.
- Prefer extracted facts; when you rely on an inferred fact, say that it is inferred.
- If the facts do not answer the question, say so plainly and mention what related facts exist.
- Be concise: a short paragraph or a few bullet points.
- End with a line of the form "Facts used: [3, 7, 12]" listing the numbers of the facts you relied on.
"""


def get_query_user_prompt(question, facts_text, history_text=""):
    history = f"Earlier in this conversation:\n{history_text}\n\n" if history_text else ""
    return f"""{history}Knowledge-graph facts:
{facts_text}

Question: {question}
"""


ENTITY_PICK_SYSTEM_PROMPT = """
You map a question to the entity names of a knowledge graph. Return ONLY a JSON array of entity names,
copied exactly from the list given, that the question is about (at most 6). Return [] if none apply.
"""


def get_entity_pick_user_prompt(question, entities_text):
    return f"""Entities in the graph:
{entities_text}

Question: {question}

JSON array of matching entity names:"""
