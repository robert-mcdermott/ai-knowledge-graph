"""Phase 1: Main extraction prompts used to guide the LLM."""

MAIN_SYSTEM_PROMPT = """
You are an advanced AI system specialized in knowledge extraction and knowledge graph generation.
Your expertise includes identifying consistent entity references and meaningful relationships in text.
CRITICAL INSTRUCTION: All relationships (predicates) MUST be no more than 3 words maximum. Ideally 1-2 words. This is a hard limit.
"""

MAIN_USER_PROMPT = """
Your task: Read the text below (delimited by triple backticks) and identify all Subject-Predicate-Object (S-P-O) relationships in each sentence. Then produce a single JSON array of objects, each representing one triple.

Follow these rules carefully:

- Entity Consistency: Use consistent names for entities throughout the document. For example, if \"John Smith\" is mentioned as \"John\", \"Mr. Smith\", and \"John Smith\" in different places, use a single consistent form (preferably the most complete one) in all triples.
- Atomic Terms: Identify distinct key terms (e.g., objects, locations, organizations, acronyms, people, conditions, concepts, feelings). Avoid merging multiple ideas into one term (they should be as \"atomistic\" as possible).
- Reuse General Terms: Prefer the short, general form of a concept as the entity and express qualifiers as separate triples. Write \"computing\" rather than \"modern computing devices\", \"artificial intelligence\" rather than \"artificial intelligence development\". When a specific kind of a general concept is important (e.g. \"quantum computing\"), keep it AND add a triple linking it to the general term (\"quantum computing\" is a \"computing\"). Never wrap an entity in words like \"development of\", \"advances in\", \"the rise of\".
- Unified References: Replace any pronouns (e.g., \"he,\" \"she,\" \"it,\" \"they,\" etc.) with the actual referenced entity, if identifiable.
- Pairwise Relationships: If multiple terms co-occur in the same sentence (or a short paragraph that makes them contextually related), create one triple for each pair that has a meaningful relationship.
- CRITICAL INSTRUCTION: Predicates MUST be 1-3 words maximum. Never more than 3 words. Keep them extremely concise.
- Ensure that all possible relationships are identified in the text and are captured in an S-P-O relation.
- Standardize terminology: If the same concept appears with slight variations (e.g., \"artificial intelligence\" and \"AI\"), use the most common or canonical form consistently.
- Make all the text of S-P-O text lower-case, even Names of people and places.
- If a person is mentioned by name, create a relation to their location, profession and what they are known for (invented, wrote, started, title, etc.) if known and if it fits the context of the informaiton.

Important Considerations:
- Aim for precision in entity naming - use specific forms that distinguish between similar but different entities
- Maximize connectedness by using identical entity names for the same concepts throughout the document
- Consider the entire context when identifying entity references
- ALL PREDICATES MUST BE 3 WORDS OR FEWER - this is a hard requirement

Output Requirements:

- Do not include any text or commentary outside of the JSON.
- Return only the JSON array, with each triple as an object containing \"subject\", \"predicate\", and \"object\".
- Make sure the JSON is valid and properly formatted.

Example of the desired output structure:

[
  {
    \"subject\": \"Term A\",
    \"predicate\": \"relates to\",  // Notice: only 2 words
    \"object\": \"Term B\"
  },
  {
    \"subject\": \"Term C\",
    \"predicate\": \"uses\",  // Notice: only 1 word
    \"object\": \"Term D\"
  }
]

Important: Only output the JSON array (with the S-P-O objects) and nothing else

Text to analyze (between triple backticks):
"""


