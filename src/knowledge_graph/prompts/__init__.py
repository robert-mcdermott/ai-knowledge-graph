"""Prompt factory and centralized prompt registry for the knowledge graph system."""

from .main_prompts import MAIN_SYSTEM_PROMPT, MAIN_USER_PROMPT
from .entity_prompts import (
    ENTITY_RESOLUTION_SYSTEM_PROMPT,
    get_entity_resolution_user_prompt,
)
from .inference_prompts import (
    RELATIONSHIP_INFERENCE_SYSTEM_PROMPT,
    get_relationship_inference_user_prompt,
    WITHIN_COMMUNITY_INFERENCE_SYSTEM_PROMPT,
    get_within_community_inference_user_prompt,
    BRIDGE_INFERENCE_SYSTEM_PROMPT,
    get_bridge_inference_user_prompt,
    HUB_INFERENCE_SYSTEM_PROMPT,
    get_hub_inference_user_prompt,
)

__all__ = [
    "PromptFactory",
    "prompt_factory",
    "MAIN_SYSTEM_PROMPT",
    "MAIN_USER_PROMPT",
    "ENTITY_RESOLUTION_SYSTEM_PROMPT",
    "get_entity_resolution_user_prompt",
    "RELATIONSHIP_INFERENCE_SYSTEM_PROMPT",
    "get_relationship_inference_user_prompt",
    "WITHIN_COMMUNITY_INFERENCE_SYSTEM_PROMPT",
    "get_within_community_inference_user_prompt",
    "BRIDGE_INFERENCE_SYSTEM_PROMPT",
    "get_bridge_inference_user_prompt",
    "HUB_INFERENCE_SYSTEM_PROMPT",
    "get_hub_inference_user_prompt",
]


class PromptFactory:
    """
    A general-purpose prompt registry and generator.

    - Store prompt generator functions or static strings in `self.prompts`.
    - Retrieve prompts dynamically using get_prompt(name, *args, **kwargs).
    - Supports static string templates as well as callables that accept arguments.
    """

    def __init__(self):
        self.prompts = {
            "main_system": MAIN_SYSTEM_PROMPT,
            "main_user": MAIN_USER_PROMPT,

            "entity_resolution_system": ENTITY_RESOLUTION_SYSTEM_PROMPT,
            "entity_resolution_user": get_entity_resolution_user_prompt,

            "relationship_inference_system": RELATIONSHIP_INFERENCE_SYSTEM_PROMPT,
            "relationship_inference_user": get_relationship_inference_user_prompt,
            "within_community_system": WITHIN_COMMUNITY_INFERENCE_SYSTEM_PROMPT,
            "within_community_user": get_within_community_inference_user_prompt,

            "bridge_inference_system": BRIDGE_INFERENCE_SYSTEM_PROMPT,
            "bridge_inference_user": get_bridge_inference_user_prompt,
            "hub_inference_system": HUB_INFERENCE_SYSTEM_PROMPT,
            "hub_inference_user": get_hub_inference_user_prompt,
        }

    def get_prompt(self, name: str, *args, **kwargs) -> str:
        """
        Retrieve and optionally format a prompt.

        Args:
            name (str): Registered prompt key.
            *args: Positional args passed to the prompt generator (if callable).
            **kwargs: Keyword args passed to the generator (if callable).

        Returns:
            str: The generated prompt text.
        """
        if name not in self.prompts:
            raise ValueError(f"Prompt '{name}' not found in PromptFactory.")

        prompt_entry = self.prompts[name]

        if callable(prompt_entry):
            return prompt_entry(*args, **kwargs)

        return prompt_entry


prompt_factory = PromptFactory()


