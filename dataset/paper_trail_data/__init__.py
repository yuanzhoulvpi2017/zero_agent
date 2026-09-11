"""PaperTrail distillation collection: virtual users and dialogue chains."""

from .chain import build_dialogue_chain
from .personas import (
    CHAIN_SCHEMA,
    MAX_CHAIN_TURNS,
    catalog,
    categories,
    move_instruction,
    persona_system_prompt,
    sample_batch,
    sample_persona,
)

__all__ = [
    "CHAIN_SCHEMA",
    "MAX_CHAIN_TURNS",
    "build_dialogue_chain",
    "catalog",
    "categories",
    "move_instruction",
    "persona_system_prompt",
    "sample_batch",
    "sample_persona",
]
