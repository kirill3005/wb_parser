"""Интерфейсы и бэкенды LLM для RAG-пайплайна."""

from .base import LLMClient, JsonValue
from .config import ChatMessage, LLMConfig, Role
from .json_utils import build_json_prompt, generate_and_parse_json, try_parse_json
from .prompts import (
    DOC_CLASSIFICATION_PROMPT,
    EXPLANATION_PROMPT,
    EXTRACTION_SCHEMA_PROMPT,
    FACT_CHECK_PROMPT,
    MULTILABEL_CLASSIFICATION_PROMPT,
    QA_WITH_CITATIONS_PROMPT,
    QUERY_SUMMARY_WITH_CITATIONS_PROMPT,
    TIMELINE_PROMPT,
)
from .transformers import TransformersLLMClient
# from .vllm_backend import VLLMLLMClient

__all__ = [
    "LLMClient",
    "JsonValue",
    "ChatMessage",
    "LLMConfig",
    "Role",
    "build_json_prompt",
    "generate_and_parse_json",
    "try_parse_json",
    "DOC_CLASSIFICATION_PROMPT",
    "EXPLANATION_PROMPT",
    "EXTRACTION_SCHEMA_PROMPT",
    "FACT_CHECK_PROMPT",
    "MULTILABEL_CLASSIFICATION_PROMPT",
    "QA_WITH_CITATIONS_PROMPT",
    "QUERY_SUMMARY_WITH_CITATIONS_PROMPT",
    "TIMELINE_PROMPT",
    "TransformersLLMClient",
    # "VLLMLLMClient",
]
