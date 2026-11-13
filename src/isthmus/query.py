from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from typing import Protocol

from anthropic import AnthropicError
from openai import OpenAIError

from isthmus.generation import AnthropicAnswerProvider, OpenAIAnswerProvider, answer_provider
from isthmus.models import QueryPlan, QueryRequirement

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "what", "when", "where", "which", "who", "why", "with",
}
_EXCEPTION_TERMS = {
    "exception", "exceptions", "except", "unless", "qualification",
    "qualifications", "condition", "conditions", "provided", "subject",
    "waive", "waived", "waiver", "exempt", "exemption", "exclusions",
}
logger = logging.getLogger(__name__)


class QueryPlanner(Protocol):
    def plan(self, query: str) -> QueryPlan: ...


class LLMQueryPlanner:
    """Optional planner backed by a caller-provided model completion function."""

    def __init__(self, complete: Callable[[str], str]) -> None:
        self.complete = complete

    def plan(self, query: str) -> QueryPlan:
        prompt = (
            "Return JSON only with fields intent (factual, explanatory or comparative), "
            "asks_exceptions (boolean), and requirements (nonempty array of objects with "
            "id and description). Identify separate evidence requirements, including implied "
            "exceptions. Do not invent facts. Original query: " + json.dumps(query)
        )
        try:
            data = json.loads(self.complete(prompt))
            if not isinstance(data, dict):
                raise ValueError("planner response must be an object")
            intent = data.get("intent")
            asks_exceptions = data.get("asks_exceptions")
            raw_requirements = data.get("requirements")
            if intent not in {"factual", "explanatory", "comparative"} or not isinstance(asks_exceptions, bool):
                raise ValueError("planner returned an invalid intent or exception flag")
            if not isinstance(raw_requirements, list) or not 1 <= len(raw_requirements) <= 8:
                raise ValueError("planner must return one to eight requirements")
            requirements = []
            seen = set()
            for item in raw_requirements:
                if not isinstance(item, dict):
                    raise ValueError("planner requirement must be an object")
                identifier, description = item.get("id"), item.get("description")
                if not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", identifier):
                    raise ValueError("planner requirement has an invalid ID")
                if identifier in seen or not isinstance(description, str) or not description.strip() or len(description) > 500:
                    raise ValueError("planner requirement has a duplicate ID or invalid description")
                seen.add(identifier)
                requirement_terms = frozenset(_EXCEPTION_TERMS) if identifier == "exceptions" else terms(description)
                requirements.append(QueryRequirement(identifier, description.strip(), requirement_terms))
            if asks_exceptions and "exceptions" not in seen:
                requirements.append(QueryRequirement("exceptions", "Exceptions and qualifications related to " + query, frozenset(_EXCEPTION_TERMS)))
            return QueryPlan(query, intent, tuple(requirements), asks_exceptions, len(requirements) > 1)
        except (ValueError, TypeError, OpenAIError, AnthropicError) as error:
            logger.warning("Query planner failed; using deterministic planner: %s", type(error).__name__)
            return understand_query(query)


def planner_from_environment() -> QueryPlanner | None:
    model = os.getenv("ISTHMUS_QUERY_PLANNER_MODEL", "").strip()
    if not model:
        return None
    protocol = os.getenv("ISTHMUS_QUERY_PLANNER_PROTOCOL") or "openai"
    endpoint = os.getenv("ISTHMUS_QUERY_PLANNER_BASE_URL") or None
    provider = answer_provider(protocol, model, endpoint)

    def complete(prompt: str) -> str:
        if isinstance(provider, OpenAIAnswerProvider):
            response = provider.client.chat.completions.create(
                model=provider.model, messages=[{"role": "user", "content": prompt}]
            )
            text = response.choices[0].message.content
        elif isinstance(provider, AnthropicAnswerProvider):
            response = provider.client.messages.create(
                model=provider.model, max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(block.text for block in response.content if block.type == "text")
        else:
            raise TypeError("Unsupported query planner provider.")
        if not text:
            raise ValueError("Query planner returned an empty response.")
        return text

    return LLMQueryPlanner(complete)


def terms(text: str) -> frozenset[str]:
    return frozenset(
        word for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in _STOP_WORDS and len(word) > 1
    )


def understand_query(query: str) -> QueryPlan:
    normalized = query.strip()
    lowered = normalized.lower()
    asks_exceptions = bool(terms(lowered) & _EXCEPTION_TERMS) or bool(
        re.search(r"\b(?:special cases|not eligible|does not apply)\b", lowered)
    )
    is_multi_part = bool(re.search(r"\b(and|also|compare|versus|vs\.?|while)\b", lowered))
    if any(word in lowered for word in ("compare", "versus", "vs.")):
        intent = "comparative"
    elif lowered.startswith(("what is", "define", "what does")):
        intent = "factual"
    elif is_multi_part or lowered.startswith(("how", "why", "explain")):
        intent = "explanatory"
    else:
        intent = "factual"

    if intent == "comparative":
        comparison = re.sub(r"^\s*compare\s+", "", normalized, flags=re.IGNORECASE)
        parts = re.split(r"\s+(?:and|versus|vs\.?)\s+", comparison, flags=re.IGNORECASE)
    else:
        parts = re.split(
            r"\s+and\s+(?=(?:who|what|when|where|why|how|which|does|do|is|are|can|should|must)\b)",
            normalized,
            flags=re.IGNORECASE,
        )
    if len(parts) < 2 or any(not (terms(part) - _EXCEPTION_TERMS) for part in parts):
        parts = [normalized]
    requirements = [
        QueryRequirement("main" if index == 0 else f"part-{index + 1}", part, terms(part) - _EXCEPTION_TERMS)
        for index, part in enumerate(parts)
    ]
    if asks_exceptions:
        requirements.append(
            QueryRequirement("exceptions", "Exceptions and qualifications related to " + normalized, frozenset(_EXCEPTION_TERMS))
        )
    return QueryPlan(
        original_query=normalized,
        intent=intent,
        requirements=tuple(requirements),
        asks_exceptions=asks_exceptions,
        is_multi_part=is_multi_part,
    )
